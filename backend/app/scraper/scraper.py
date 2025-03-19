import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session
from app.database.database import SessionLocal
from app import schemas
from app.services import (
    store_service,
    unit_service,
    product_service,
    product_store_data_service
)
import random
import time
import json
import urllib.robotparser
from app.core.logger import setup_logger
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from tenacity import retry, stop_after_attempt, wait_exponential
import re
from urllib.parse import urljoin
import uuid

logger = setup_logger("scraper")
logger.setLevel("DEBUG")

def random_delay(min_seconds, max_seconds):
    return random.uniform(min_seconds, max_seconds) * 0.5

def is_allowed(url, user_agent='Soodne/1.0'):
    domain = '/'.join(url.split('/')[:3])
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"{domain}/robots.txt")
    rp.read()
    return rp.can_fetch(user_agent, url)

def scrape_store_products():
    db = SessionLocal()
    try:
        logger.info("=== Starting parsing process ===")
        
        stores = {
            "Barbora": "https://barbora.ee",
            "Rimi": "https://www.rimi.ee/epood/ee",
            "Selver": "https://www.selver.ee",
            "Prisma": "https://www.prismamarket.ee"
        }
        
        for store_name, url in stores.items():
            try:
                logger.info(f"Processing store: {store_name}")
                store = store_service.get_by_name(db, name=store_name)
                if not store:
                    logger.info(f"Creating store {store_name} in database")
                    store = store_service.create(db, schemas.StoreCreate(
                        name=store_name,
                        website_url=url
                    ))
                
                process_store(db, store)
                logger.info(f"Finished processing store: {store_name}")
                
            except Exception as e:
                logger.error(f"Error processing store {store_name}: {str(e)}", exc_info=True)
                continue
                
        logger.info("=== Parsing completed successfully ===")
        
    except Exception as e:
        logger.error(f"Critical error during parsing process: {str(e)}", exc_info=True)
    finally:
        db.close()

def process_store(db: Session, store):
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    headers = {
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }

    match store.name:
        case 'Barbora':
            logger.info("Starting Barbora scraping...")
            get_all_barbora_items(db, store, headers, user_agent)
            logger.info("Finished Barbora scraping")
        case 'Rimi':
            logger.info("Starting Rimi scraping...")
            get_all_rimi_items(db, store, headers, user_agent)
            logger.info("Finished Rimi scraping")
        case 'Selver':
            logger.info("Starting Selver scraping...")
            get_all_selver_items(db, store, headers, user_agent)
            logger.info("Finished Selver scraping")
        case 'Prisma':
            logger.info("Starting Prisma scraping...")
            get_all_prisma_items(db, store, headers, user_agent)
            logger.info("Finished Prisma scraping")
        case _:
            logger.info(f"No scraper for store {store} is implemented")

def get_all_barbora_items(db: Session, store, headers, user_agent):
    logger.info("Fetching Barbora categories...")
    categories = get_barbora_categories(headers, user_agent)
    logger.info(f"Found {len(categories)} Barbora categories")
    
    for category_index, category in enumerate(categories, 1):
        logger.info(f"Processing Barbora category {category_index}/{len(categories)}: {category['title']}")
        if not category['link']:
            logger.warning(f"Skipping category {category['title']} - no link available")
            continue
            
        category_items = get_barbora_items_by_category(category, headers, user_agent)
        logger.info(f"Found {len(category_items)} items in category {category['title']}")
        
        for item_index, item in enumerate(category_items, 1):
            logger.info(f"Processing item {item_index}/{len(category_items)}: {item['name']}")
            process_item(db, store, item)

def get_barbora_categories(headers, user_agent):
    chrome_options = Options()
    chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument(f'user-agent={user_agent}')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    try:
        service = Service()
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        url = 'https://barbora.ee'
        logger.debug(f"Navigating to URL: {url}")
        driver.get(url)
        
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CLASS_NAME, "desktop-menu--parent-category-list"))
        )
        
        time.sleep(5)  
        categories = driver.find_elements(By.CSS_SELECTOR, ".desktop-menu--category a.category-item--title")
        
        result_categories = []
        for category in categories:
            try:
                title = category.find_element(By.TAG_NAME, "span").text
                link = category.get_attribute("href")
                if link:
                    link = link.replace("https://barbora.ee", "")
                    result_categories.append({
                        'title': title,
                        'link': link
                    })
                    logger.debug(f"Added category: {title} with link: {link}")
            except Exception as e:
                logger.warning(f"Error processing category element: {str(e)}")
                continue
        
        logger.info(f"Found {len(result_categories)} categories")
        return result_categories
        
    except Exception as e:
        logger.error(f"Error getting Barbora categories: {str(e)}", exc_info=True)
        return []
        
    finally:
        try:
            if 'driver' in locals():
                driver.quit()
                logger.debug("Chrome driver closed successfully")
        except Exception as e:
            logger.error(f"Error closing driver: {str(e)}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def get_barbora_items_by_category(category, headers, user_agent):
    chrome_options = Options()
    chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument(f'user-agent={user_agent}')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    result_products = []
    page = 1
    
    try:
        service = Service()
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        logger.debug("Chrome driver successfully initialized")
        
        while True:
            url = f"https://barbora.ee{category['link']}?page={page}"
            logger.debug(f"Loading page: {url}")
            
            try:
                driver.get(url)
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.ID, "CybotCookiebotDialog"))
                    )
                    driver.execute_script("document.getElementById('CybotCookiebotDialog').remove()")
                except Exception as e:
                    logger.debug(f"Cookie banner not found or couldn't be closed: {e}")

                products = driver.execute_script("return window.b_productList;")
                if len(products) == 0:
                    logger.info(f"No more products found on page {page}, stopping pagination")
                    break

                logger.debug(f"Raw products found on page {page}: {len(products)}")
                
                valid_products_count = 0
                for product in products:
                    try:
                        name = product['title']

                        weight_value, unit_name = parse_product_details(name)

                        if not product['status'] == "active":
                            return result_products

                        price = product['price']
                        image_url = product['image']

                        result_products.append({
                            'name': name,
                            'price': price,
                            'image': image_url,
                            'weight_value': weight_value,
                            'unit_name': unit_name
                        })
                        valid_products_count += 1
                        
                        logger.debug(f"""
                            Processing raw item data:
                            - Name: {name}
                            - Price: {price}
                            - Image: {image_url}
                            - Weight Value: {weight_value}
                            - Unit Name: {unit_name}
                        """)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing product: {str(e)}")
                        continue
                
                logger.debug(f"Valid products added from page {page}: {valid_products_count}")
                logger.debug(f"Current total products: {len(result_products)}")


                page += 1
                time.sleep(random_delay(1, 2))
                
            except Exception as e:
                logger.error(f"Error loading page: {str(e)}")
                break
                
    finally:
        try:
            if 'driver' in locals():
                driver.quit()
                logger.debug("Chrome driver successfully closed")
        except Exception as e:
            logger.error(f"Error while closing driver: {str(e)}")
            
    logger.debug(f"Final total products collected: {len(result_products)}")
    return result_products

def parse_product_details(name):
    weight_value = None
    unit_name = None
    
    weight_patterns = [
        r'(\d+(?:[.,]\d+)?)\s*(g|kg|ml|l|tk|tk\/pk|pk)',
        r'(\d+(?:[.,]\d+)?)(g|kg|ml|l)(?!\w)',
        r'(?:^|,\s*)(\d+)?\s*,?\s*(kg)$',
        r',\s*(\d+(?:[.,]\d+)?)\s*(g|kg|ml|l|tk|tk\/pk|pk)',
        r'(\d+)\s*kl\s*\.,\s*(kg)',
        r'(?:^|,\s*)(\d+)?\s*(tk)(?:\s|$)',
    ]
    
    for pattern in weight_patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            if len(match.groups()) == 2:
                weight_str = match.group(1)
                unit_name = match.group(2).lower()
            else:
                weight_str = '1'
                unit_name = match.group(1).lower()
            
            unit_mapping = {
                'g': 'g',
                'kg': 'kg',
                'ml': 'ml',
                'l': 'l',
                'tk': 'pc',
                'tk/pk': 'pc',
                'pk': 'pack'
            }
            unit_name = unit_mapping.get(unit_name, unit_name)
            
            try:
                if weight_str:
                    weight_value = float(weight_str.replace(',', '.'))
                else:
                    weight_value = 1.0
            except (ValueError, TypeError):
                continue
            
            break
    
    return weight_value, unit_name

def get_all_rimi_items(db: Session, store, headers, user_agent):
    logger.info("Fetching Rimi categories...")
    categories = get_rimi_categories(headers, user_agent)
    logger.info(f"Found {len(categories)} Rimi categories")
    
    for category_index, category in enumerate(categories, 1):
        logger.info(f"Processing Rimi category {category_index}/{len(categories)}: {category['title']}")
        if not category['link']:
            logger.warning(f"Skipping category {category['title']} - no link available")
            continue
            
        category_items = get_rimi_items_by_category(category, headers, user_agent)
        logger.info(f"Found {len(category_items)} items in category {category['title']}")
        
        for item_index, item in enumerate(category_items, 1):
            logger.info(f"Processing item {item_index}/{len(category_items)}: {item['name']}")
            process_item(db, store, item)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def get_rimi_categories(headers, user_agent):
    chrome_options = Options()
    chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument(f'user-agent={user_agent}')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    categories = []
    driver = None
    
    try:
        logger.debug("Initializing Chrome driver...")
        service = Service()
        driver = webdriver.Chrome(service=service, options=chrome_options)
        logger.debug("Chrome driver initialized successfully")
        
        url = 'https://www.rimi.ee/epood/ee'
        logger.debug(f"Navigating to URL: {url}")
        driver.get(url)
        
        time.sleep(10)
        
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".category-menu button.trigger"))
        )
        
        category_elements = driver.find_elements(By.CSS_SELECTOR, ".category-menu button.trigger")
        logger.debug(f"Found {len(category_elements)} category elements")
        
        for element in category_elements:
            try:
                href = element.get_attribute('href')
                name = element.find_element(By.CSS_SELECTOR, "span.name").text.strip()
                
                if href and name:
                    category_id = href.split('/c/')[-1] if '/c/' in href else None
                    
                    if category_id:
                        categories.append({
                            'title': name,
                            'link': href,
                            'id': category_id.strip()
                        })
                        logger.debug(f"Added category: {name} (ID: {category_id})")
            except Exception as e:
                logger.warning(f"Error processing category element: {str(e)}")
                continue

        if not categories:
            logger.debug("Trying JavaScript method to get categories")
            categories_js = driver.execute_script("""
                return Array.from(document.querySelectorAll('.category-menu button.trigger')).map(el => {
                    const href = el.getAttribute('href');
                    const categoryId = href ? href.split('/c/').pop().trim() : null;
                    return {
                        title: el.querySelector('span.name').textContent.trim(),
                        link: href,
                        id: categoryId
                    };
                }).filter(cat => cat.title && cat.link && cat.id);
            """)
            
            if categories_js:
                categories = categories_js
                logger.debug(f"Found {len(categories)} categories using JavaScript")
        
        return categories
        
    except Exception as e:
        logger.error(f"Error getting Rimi categories: {str(e)}", exc_info=True)
        return []
        
    finally:
        try:
            if driver:
                driver.quit()
                logger.debug("Chrome driver closed successfully")
        except Exception as e:
            logger.error(f"Error closing driver: {str(e)}")

def get_rimi_items_by_category(category, headers, user_agent):
    chrome_options = Options()
    chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument(f'user-agent={user_agent}')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    result_products = []
    page = 1

    try:
        service = Service()
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        logger.debug("Chrome driver successfully initialized")

        while True:
            url = f"https://www.rimi.ee{category['link']}?pageSize=100&currentPage={page}"
            logger.debug(f"Loading page: {url}")

            try:
                driver.get(url)
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.ID, "CybotCookiebotDialog"))
                    )
                    driver.execute_script("document.getElementById('CybotCookiebotDialog').remove()")
                except Exception as e:
                    logger.debug(f"Cookie banner not found or couldn't be closed: {e}")

                products = driver.execute_script('''
                    const targetScript = Array.from(document.querySelectorAll("script")).find(s => 
                        s.textContent.includes('dataLayer.push') && 
                        s.textContent.includes('"impressions"') &&
                        s.textContent.includes('ecommerce')
                    );
                
                    const jsonStr = targetScript.textContent
                        .replace(/\s+/g, ' ')  // Collapse whitespace
                        .match(/dataLayer\.push\(\s*({.*?})\s*\)/)[1]  // Capture JSON
                        .replace(/'/g, '"')     // Standardize quotes
                        .replace(/([{,])(\s*)([A-Za-z_]+)(\s*):/g, '$1"$3":')  // Fix keys
                        .replace(/,\s*}/g, '}');  // Remove trailing commas
                
                    return JSON.parse(jsonStr).ecommerce.impressions;
                ''')

                if len(products) == 0:
                    logger.info(f"No more products found on page {page}, stopping pagination")
                    break

                logger.debug(f"Raw products found on page {page}: {len(products)}")

                valid_products_count = 0
                for product in products:
                    try:
                        name = product['name']

                        weight_value, unit_name = parse_product_details(name)

                        price = product['price']
                        image_url = f"https://rimibaltic-res.cloudinary.com/image/upload/b_white,c_limit,dpr_3.0,f_auto,q_auto:low,w_250/d_ecommerce:backend-fallback.png/MAT_{product['id']}_PCE_EE"

                        result_products.append({
                            'name': name,
                            'price': price,
                            'image': image_url,
                            'weight_value': weight_value,
                            'unit_name': unit_name
                        })
                        valid_products_count += 1

                        logger.debug(f"""
                            Processing raw item data:
                            - Name: {name}
                            - Price: {price}
                            - Image: {image_url}
                            - Weight Value: {weight_value}
                            - Unit Name: {unit_name}
                        """)

                    except Exception as e:
                        logger.warning(f"Error parsing product: {str(e)}")
                        continue

                logger.debug(f"Valid products added from page {page}: {valid_products_count}")
                logger.debug(f"Current total products: {len(result_products)}")


                page += 1
                time.sleep(random_delay(1, 2))

            except Exception as e:
                logger.error(f"Error loading page: {str(e)}")
                break

    finally:
        try:
            if 'driver' in locals():
                driver.quit()
                logger.debug("Chrome driver successfully closed")
        except Exception as e:
            logger.error(f"Error while closing driver: {str(e)}")

    logger.debug(f"Final total products collected: {len(result_products)}")
    return result_products

def process_item(db: Session, store, item):
    logger.debug(f"""
    Processing raw item data:
    - Name: {item['name']}
    - Price: {item['price']}
    - Image: {item['image']}
    - Category: {item.get('category', 'N/A')}
    - Weight Value: {item.get('weight_value', 'N/A')}
    - Unit Name: {item.get('unit_name', 'N/A')}
    """)

    weight_value = item.get('weight_value')
    unit_name = item.get('unit_name')
    
    if weight_value and unit_name:
        unit = unit_service.get_by_name(db, name=unit_name)
        if not unit:
            unit_data = schemas.UnitCreate(
                name=unit_name,
                conversion_factor=get_conversion_factor(unit_name)
            )
            unit = unit_service.create(db, unit=unit_data)
            logger.debug(f"Created new unit: {unit_name} with factor {get_conversion_factor(unit_name)}")
    else:
        unit = None
        logger.debug("No unit information available for this product")

    product_data = schemas.ProductCreate(
        name=item['name'],
        image_url=item['image'],
        weight_value=weight_value,
        unit_id=unit.unit_id if unit else None
    )
    
    product = product_service.get_by_name_and_unit(db, name=item['name'], unit_id=unit.unit_id if unit else None)
    if not product:
        product = product_service.create(db, product=product_data)
        logger.debug(f"Created new product: {product.name} (ID: {product.product_id})")
    else:
        logger.debug(f"Found existing product: {product.name} (ID: {product.product_id})")

    psd_data = schemas.ProductStoreDataCreate(
        product_id=product.product_id,
        store_id=store.store_id,
        price=item['price'],
        store_product_name=item['name'],
        store_image_url=item['image'],
        store_weight_value=weight_value,
        store_unit_id=unit.unit_id if unit else None
    )

    existing_psd = product_store_data_service.get_by_product_and_store(
        db, 
        product_id=product.product_id, 
        store_id=store.store_id
    )
    
    if existing_psd:
        old_price = existing_psd.price
        existing_psd.price = item['price']
        existing_psd.last_updated = time.strftime('%Y-%m-%d %H:%M:%S')
        db.commit()
        logger.debug(f"""
        Updated existing store data:
        - Product: {item['name']}
        - Old price: {old_price}
        - New price: {item['price']}
        - Last updated: {existing_psd.last_updated}
        """)
    else:
        new_psd = product_store_data_service.create(db, psd=psd_data)
        logger.debug(f"""
        Created new store data:
        - Product: {item['name']}
        - Price: {item['price']}
        - Store: {store.name}
        - Product ID: {new_psd.product_id}
        """)

    logger.debug("=" * 50)

def parse_weight(weight_str):
    if not weight_str:
        return None, None
    parts = weight_str.strip().split()
    if len(parts) != 2:
        return None, None
    try:
        value = float(parts[0].replace(',', '.'))
        unit = parts[1].lower()
        return value, unit 
    except ValueError:
        return None, None

def get_conversion_factor(unit_name):
    unit_conversions = {
        'g': 1,
        'kg': 1000,
        'ml': 1,
        'l': 1000,
    }
    return unit_conversions.get(unit_name, 1)

def get_all_selver_items(db: Session, store, headers, user_agent):
    logger.info("Fetching Selver categories...")
    categories = get_selver_categories(headers, user_agent)
    logger.info(f"Found {len(categories)} Selver categories")
    
    for category_index, category in enumerate(categories, 1):
        logger.info(f"Processing Selver category {category_index}/{len(categories)}: {category['name']}")
        if not category['url_path']:
            logger.warning(f"Skipping category {category['name']} - no URL path available")
            continue
            
        category_items = get_selver_items_by_category(category, headers, user_agent)
        logger.info(f"Found {len(category_items)} items in category {category['name']}")
        
        for item_index, item in enumerate(category_items, 1):
            logger.info(f"Processing item {item_index}/{len(category_items)}: {item['name']}")
            process_item(db, store, item)

def get_selver_categories(headers, user_agent):
    try:
        categories_api_url = "https://www.selver.ee/api/catalog/vue_storefront_catalog_et/category/_search?from=0&request=%7B%22query%22%3A%7B%22bool%22%3A%7B%22filter%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22terms%22%3A%7B%22id%22%3A%5B3%5D%7D%7D%2C%7B%22terms%22%3A%7B%22is_active%22%3A%5Btrue%5D%7D%7D%5D%7D%7D%7D%7D%7D&size=4000&sort=position%3Aasc"
        logger.debug(f"Requesting Selver categories from: {categories_api_url}")
        
        response = requests.get(categories_api_url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        main_categories = []
        
        if 'hits' in data and 'hits' in data['hits']:
            for hit in data['hits']['hits']:
                source = hit.get('_source', {})
                
                # Skip categories with include_in_menu=0 or any system categories
                if not source.get('include_in_menu', 0) or not source.get('is_active', False):
                    continue
                
                # Get only level 3 categories (main categories)
                if source.get('level') == 3:
                    category = {
                        'id': source.get('id'),
                        'name': source.get('name'),
                        'url_path': source.get('url_path'),
                        'product_count': source.get('product_count', 0)
                    }
                    main_categories.append(category)
                    logger.debug(f"Added main category: {category['name']} with URL path: {category['url_path']}")
        
        logger.info(f"Found {len(main_categories)} main categories from Selver")
        return main_categories
    except Exception as e:
        logger.error(f"Error getting Selver categories: {str(e)}", exc_info=True)
        return []

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def get_selver_items_by_category(category, headers, user_agent):
    result_products = []
    page = 1
    page_size = 1000  # Set a large page size to get all products at once
    
    try:
        category_id = category['id']
        # Format the API URL to query products by category_id
        api_url = f"https://www.selver.ee/api/catalog/vue_storefront_catalog_et/product/_search?_source_exclude=configurable_options%2Cproduct_nutr_info%2Cproduct_nutr_unit%2Cproduct_nutr_energy%2Cproduct_nutr_fats%2Cproduct_nutr_fats_acids%2Cproduct_nutr_carbohydrates%2Cproduct_nutr_sugars%2Cproduct_nutr_proteins%2Cproduct_nutr_salt%2Csgn%2C%2A.sgn%2Cmsrp_display_actual_price_type%2C%2A.msrp_display_actual_price_type%2Crequired_options&_source_include=documents%2Cactivity%2Cconfigurable_children.attributes%2Cconfigurable_children.id%2Cconfigurable_children.final_price%2Cconfigurable_children.color%2Cconfigurable_children.original_price%2Cconfigurable_children.original_price_incl_tax%2Cconfigurable_children.price%2Cconfigurable_children.price_incl_tax%2Cconfigurable_children.size%2Cconfigurable_children.sku%2Cconfigurable_children.special_price%2Cconfigurable_children.special_price_incl_tax%2Cconfigurable_children.tier_prices%2Cfinal_price%2Cid%2Cimage%2Cname%2Cnew%2Coriginal_price_incl_tax%2Coriginal_price%2Cprice%2Cprice_incl_tax%2Cproduct_links%2Csale%2Cspecial_price%2Cspecial_to_date%2Cspecial_from_date%2Cspecial_price_incl_tax%2Cstatus%2Ctax_class_id%2Ctier_prices%2Ctype_id%2Curl_path%2Curl_key%2C%2Aimage%2C%2Asku%2C%2Asmall_image%2Cshort_description%2Cmanufacturer%2Cproduct_%2A%2Cextension_attributes.deposit_data%2Cstock%2Cproduct_stocktype%2Cproduct_stocksource%2Cprices%2Cvmo_badges%2Cproduct_nutr_energy_kcal&from=0&request=%7B%22query%22%3A%7B%22bool%22%3A%7B%22filter%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22terms%22%3A%7B%22visibility%22%3A%5B2%2C3%2C4%5D%7D%7D%2C%7B%22terms%22%3A%7B%22status%22%3A%5B0%2C1%5D%7D%7D%2C%7B%22terms%22%3A%7B%22category_ids%22%3A%5B{category_id}%5D%7D%7D%5D%7D%7D%7D%7D%7D&size={page_size}&sort=position%3Aasc"
        
        logger.debug(f"Requesting Selver products from API for category: {category['name']} (ID: {category_id})")
        
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        if 'hits' in data and 'hits' in data['hits']:
            products_data = data['hits']['hits']
            logger.debug(f"Found {len(products_data)} products from API for category: {category['name']}")
            
            for product in products_data:
                try:
                    source = product.get('_source', {})
                    
                    # Skip products that are not active
                    if source.get('status') != 1:
                        continue
                    
                    name = source.get('name')
                    if not name:
                        continue
                    
                    # Extract price from prices array (prefer customer_group_id 0 which is the default)
                    price = source.get('price', 0)
                    for price_item in source.get('prices', []):
                        if price_item.get('customer_group_id') == 0:
                            price = price_item.get('final_price')
                            break
                    
                    # Get image URL
                    image_path = source.get('image')
                    image_url = f"https://www.selver.ee/img/800/800/resize{image_path}" if image_path else None
                    
                    # Extract weight and unit from product name or volume field
                    weight_value, unit_name = parse_product_details(name)
                    
                    # If no weight/unit found, try from product_volume field
                    if (not weight_value or not unit_name) and source.get('product_volume'):
                        weight_value, unit_name = parse_product_details(source.get('product_volume'))
                    
                    # Create product data
                    result_products.append({
                        'name': name,
                        'price': price,
                        'image': image_url,
                        'weight_value': weight_value,
                        'unit_name': unit_name
                    })
                    
                    logger.debug(f"""
                        Processing raw item data:
                        - Name: {name}
                        - Price: {price}
                        - Image: {image_url}
                        - Weight Value: {weight_value}
                        - Unit Name: {unit_name}
                    """)
                    
                except Exception as e:
                    logger.warning(f"Error parsing product: {str(e)}")
                    continue
        
    except Exception as e:
        logger.error(f"Error fetching products from Selver API: {str(e)}", exc_info=True)
    
    logger.debug(f"Final total products collected for category {category['name']}: {len(result_products)}")
    return result_products

def get_all_prisma_items(db: Session, store, headers, user_agent):
    logger.info("Fetching Prisma categories...")
    categories = get_prisma_categories(headers, user_agent)
    logger.info(f"Found {len(categories)} Prisma categories")
    
    for category_index, category in enumerate(categories, 1):
        logger.info(f"Processing Prisma category {category_index}/{len(categories)}: {category['name']}")
        if not category['link']:
            logger.warning(f"Skipping category {category['name']} - no link available")
            continue
            
        category_items = get_prisma_items_by_category(category, headers, user_agent)
        logger.info(f"Found {len(category_items)} items in category {category['name']}")
        
        for item_index, item in enumerate(category_items, 1):
            logger.info(f"Processing item {item_index}/{len(category_items)}: {item['name']}")
            process_item(db, store, item)

def get_prisma_categories(headers, user_agent):
    try:
        # Make a request to the Prisma homepage to get the store ID
        url = "https://www.prismamarket.ee/"
        logger.debug(f"Requesting Prisma homepage: {url}")
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Look for the Apollo state data containing store information
        html_content = response.text
        
        # Find the Apollo state data that contains store information
        apollo_state_pattern = r'<script id="__APOLLO_STATE__" type="application/json">(.*?)</script>'
        apollo_state_match = re.search(apollo_state_pattern, html_content, re.DOTALL)
        
        if not apollo_state_match:
            logger.warning("Could not find Apollo state data in Prisma homepage")
            return []
            
        apollo_state_json = apollo_state_match.group(1)
        apollo_state = json.loads(apollo_state_json)
        
        # Find the store ID
        store_id = None
        for key in apollo_state:
            if 'store(' in key and '"id"' in key:
                store_id_match = re.search(r'store\(\{\"id\":\"(\d+)\"\}\)', key)
                if store_id_match:
                    store_id = store_id_match.group(1)
                    logger.debug(f"Found Prisma store ID: {store_id}")
                    break
        
        if not store_id:
            store_id = "542860184"  # Default store ID if not found
            logger.debug(f"Using default Prisma store ID: {store_id}")
        
        # Now use the specific GraphQL API call for categories
        api_url = "https://graphql-api.prismamarket.ee/"
        
        # Prepare the query parameters exactly as provided
        graphql_params = {
            "operationName": "RemoteGetCategoryNavigationMenuContent",
            "variables": json.dumps({
                "userConsent": {"tracking": None},
                "preview": False,
                "storeId": store_id
            }),
            "extensions": json.dumps({
                "persistedQuery": {
                    "version": 1,
                    "sha256Hash": "251b0aac3125368e16e78bb4eb71c26dc765cbf57af6ef48de91bef7e038a032"
                }
            })
        }
        
        # Setup headers for the GraphQL request
        graphql_headers = {
            'User-Agent': user_agent,
            'Accept': '*/*',
            'Accept-Language': 'et',
            'Content-Type': 'application/json',
            'Origin': 'https://www.prismamarket.ee',
            'Referer': 'https://www.prismamarket.ee/',
            'sec-ch-ua': '"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'x-client-name': 'skaupat-web',
            'x-client-version': 'production-1a3b473e0b939473b0810fc73dea567de2c4e3a9'
        }
        
        logger.debug(f"Requesting Prisma categories from GraphQL API with store ID: {store_id}")
        
        # Make the API request
        response = requests.get(api_url, params=graphql_params, headers=graphql_headers)
        response.raise_for_status()
        
        categories_data = response.json()
        logger.debug(f"Received categories data from GraphQL API: {json.dumps(categories_data)[:200]}...")
        
        # Extract categories from the response
        categories = []
        
        # Extract navigation items from the GraphQL response
        if 'data' in categories_data and 'categoryNavigationMenu' in categories_data['data']:
            nav_items = categories_data['data']['categoryNavigationMenu'].get('items', [])
            
            for item in nav_items:
                if 'id' in item and 'name' in item and 'slug' in item:
                    category = {
                        'id': item['id'],
                        'name': item['name'],
                        'slug': item['slug'],
                        'link': f"/kategooriad/{item['slug']}",
                        'store_id': store_id
                    }
                    categories.append(category)
                    logger.debug(f"Added Prisma category: {category['name']} with slug: {category['slug']}")
        
        # Filter to focus on food-related categories if needed
        if categories:
            logger.info(f"Found {len(categories)} main categories from Prisma")
            return categories
        else:
            # If GraphQL API didn't work, fallback to extracting from Apollo state
            logger.warning("Couldn't get categories from GraphQL API, falling back to Apollo state extraction")
            
            # Extract main navigation items from Apollo state
            main_navigation_key = None
            
            # Find the main navigation key
            for key in apollo_state:
                if 'mainNavigation' in key and 'navigationItems' in apollo_state[key]:
                    main_navigation_key = key
                    break
            
            if not main_navigation_key:
                logger.warning("Could not find main navigation data in Apollo state")
                return []
                
            # Get navigation item references
            nav_item_refs = apollo_state[main_navigation_key]['navigationItems']
            
            # Extract category data from each navigation item reference
            for item_ref in nav_item_refs:
                ref = item_ref.get('__ref')
                if ref and ref in apollo_state:
                    item_data = apollo_state[ref]
                    
                    # Only consider categories with necessary data
                    if 'id' in item_data and 'name' in item_data and 'link' in item_data:
                        category = {
                            'id': item_data['id'],
                            'name': item_data['name'],
                            'slug': item_data.get('slug', item_data['link'].split('/')[-1]),
                            'link': item_data['link'],
                            'store_id': store_id
                        }
                        categories.append(category)
                        logger.debug(f"Added Prisma category from Apollo state: {category['name']} with link: {category['link']}")
            
            logger.info(f"Found {len(categories)} categories from Apollo state")
            return categories
        
    except Exception as e:
        logger.error(f"Error getting Prisma categories: {str(e)}", exc_info=True)
        return []

def get_prisma_items_by_category(category, headers, user_agent):
    result_products = []
    
    try:
        # Get category slug from the link
        slug = category['slug'] if 'slug' in category else category['link'].split('/')[-1]
        
        # Get store ID from category data or use default
        store_id = category.get('store_id', '542860184')
        
        # Set up the referrer URL for this category
        category_url = f"https://www.prismamarket.ee{category['link']}"
        logger.debug(f"Processing category: {category['name']} (slug: {slug}, store_id: {store_id})")
        
        # First visit the category page to extract the needed hash value
        response = requests.get(category_url, headers=headers)
        response.raise_for_status()
        
        html_content = response.text
        
        # Extract the hash for product queries
        hash_pattern = r'sha256Hash":"([a-f0-9]+)".*?RemoteFilteredProducts'
        hash_match = re.search(hash_pattern, html_content)
        
        if not hash_match:
            logger.error(f"Could not find hash parameter for RemoteFilteredProducts on category page: {category_url}")
            return result_products
            
        products_hash = hash_match.group(1)
        logger.debug(f"Extracted sha256Hash for products: {products_hash}")
        
        # Generate a random session ID
        session_id = str(uuid.uuid4())
        
        # GraphQL API endpoint
        api_url = "https://graphql-api.prismamarket.ee/"
        
        # Use a large limit to get many products at once (1000)
        limit = 1000
        from_index = 0
        total_items = None
        
        # Setup proper headers for the GraphQL request
        graphql_headers = {
            'User-Agent': user_agent,
            'Accept': '*/*',
            'Accept-Language': 'et',
            'Content-Type': 'application/json',
            'Origin': 'https://www.prismamarket.ee',
            'Referer': category_url,
            'sec-ch-ua': '"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'x-client-name': 'skaupat-web',
            'x-client-version': 'production-1a3b473e0b939473b0810fc73dea567de2c4e3a9'
        }
        
        logger.debug(f"Fetching products for category {category['name']} with session ID: {session_id}")
        
        while total_items is None or from_index < total_items:
            # Variables for the GraphQL query
            variables = {
                "facets": [
                    {"key": "brandName", "order": "asc"},
                    {"key": "labels"}
                ],
                "generatedSessionId": session_id,
                "includeAgeLimitedByAlcohol": True,
                "limit": limit,
                "queryString": "",
                "searchProvider": "loop54",
                "slug": slug,
                "storeId": store_id,
                "useRandomId": False,
                "from": from_index
            }
            
            # Construct the GraphQL request
            payload = {
                "operationName": "RemoteFilteredProducts",
                "variables": variables,
                "extensions": {
                    "persistedQuery": {
                        "version": 1,
                        "sha256Hash": products_hash
                    }
                }
            }
            
            # Make the request
            response = requests.post(api_url, headers=graphql_headers, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Error fetching products from Prisma API: {response.status_code} - {response.text}")
                break
                
            data = response.json()
            
            # Extract products data
            if 'data' in data and 'store' in data['data'] and 'products' in data['data']['store']:
                products_data = data['data']['store']['products']
                items = products_data.get('items', [])
                
                # Update total for pagination
                if total_items is None:
                    total_items = products_data.get('total', 0)
                    logger.debug(f"Total products in category {category['name']}: {total_items}")
                
                # Process each product
                for item in items:
                    try:
                        name = item.get('name', '')
                        price = item.get('price', 0)
                        
                        # Get image URL
                        image_url = None
                        if 'images' in item and item['images'] and len(item['images']) > 0:
                            image_url = item['images'][0].get('url', '')
                        
                        # Extract weight and unit from product name
                        weight_value, unit_name = parse_product_details(name)
                        
                        result_products.append({
                            'name': name,
                            'price': price,
                            'image': image_url,
                            'weight_value': weight_value,
                            'unit_name': unit_name
                        })
                        
                        logger.debug(f"""
                            Processing raw item data:
                            - Name: {name}
                            - Price: {price}
                            - Image: {image_url}
                            - Weight Value: {weight_value}
                            - Unit Name: {unit_name}
                        """)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing product: {str(e)}")
                        continue
                
                # Increment from_index for next page
                from_index += len(items)
                
                # Break if we got fewer items than requested (last page)
                if len(items) < limit:
                    break
                    
                # Add a small delay between requests
                time.sleep(random_delay(0.5, 1))
            else:
                logger.warning("Unexpected response structure from Prisma API")
                break
                
    except Exception as e:
        logger.error(f"Error in get_prisma_items_by_category: {str(e)}", exc_info=True)
    
    logger.debug(f"Total products collected for category {category['name']}: {len(result_products)}")
    return result_products
