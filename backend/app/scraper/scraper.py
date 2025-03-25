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
                        .replace(/\s+/g, ' ')
                        .match(/dataLayer\.push\(\s*({.*?})\s*\)/)[1]
                        .replace(/'/g, '"')
                        .replace(/([{,])(\s*)([A-Za-z_]+)(\s*):/g, '$1"$3":')
                        .replace(/,\s*}/g, '}');
                
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

    for category_index, category in enumerate(categories, 1):
        logger.info(f"Processing Selver category {category_index}/{len(categories)}: {category['name']}")

        category_items = get_selver_items_by_category(category, headers, user_agent)
        logger.info(f"Found {len(category_items)} items in category {category['name']}")

        for item_index, item in enumerate(category_items, 1):
            logger.info(f"Processing item {item_index}/{len(category_items)}: {item['name']}")
            process_item(db, store, item)

def get_selver_categories(headers, user_agent):
    try:
        categories_api_url = "https://www.selver.ee/api/catalog/vue_storefront_catalog_et/category/_search?q=parent_id:3%20AND%20-_exists_:display_mode&_source_include=name,id,children_data,url_path&size=1000"
        logger.debug(f"Requesting Selver categories from: {categories_api_url}")

        response = requests.get(categories_api_url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        categories = []

        
        if 'hits' in data and 'hits' in data['hits']:
            for hit in data['hits']['hits']:
                current_category = hit['_source']

                category = {
                    'name': current_category.get('name'),
                    'url_path': current_category.get('url_path'),
                    'id': current_category.get('id'),
                    'all_categories': getSubcategories(current_category)
                }
                categories.append(category)
                logger.debug(f"Added category: {category['name']} with URL path: {category['url_path']}")
        
        logger.info(f"Found {len(categories)} categories from Selver")
        return categories
    except Exception as e:
        logger.error(f"Error getting Selver categories: {str(e)}", exc_info=True)
        return []


# returns an array containing parent category and all subcategory ids
def getSubcategories(category):
    category_ids = [category['id']]

    if category['children_data']:
        for child in category['children_data']:
            category_ids.extend(getSubcategories(child))
    return category_ids

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def get_selver_items_by_category(category, headers, user_agent):
    result_products = []
    
    try:
        api_url = "https://www.selver.ee/api/catalog/vue_storefront_catalog_et/product/_search"
        payload = {
            "_source": ["name", "prices", "media_gallery", "product_volume"],
            "query": {
                "bool": {
                    "filter": {
                        "bool": {
                            "must": [
                                {"terms": {"category_ids": category['all_categories']}}
                            ]
                        }
                    }
                }
            },
            "size": 10000
        }

        logger.debug(f"Requesting Selver products from API for category: {category['name']} (ID: {category['id']})")
        
        response = requests.post(api_url, json=payload, headers=headers)
        
        data = response.json()
        
        if 'hits' in data and 'hits' in data['hits']:
            products_data = data['hits']['hits']
            logger.debug(f"Found {len(products_data)} products from API for category: {category['name']}")
            
            for product in products_data:
                try:
                    current_product = product.get('_source')
                    name = current_product.get('name')
                    price = current_product['prices'][0]['final_price']
                    image_path = current_product['media_gallery'][0]['image']
                    image_url = f"https://www.selver.ee/img/800/800/resize{image_path}" if image_path else None
                    weight_value, unit_name = parse_product_details(current_product['product_volume'])

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
        category_items = get_prisma_items_by_category(category, headers, user_agent)
        logger.info(f"Found {len(category_items)} items in category {category['name']}")
        
        for item_index, item in enumerate(category_items, 1):
            logger.info(f"Processing item {item_index}/{len(category_items)}: {item['name']}")
            process_item(db, store, item)

def get_prisma_categories(headers, user_agent):
    try:
        base_url = "https://graphql-api.prismamarket.ee"
        body = {
            "operationName": "RemoteNavigation",
            "variables": {
                "id":"542860184"
            },
            "extensions": {
                "persistedQuery": {
                    "version":1,
                    "sha256Hash": "707a9c68de67bcde9992a5d135e696c61d48abe1a9c765ca73ecf07bd80c513f"
                }
            }
        }
        logger.debug(f"Requesting Prisma categories")
        
        response = requests.post(base_url, json=body, headers=headers)
        response.raise_for_status()

        data = response.json()
        categories = []

        if 'data' in data and 'store' in data['data'] and 'navigation' in data['data']['store']:
            # skip the first 2 categories ("Aktuaalne", "Food Market")
            data['data']['store']['navigation'].pop(0)
            data['data']['store']['navigation'].pop(0)
            for current_category in data['data']['store']['navigation']:
                category = {
                    'name': current_category.get('name'),
                    'slug': current_category.get('slug'),
                    'id': current_category.get('id')
                }
                categories.append(category)
                logger.debug(f"Added category: {category['name']} ")

        logger.info(f"Found {len(categories)} categories from Prisma")
        return categories
        
    except Exception as e:
        logger.error(f"Error getting Prisma categories: {str(e)}", exc_info=True)
        return []

def get_prisma_items_by_category(category, headers, user_agent):
    result_products = []
    
    try:
        offset = 0
        while True:
            base_url = "https://graphql-api.prismamarket.ee"
            body = {
                "operationName": "RemoteFilteredProducts",
                "variables": {
                    "includeAgeLimitedByAlcohol": True,
                    "limit": 120,
                    "from": offset,
                    "queryString": "",
                    "searchProvider": "loop54",
                    "slug": category['slug'],
                    "storeId": "542860184"
                },
                "extensions": {
                    "persistedQuery": {
                        "version": 1,
                        "sha256Hash": "86214929199d2277cbe0a8c138b2be4db7d5b32df8399bd3d266377ffc9c29b4"
                    }
                }
            }

            logger.debug(f"Requesting Prisma products from API for category: {category['name']} (ID: {category['id']}, SLUG: {category['slug']})")

            response = requests.post(base_url, json=body, headers=headers)
            response.raise_for_status()

            data = response.json()
            products = data['data']['store']['products']['items']
            if len(products) > 0:
                logger.debug(f"Found {len(products)} products from API for category: {category['name']}")

                for product in products:
                    try:
                        name = product.get('name')
                        price = product['price']
                        image_url = (product['productDetails']['productImages']['mainImage']['urlTemplate']
                                      .replace('/{MODIFIERS}', '')
                                      .replace('{EXTENSION}', 'png'))
                        weight_value, unit_name = parse_product_details(product['name'])

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
                offset = offset + 120
            else:
                break
    except Exception as e:
        logger.error(f"Error in get_prisma_items_by_category: {str(e)}", exc_info=True)
    
    logger.debug(f"Total products collected for category {category['name']}: {len(result_products)}")
    return result_products
