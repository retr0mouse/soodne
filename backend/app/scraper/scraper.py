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
            "Rimi": "https://www.rimi.ee/epood/ee"
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

    # Rimi scraping
    logger.info("Starting Rimi scraping...")
    get_all_rimi_items(db, store, headers, user_agent)
    logger.info("Finished Rimi scraping")


    # Barbora scraping
    logger.info("Starting Barbora scraping...")
    get_all_barbora_items(db, store, headers, user_agent)
    logger.info("Finished Barbora scraping")


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

if __name__ == "__main__":
    scrape_store_products()
