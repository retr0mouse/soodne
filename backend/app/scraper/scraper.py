import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session
from app.database.database import SessionLocal
from app import schemas
from app.services import (
    store_service,
    unit_service,
    product_service,
    product_store_data_service,
    product_price_history_service
)
from app.models.product_price_history import ProductPriceHistory
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
from app.services.category_service import category_service

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
            # "Barbora": "https://barbora.ee",
            "Rimi": "https://www.rimi.ee/epood/ee",
            # "Selver": "https://www.selver.ee",
            # "Prisma": "https://www.prismamarket.ee"
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
    # logger.info("Fetching Rimi categories...")
    # categories = get_rimi_categories(headers, user_agent)
    # add_rimi_categories(db, categories)
    top_categories = category_service.get_top_categories(db)
    logger.info(f"Found {len(top_categories)} Rimi categories")

    for category_index, category in enumerate(top_categories, 1):
        logger.info(f"Processing Rimi category {category_index}/{len(top_categories)}: {category.name}")
        category_items = get_rimi_items_by_category(category, headers, user_agent)
        logger.info(f"Found {len(category_items)} items in category {category.name}")

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
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        logger.debug("Chrome driver initialized successfully")

        url = 'https://www.rimi.ee/epood/ee'

        driver.get(url)

        # Wait for page to be fully loaded
        WebDriverWait(driver, 30).until(
            EC.visibility_of_element_located((By.TAG_NAME, "body"))
        )

        # Try to handle cookie dialog if it appears
        try:
            cookie_dialog = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.ID, "CybotCookiebotDialogBodyButtonDecline"))
            )
            cookie_dialog.click()
        except Exception as e:
            logger.debug(f"Cookie dialog handling: {str(e)}")

        driver.find_element(By.CLASS_NAME, 'category-menu').click()

        categories_selector = "category-list-item"
        categories = []

        def find_rimi_categories_recursively(depth = 0):
            menu = driver.find_elements(By.CLASS_NAME, 'category-menu')[depth]
            found_categories = menu.find_elements(By.CLASS_NAME, categories_selector)
            nested_categories = []

            for category in found_categories:
                found_text_elements = category.find_elements(By.CSS_SELECTOR, "span.name")
                if not len(found_text_elements): continue

                category_name = found_text_elements[0].text.strip()
                category_dict = {"name": category_name, "subcategories": []}

                found_button_elements = category.find_elements(By.TAG_NAME, 'button')
                if found_button_elements:
                    found_button_elements[0].click()
                    category_dict["subcategories"] = find_rimi_categories_recursively(depth + 1)
                    category_dict["url"] = found_button_elements[0].get_attribute('href')

                nested_categories.append(category_dict)

            return nested_categories

        categories = find_rimi_categories_recursively()

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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
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
    max_pages = 5  # Limit to prevent infinite loops

    try:
        service = Service()
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        logger.debug("Chrome driver successfully initialized")

        while page <= max_pages:  # Add a maximum page limit to prevent infinite loops
            url = f"https://www.rimi.ee{category.url}?pageSize=100&currentPage={page}"
            logger.debug(f"Loading page: {url}")

            try:
                driver.get(url)
                
                # Wait for page to at least partially load
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # Handle cookie banner more reliably
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.ID, "CybotCookiebotDialog"))
                    )
                    driver.execute_script("document.getElementById('CybotCookiebotDialog').remove()")
                    logger.debug("Cookie banner removed")
                except Exception as e:
                    logger.debug(f"Cookie banner not found or couldn't be closed: {e}")

                # Scroll down to trigger lazy loading
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight/3);")
                time.sleep(2)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight*2/3);")
                time.sleep(2)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)

                # Try multiple extraction methods in sequence
                products = []
                extraction_methods = [
                    # Method 1: DataLayer extraction
                    '''
                    const targetScript = Array.from(document.querySelectorAll("script")).find(s => 
                        s.textContent.includes('dataLayer.push') && 
                        s.textContent.includes('"impressions"') &&
                        s.textContent.includes('ecommerce')
                    );
                    
                    if (!targetScript) return [];
                    
                    try {
                        const jsonStr = targetScript.textContent
                            .replace(/\\s+/g, ' ')  // Collapse whitespace
                            .match(/dataLayer\\.push\\(\\s*({.*?})\\s*\\)/)[1]  // Capture JSON
                            .replace(/'/g, '"')     // Standardize quotes
                            .replace(/([{,])(\\s*)([A-Za-z_]+)(\\s*):/g, '$1"$3":')  // Fix keys
                            .replace(/,\\s*}/g, '}');  // Remove trailing commas
                        
                        const parsed = JSON.parse(jsonStr);
                        return parsed.ecommerce && parsed.ecommerce.impressions ? 
                            parsed.ecommerce.impressions : [];
                    } catch (e) {
                        console.error("Error parsing product data:", e);
                        return [];
                    }
                    ''',
                    
                    # Method 2: Direct DOM extraction
                    '''
                    return Array.from(document.querySelectorAll('.product-grid__item')).map(item => {
                        try {
                            const nameEl = item.querySelector('.card__name');
                            const priceEl = item.querySelector('.card__price-regular');
                            const idEl = item.querySelector('[data-product-code]');
                            
                            return {
                                name: nameEl ? nameEl.textContent.trim() : '',
                                price: priceEl ? priceEl.textContent.trim().replace('€', '').replace(',', '.').trim() : '',
                                id: idEl ? idEl.getAttribute('data-product-code') : ''
                            };
                        } catch (e) {
                            return null;
                        }
                    }).filter(p => p && p.name && p.price);
                    ''',
                    
                    # Method 3: Alternative DOM selectors
                    '''
                    return Array.from(document.querySelectorAll('.js-product-container')).map(item => {
                        try {
                            const nameEl = item.querySelector('.name');
                            const priceEl = item.querySelector('.price');
                            const idEl = item.querySelector('[data-product-id]');
                            
                            return {
                                name: nameEl ? nameEl.textContent.trim() : '',
                                price: priceEl ? priceEl.textContent.trim().replace('€', '').replace(',', '.').trim() : '',
                                id: idEl ? idEl.getAttribute('data-product-id') : ''
                            };
                        } catch (e) {
                            return null;
                        }
                    }).filter(p => p && p.name && p.price);
                    ''',
                    
                    # Method 4: Generic product card extraction
                    '''
                    return Array.from(document.querySelectorAll('[class*="product"],[class*="card"]')).map(item => {
                        try {
                            // Look for elements that might contain product name
                            const nameEl = item.querySelector('[class*="name"],[class*="title"]');
                            
                            // Look for elements that might contain price
                            const priceEl = item.querySelector('[class*="price"]');
                            
                            // Look for product ID in data attributes
                            const id = item.getAttribute('data-product-code') || 
                                      item.getAttribute('data-product-id') || 
                                      item.getAttribute('data-id') || 
                                      '';
                            
                            return {
                                name: nameEl ? nameEl.textContent.trim() : '',
                                price: priceEl ? priceEl.textContent.trim().replace('€', '').replace(',', '.').trim() : '',
                                id: id
                            };
                        } catch (e) {
                            return null;
                        }
                    }).filter(p => p && p.name && p.price);
                    '''
                ]
                

                # Try each extraction method
                for method_index, method in enumerate(extraction_methods):
                    try:
                        logger.debug(f"Trying product extraction method {method_index + 1}/{len(extraction_methods)}")
                        extracted_products = driver.execute_script(method)
                        
                        if extracted_products and len(extracted_products) > 0:
                            logger.debug(f"Method {method_index + 1} found {len(extracted_products)} products")
                            products = extracted_products
                            break
                        else:
                            logger.debug(f"Method {method_index + 1} found no products")
                    except Exception as e:
                        logger.warning(f"Error with extraction method {method_index + 1}: {str(e)}")
                
                # Check if we found any products
                if len(products) == 0:
                    # Check if we're on a page with no products (end of pagination or empty category)
                    try:
                        no_products_text = driver.execute_script('''
                            const emptyEl = document.querySelector('.empty-search-results') || 
                                           document.querySelector('.no-results') ||
                                           document.querySelector('.empty-results');
                            return emptyEl ? emptyEl.textContent.trim() : '';
                        ''')
                        
                        if no_products_text:
                            logger.info(f"Category appears to be empty or end of pagination: '{no_products_text}'")
                            break
                    except Exception as e:
                        logger.warning(f"Error checking for empty results: {str(e)}")
                    
                    # If we're on page 1 and found no products, try one more time with a refresh
                    if page == 1:
                        logger.warning("No products found on first page, trying with refresh")
                        driver.refresh()
                        time.sleep(10)
                        continue
                    else:
                        logger.info(f"No more products found after page {page-1}, stopping pagination")
                        break

                logger.debug(f"Raw products found on page {page}: {len(products)}")

                valid_products_count = 0
                for product in products:
                    try:
                        name = product.get('name', '')
                        if not name:
                            continue

                        weight_value, unit_name = parse_product_details(name)

                        price = product.get('price', '')
                        if not price:
                            continue
                            
                        product_id = product.get('id', '')
                        image_url = f"https://rimibaltic-res.cloudinary.com/image/upload/b_white,c_limit,dpr_3.0,f_auto,q_auto:low,w_250/d_ecommerce:backend-fallback.png/MAT_{product_id}_PCE_EE"

                        result_products.append({
                            'name': name,
                            'price': price,
                            'image': image_url,
                            'weight_value': weight_value,
                            'unit_name': unit_name,
                            'category_id': category.category_id
                        })
                        valid_products_count += 1
                    except Exception as e:
                        logger.warning(f"Error parsing product: {str(e)}")
                        continue

                logger.debug(f"Valid products added from page {page}: {valid_products_count}")
                logger.debug(f"Current total products: {len(result_products)}")

                # If we found products, continue to next page
                if valid_products_count > 0:
                    page += 1
                    # Use a longer delay between pages to avoid rate limits
                    time.sleep(random_delay(2, 4))
                else:
                    # If we didn't find any valid products, stop pagination
                    logger.info(f"No valid products found on page {page}, stopping pagination")
                    break

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
    - Category Id: {item.get('category_id', 'N/A')}
    """)

    weight_value = item.get('weight_value')
    unit_name = item.get('unit_name')
    
    unit = None
    if weight_value and unit_name:
        unit = unit_service.get_by_name(db, name=unit_name)
        if not unit:
            unit_data = schemas.UnitCreate(
                name=unit_name,
                conversion_factor=get_conversion_factor(unit_name)
            )
            unit = unit_service.create(db, unit=unit_data)
            logger.debug(f"Created new unit: {unit_name} with factor {get_conversion_factor(unit_name)}")

    # Create or update ProductStoreData entry
    existing_psd = product_store_data_service.get_by_store_product_name_and_store(
        db,
        store_product_name=item['name'],
        store_id=store.store_id
    )

    psd_data = schemas.ProductStoreDataCreate(
        product_id=None,  # Initially null, will be set by matcher
        store_id=store.store_id,
        price=item['price'],
        store_product_name=item['name'],
        store_weight_value=weight_value,
        store_image_url=item['image'],
        store_unit_id=unit.unit_id if unit else None,
        store_category_id=item['category_id']
    )

    if existing_psd:
        old_price = existing_psd.price
        # If price has changed, save the old price to history
        if float(old_price) != float(item['price']):
            price_history = schemas.ProductPriceHistoryCreate(
                product_store_id=existing_psd.product_store_id,
                price=old_price
            )
            product_price_history_service.create(db, price_history)
        
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
        - Product Store ID: {new_psd.product_store_id}
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

def add_rimi_categories(db: Session, categories, parent_id=None):
    for category in categories:
        existing_category = category_service.get_by_name(db, name=category['name'])
        if not existing_category:
            category_data = schemas.CategoryCreate(
                name=category['name'],
                parent_id=parent_id,
                url=category['url'] if 'url' in category else None
            )

            db_category = category_service.create(db, category_data)
            logger.debug(f"Added category: {db_category.name} with ID: {db_category.category_id}")

            if category['subcategories']:
                add_rimi_categories(db, category['subcategories'], parent_id=db_category.category_id)
        else:
            logger.debug(f"Category '{category['name']}' already exists with ID: {existing_category.category_id}")


