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
            "Rimi": "https://www.rimi.ee/epood/ee",
            "Barbora": "https://barbora.ee"
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
    
    max_retries = 3
    retry_count = 0
    categories = []
    
    # Try to get categories with multiple attempts if needed
    while retry_count < max_retries and not categories:
        try:
            categories = get_rimi_categories(headers, user_agent)
            if not categories:
                logger.warning(f"No categories returned (attempt {retry_count + 1}/{max_retries})")
                retry_count += 1
                time.sleep(10)  # Wait before retrying
            else:
                logger.info(f"Successfully found {len(categories)} Rimi categories")
        except Exception as e:
            logger.error(f"Error retrieving categories (attempt {retry_count + 1}/{max_retries}): {str(e)}")
            retry_count += 1
            time.sleep(10)  # Wait before retrying
    
    if not categories:
        logger.error("Failed to retrieve any categories after multiple attempts")
        return
    
    total_products_processed = 0
    successful_categories = 0
    failed_categories = 0
    
    # Shuffle categories to avoid getting stuck on problematic ones
    import random
    random.shuffle(categories)
    logger.info("Categories shuffled to randomize processing order")
    
    for category_index, category in enumerate(categories, 1):
        try:
            logger.info(f"Processing Rimi category {category_index}/{len(categories)}: {category['title']}")
            if not category['link']:
                logger.warning(f"Skipping category {category['title']} - no link available")
                failed_categories += 1
                continue
            
            # Process each category with retry logic
            category_retry_count = 0
            category_items = []
            
            while category_retry_count < 3 and not category_items:  # Try up to 3 times per category
                try:
                    category_items = get_rimi_items_by_category(category, headers, user_agent)
                    if not category_items and category_retry_count < 2:
                        logger.warning(f"No items found in category {category['title']} (attempt {category_retry_count + 1}/3)")
                        category_retry_count += 1
                        time.sleep(random_delay(5, 10))
                except Exception as e:
                    logger.error(f"Error processing category {category['title']} (attempt {category_retry_count + 1}/3): {str(e)}")
                    category_retry_count += 1
                    if category_retry_count < 3:
                        time.sleep(random_delay(5, 10))
            
            if not category_items:
                logger.error(f"Failed to retrieve items for category {category['title']} after retries")
                failed_categories += 1
                continue
                
            logger.info(f"Found {len(category_items)} items in category {category['title']}")
            successful_categories += 1
            
            # Process items in batches to avoid long transactions
            batch_size = 50
            for i in range(0, len(category_items), batch_size):
                batch = category_items[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(category_items) + batch_size - 1)//batch_size} of items in category {category['title']}")
                
                batch_success_count = 0
                for item_index, item in enumerate(batch, 1):
                    try:
                        logger.info(f"Processing item {item_index}/{len(batch)}: {item['name']}")
                        process_item(db, store, item)
                        total_products_processed += 1
                        batch_success_count += 1
                    except Exception as e:
                        logger.error(f"Error processing item {item['name']}: {str(e)}")
                        continue
                
                logger.info(f"Successfully processed {batch_success_count}/{len(batch)} items in this batch")
                
                # Commit the database changes after each batch
                try:
                    db.commit()
                    logger.info(f"Batch {i//batch_size + 1} committed successfully")
                except Exception as e:
                    logger.error(f"Error committing batch {i//batch_size + 1}: {str(e)}")
                    db.rollback()
                
                # Small delay between batches
                time.sleep(random_delay(1, 3))
                
        except Exception as e:
            logger.error(f"Unhandled error processing category {category['title']}: {str(e)}")
            failed_categories += 1
            continue
        
        # After each category, log progress
        logger.info(f"Progress: {category_index}/{len(categories)} categories processed. Success: {successful_categories}, Failed: {failed_categories}")
    
    logger.info(f"Rimi scraping completed. Total products processed: {total_products_processed}")
    logger.info(f"Categories summary - Total: {len(categories)}, Successful: {successful_categories}, Failed: {failed_categories}")

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
        
        # Try multiple URLs to get categories
        urls_to_try = [
            'https://www.rimi.ee/epood/ee',
            'https://www.rimi.ee/epood/ee/categories',
            'https://www.rimi.ee/epood/ee/products'
        ]
        
        for url_index, url in enumerate(urls_to_try):
            logger.debug(f"Trying URL {url_index + 1}/{len(urls_to_try)}: {url}")
            driver.get(url)
            
            # First wait for page to basically load
            time.sleep(10)
            
            # More reliable wait - wait for page to be fully loaded
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Try to handle cookie dialog if it appears
            try:
                cookie_dialog = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.ID, "CybotCookiebotDialog"))
                )
                if cookie_dialog:
                    driver.execute_script("document.getElementById('CybotCookiebotDialog').remove()")
                    logger.debug("Removed cookie dialog")
            except Exception as e:
                logger.debug(f"Cookie dialog handling: {str(e)}")
            
            # Try multiple selectors to find categories
            selectors = [
                ".category-menu button.trigger",
                ".category-navigation a[href*='/c/']",
                "a[href*='/c/']",
                ".category-menu__item a",
                ".category-list a",
                "nav a[href*='/c/']",
                ".main-navigation a[href*='/c/']"
            ]
            
            category_elements = []
            for selector in selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements and len(elements) > 0:
                        logger.debug(f"Found {len(elements)} category elements with selector: {selector}")
                        category_elements = elements
                        break
                except Exception as e:
                    logger.warning(f"Error finding elements with selector {selector}: {str(e)}")
            
            if category_elements:
                logger.debug(f"Found {len(category_elements)} category elements")
                break
            
            logger.warning(f"No category elements found with URL {url}, trying next URL if available")
        
        # Process found category elements
        for element in category_elements:
            try:
                href = element.get_attribute('href')
                # Different ways to get the name depending on element type
                try:
                    name = element.find_element(By.CSS_SELECTOR, "span.name").text.strip()
                except:
                    name = element.text.strip()
                
                if href and name and '/c/' in href:
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
            # Try multiple JavaScript approaches
            js_scripts = [
                """
                return Array.from(document.querySelectorAll('.category-menu button.trigger')).map(el => {
                    const href = el.getAttribute('href');
                    const categoryId = href ? href.split('/c/').pop().trim() : null;
                    return {
                        title: el.querySelector('span.name').textContent.trim(),
                        link: href,
                        id: categoryId
                    };
                }).filter(cat => cat.title && cat.link && cat.id);
                """,
                """
                return Array.from(document.querySelectorAll('a[href*="/c/"]')).map(el => {
                    const href = el.getAttribute('href');
                    const categoryId = href ? href.split('/c/').pop().trim() : null;
                    return {
                        title: el.textContent.trim(),
                        link: href,
                        id: categoryId
                    };
                }).filter(cat => cat.title && cat.link && cat.id);
                """,
                """
                // Try to find categories in any navigation element
                return Array.from(document.querySelectorAll('nav a, .navigation a, .menu a')).map(el => {
                    const href = el.getAttribute('href');
                    if (!href || !href.includes('/c/')) return null;
                    
                    const categoryId = href.split('/c/').pop().trim();
                    return {
                        title: el.textContent.trim(),
                        link: href,
                        id: categoryId
                    };
                }).filter(cat => cat && cat.title && cat.link && cat.id);
                """,
                """
                // Last resort - try to find any links that might be categories
                return Array.from(document.querySelectorAll('a')).map(el => {
                    const href = el.getAttribute('href');
                    if (!href || !href.includes('/c/')) return null;
                    
                    const categoryId = href.split('/c/').pop().trim();
                    return {
                        title: el.textContent.trim(),
                        link: href,
                        id: categoryId
                    };
                }).filter(cat => cat && cat.title && cat.link && cat.id);
                """
            ]
            
            for script in js_scripts:
                try:
                    categories_js = driver.execute_script(script)
                    
                    if categories_js and len(categories_js) > 0:
                        categories = categories_js
                        logger.debug(f"Found {len(categories)} categories using JavaScript")
                        break
                except Exception as e:
                    logger.warning(f"JavaScript extraction attempt failed: {str(e)}")
        
        # If still no categories, try hardcoded fallback categories
        if not categories:
            logger.warning("No categories found through normal methods, using fallback categories")
            fallback_categories = [
                {"title": "Puu- ja köögiviljad", "link": "/epood/ee/c/puu-ja-koogiviljad", "id": "puu-ja-koogiviljad"},
                {"title": "Liha ja kala", "link": "/epood/ee/c/liha-ja-kala", "id": "liha-ja-kala"},
                {"title": "Piimatooted ja munad", "link": "/epood/ee/c/piimatooted-ja-munad", "id": "piimatooted-ja-munad"},
                {"title": "Leivad ja saiad", "link": "/epood/ee/c/leivad-ja-saiad", "id": "leivad-ja-saiad"},
                {"title": "Valmistoit", "link": "/epood/ee/c/valmistoit", "id": "valmistoit"},
                {"title": "Külmutatud toit", "link": "/epood/ee/c/kulmutatud-toit", "id": "kulmutatud-toit"},
                {"title": "Kuivained ja hommikusöök", "link": "/epood/ee/c/kuivained-ja-hommikusook", "id": "kuivained-ja-hommikusook"},
                {"title": "Joogid", "link": "/epood/ee/c/joogid", "id": "joogid"},
                {"title": "Maiustused ja snäkid", "link": "/epood/ee/c/maiustused-ja-snakid", "id": "maiustused-ja-snakid"}
            ]
            categories = fallback_categories
            
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
            url = f"https://www.rimi.ee{category['link']}?pageSize=100&currentPage={page}"
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
                            'unit_name': unit_name
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
        store_unit_id=unit.unit_id if unit else None
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
