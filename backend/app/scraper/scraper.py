import time
import requests

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential

from app import schemas
from app.core.logger import setup_logger
from app.database.database import SessionLocal
from app.scraper.rimi_scraper import get_rimi_categories, get_rimi_items_by_category
from app.services import (
    store_service,
    unit_service,
    product_store_data_service,
    product_price_history_service
)
from app.services.category_service import category_service
from app.utils.parse import get_conversion_factor, random_delay, parse_product_details

logger = setup_logger("scraper")
logger.setLevel("DEBUG")

user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'

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

service = Service()
driver = webdriver.Chrome(service=service, options=chrome_options)

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
        try:
            if 'driver' in locals():
                driver.quit()
                logger.debug("Chrome driver successfully closed")
        except Exception as e:
            logger.error(f"Error while closing driver: {str(e)}")
        db.close()

def get_all_rimi_items(db: Session, store, driver):
    logger.info("Fetching Rimi categories...")
    # add_categories_to_db(db, get_rimi_categories(driver), store.store_id)
    top_categories = category_service.get_top_categories(db, store.store_id)
    logger.info(f"Found {len(top_categories)} Rimi categories")

    for category_index, category in enumerate(top_categories, 1):
        # if category.name != "Peolaud - telli ette!":
        #     continue
        logger.info(f"Processing Rimi category {category_index}/{len(top_categories)}: {category.name}")
        category_items = get_rimi_items_by_category(category, driver)
        logger.info(f"Found {len(category_items)} items in category {category.name}")

        for item_index, item in enumerate(category_items, 1):
            logger.info(f"Processing item {item_index}/{len(category_items)}: {item['name']}")
            process_item(db, store, item)

def process_store(db: Session, store):


    match store.name:
        case 'Barbora':
            logger.info("Starting Barbora scraping...")
            # get_all_barbora_items(db, store, headers, user_agent)
            logger.info("Finished Barbora scraping")
        case 'Rimi':
            logger.info("Starting Rimi scraping...")
            get_all_rimi_items(db, store, driver)
            logger.info("Finished Rimi scraping")
        case 'Selver':
            logger.info("Starting Selver scraping...")
            # get_all_selver_items(db, store, headers, user_agent)
            logger.info("Finished Selver scraping")
        case 'Prisma':
            logger.info("Starting Prisma scraping...")
            # get_all_prisma_items(db, store, headers, user_agent)
            logger.info("Finished Prisma scraping")
        case _:
            logger.info(f"No scraper for store {store} is implemented")

def add_barbora_categories_to_db_by_url(db: Session, category_path_url, store_id):
    """
    Process category hierarchy from category_path_url.
    Returns the leaf category (the most specific one).
    """
    if not category_path_url:
        return None

    categories = category_path_url.split('/')
    parent_id = None
    last_category = None

    for index, cat_name in enumerate(categories, 1):
        if not cat_name:
            continue

        # Convert URL-friendly format to readable name
        display_name = cat_name.replace('-', ' ').title()

        # Check if category exists
        category = category_service.get_by_name(db, display_name)

        if not category:
            # Create new category
            category_data = schemas.CategoryCreate(
                name=display_name,
                parent_id=parent_id,
                store_id=store_id,
                url="https://barbora.ee/" + '/'.join(categories[0:index])
            )
            category = category_service.create(db, category_data)
            logger.debug(f"Created new category: {display_name} with parent_id: {parent_id}")

        # Set as parent for next iteration
        parent_id = category.category_id
        last_category = category

    return last_category

def process_item(db: Session, store, item):
    try:
        logger.debug(f"""
        Processing raw item data:
        - Name: {item['name']}
        - Price: {item['price']}
        - Image: {item['image']}
        - Category: {item.get('category', 'N/A')}
        - Weight Value: {item.get('weight_value', 'N/A')}
        - Unit Name: {item.get('unit_name', 'N/A')}
        - Product Url: {item.get('url', 'N/A')}
        - Category Url: {item.get('category_path_url', 'N/A')}
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
            store_category_id=item['category_id'],
            store_product_url=item['url']
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
            - Category: {item['category_id']}
            - Product URL: {item['url']}
            """)
        else:
            new_psd = product_store_data_service.create(db, psd=psd_data)
            logger.debug(f"""
            Created new store data:
            - Product: {item['name']}
            - Price: {item['price']}
            - Store: {store.name}
            - Product Store ID: {new_psd.product_store_id}
            - Category: {item['category_id']}
            - Product URL: {item['url']}
            """)

        logger.debug("=" * 50)
    except Exception as e:
        logger.error(f"Error processing items: {str(e)}", exc_info=True)
        return []

def get_all_barbora_items(db: Session, store, headers, user_agent):
    logger.info("Fetching Barbora categories...")
    categories = get_barbora_categories(headers, user_agent)

    logger.info(f"Found {len(categories)} Barbora categories")

    for category_index, category in enumerate(categories, 1):
        logger.info(f"Processing Barbora category {category_index}/{len(categories)}: {category['title']}")

        category_items = get_barbora_items_by_category(db, category['link'], store.store_id, headers, user_agent)
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
def get_barbora_items_by_category(db: Session, category_link, store_id, headers, user_agent):
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
            url = f"https://barbora.ee{category_link}?page={page}"
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
                        category = add_barbora_categories_to_db_by_url(db, product.get('category_path_url'), store_id)

                        url = product.get('Url', None)
                        store_url = "https://barbora.ee"
                        product_path = url.lstrip('/')
                        product_path = f"toode/{product_path}"
                        product_url = f"{store_url}/{product_path}"
                        logger.debug(f"Created product URL: {product_url}")

                        result_products.append({
                            'name': name,
                            'price': price,
                            'image': image_url,
                            'weight_value': weight_value,
                            'unit_name': unit_name,
                            'category_path_url': category.url,
                            'category_id': category.category_id,
                            'url': product_url
                        })
                        valid_products_count += 1
                        
                        logger.debug(f"""
                            Processing raw item data:
                            - Name: {name}
                            - Price: {price}
                            - Image: {image_url}
                            - Weight Value: {weight_value}
                            - Unit Name: {unit_name}
                            - Category Path: {category.url}
                            - Category Id: {category.category_id}
                            - Product Url: {product_url}
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

def get_all_selver_items(db: Session, store, headers, user_agent):
    logger.info("Fetching Selver categories...")
    categories = get_selver_categories(headers, user_agent)
    add_categories_to_db(db, categories, store.store_id)
    top_categories = category_service.get_top_categories(db, store.store_id)
    logger.info(f"Found {len(top_categories)} Selver categories")
    for category_index, category in enumerate(top_categories, 1):
        logger.info(f"Processing Selver category {category_index}/{len(top_categories)}: {category.name}")
        # Weird logic to get the list of category ids
        category_ids = []
        for c in categories:
            if c['name'] == category.name:
                category_ids = c['all_category_ids']
                break

        category_items = get_selver_items_by_category(category, category_ids, headers, user_agent)
        logger.info(f"Found {len(category_items)} items in category {category.name}")

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
                    'url': 'https://www.selver.ee/' + current_category.get('url_path'),
                    'subcategories': get_selver_subcategories(current_category),
                    'all_category_ids': get_selver_category_ids(current_category)
                }
                categories.append(category)
                logger.debug(f"Added category: {category['name']} with URL path: {category['url']}")
        
        logger.info(f"Found {len(categories)} categories from Selver")
        return categories
    except Exception as e:
        logger.error(f"Error getting Selver categories: {str(e)}", exc_info=True)
        return []


# returns an array containing all subcategories
def get_selver_subcategories(category):
    subcategories = []

    if category.get('children_data'):
        for child in category['children_data']:
            subcategory = {
                'name': child['name'],
                'url': 'https://www.selver.ee/' + child['url_path'],
                'subcategories': get_selver_subcategories(child),  # Recursively get subcategories
            }
            subcategories.append(subcategory)

    return subcategories

def get_selver_category_ids(category):
    category_ids = [category['id']]

    if category['children_data']:
        for child in category['children_data']:
            category_ids.extend(get_selver_category_ids(child))
    return category_ids


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def get_selver_items_by_category(category, category_ids, headers, user_agent):
    result_products = []
    
    try:
        api_url = "https://www.selver.ee/api/catalog/vue_storefront_catalog_et/product/_search"
        payload = {
            "_source": ["name", "prices", "media_gallery", "product_volume", "url_path"],
            "query": {
                "bool": {
                    "filter": {
                        "bool": {
                            "must": [
                                {"terms": {"category_ids": category_ids}}
                            ]
                        }
                    }
                }
            },
            "size": 10000
        }

        logger.debug(f"Requesting Selver products from API for category: {category.name}")
        
        response = requests.post(api_url, json=payload, headers=headers)
        
        data = response.json()
        
        if 'hits' in data and 'hits' in data['hits']:
            products_data = data['hits']['hits']
            logger.debug(f"Found {len(products_data)} products from API for category: {category.name}")
            
            for product in products_data:
                try:
                    current_product = product.get('_source')
                    name = current_product.get('name')
                    price = current_product['prices'][0]['final_price']
                    image_path = current_product['media_gallery'][0]['image']
                    image_url = f"https://www.selver.ee/img/800/800/resize{image_path}" if image_path else None
                    weight_value, unit_name = parse_product_details(current_product['product_volume'])
                    product_url = f"https://www.selver.ee/{current_product['url_path']}"

                    # Create product data
                    result_products.append({
                        'name': name,
                        'price': price,
                        'image': image_url,
                        'weight_value': weight_value,
                        'unit_name': unit_name,
                        'category_path_url': category.url,
                        'category_id': category.category_id,
                        'url': product_url
                    })
                    
                    logger.debug(f"""
                        Processing raw item data:
                        - Name: {name}
                        - Price: {price}
                        - Image: {image_url}
                        - Weight Value: {weight_value}
                        - Unit Name: {unit_name}
                        - Category Url: {category.url}
                        - Category Id: {category.category_id},
                        - Url: {product_url}
                    """)
                    
                except Exception as e:
                    logger.warning(f"Error parsing product: {str(e)}")
                    continue
        
    except Exception as e:
        logger.error(f"Error fetching products from Selver API: {str(e)}", exc_info=True)

    logger.debug(f"Final total products collected for category {category.name}: {len(result_products)}")
    return result_products

def get_all_prisma_items(db: Session, store, headers, user_agent):
    logger.info("Fetching Prisma categories...")
    add_categories_to_db(db, get_prisma_categories(headers, user_agent), store.store_id)
    top_categories = category_service.get_top_categories(db, store.store_id)
    logger.info(f"Found {len(top_categories)} Prisma categories")

    for category_index, category in enumerate(top_categories, 1):
        logger.info(f"Processing Prisma category {category_index}/{len(top_categories)}: {category.name}")
        category_items = get_prisma_items_by_category(category, headers, user_agent)
        logger.info(f"Found {len(category_items)} items in category {category.name}")

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
        found_categories = data['data']['store']['navigation']
        if found_categories:
            # skip the first 2 categories ("Aktuaalne", "Food Market")
            found_categories.pop(0)
            found_categories.pop(0)
            for current_category in found_categories:
                category = {
                    'name': current_category.get('name'),
                    'slug': current_category.get('slug'),
                    'id': current_category.get('id'),
                    'url': 'https://www.prismamarket.ee/tooted/' + current_category.get('slug'),
                    'subcategories': get_prisma_subcategories(current_category)
                }
                categories.append(category)
                logger.debug(f"Added category: {category['name']} ")

        logger.info(f"Found {len(categories)} categories from Prisma")
        return categories
        
    except Exception as e:
        logger.error(f"Error getting Prisma categories: {str(e)}", exc_info=True)
        return []

def get_prisma_subcategories(category):
    subcategories = []

    if category.get('children'):
        for child in category['children']:
            subcategory = {
                'name': child.get('name'),
                'slug': child.get('slug'),
                'id': child.get('id'),
                'url': 'https://www.prismamarket.ee/tooted/' + child.get('slug'),
                'subcategories': get_selver_subcategories(child),  # Recursively get subcategories
            }
            subcategories.append(subcategory)

    return subcategories

def get_prisma_items_by_category(category, headers, user_agent):
    result_products = []
    
    try:
        offset = 0
        while True:
            base_url = "https://graphql-api.prismamarket.ee"
            slug = category.url.replace("https://www.prismamarket.ee/tooted/", "")
            body = {
                "operationName": "RemoteFilteredProducts",
                "variables": {
                    "includeAgeLimitedByAlcohol": True,
                    "limit": 120,
                    "from": offset,
                    "queryString": "",
                    "searchProvider": "loop54",
                    "slug": slug,
                    "storeId": "542860184"
                },
                "extensions": {
                    "persistedQuery": {
                        "version": 1,
                        "sha256Hash": "86214929199d2277cbe0a8c138b2be4db7d5b32df8399bd3d266377ffc9c29b4"
                    }
                }
            }

            logger.debug(f"Requesting Prisma products from API for category: {category.name} (SLUG: {slug})")

            response = requests.post(base_url, json=body, headers=headers)
            response.raise_for_status()

            data = response.json()
            products = data['data']['store']['products']['items']
            if len(products) > 0:
                logger.debug(f"Found {len(products)} products from API for category: {category.name}")

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
                            'unit_name': unit_name,
                            'url': f"https://www.prismamarket.ee/toode/{product['slug']}/{product['id']}",
                            'category_id': category.category_id
                        })

                        logger.debug(f"""
                                    Processing raw item data:
                                    - Name: {name}
                                    - Price: {price}
                                    - Image: {image_url}
                                    - Weight Value: {weight_value}
                                    - Unit Name: {unit_name}
                                    - Category Id': {category.category_id}
                                """)
                    except Exception as e:
                        logger.warning(f"Error parsing product: {str(e)}")
                        continue
                offset = offset + 120
            else:
                break
    except Exception as e:
        logger.error(f"Error in get_prisma_items_by_category: {str(e)}", exc_info=True)
    
    logger.debug(f"Total products collected for category {category.name}: {len(result_products)}")
    return result_products

def add_categories_to_db(db: Session, categories, store_id, parent_id=None):
    for category in categories:
        existing_category = category_service.get_by_store_id_and_name(db, store_id=store_id, name=category['name'])
        if not existing_category:
            category_data = schemas.CategoryCreate(
                name=category['name'],
                parent_id=parent_id,
                url=category['url'] if 'url' in category else None,
                store_id=store_id
            )

            db_category = category_service.create(db, category_data)
            logger.info(f"Added category: {db_category.name} with Category ID: {db_category.category_id} and Store ID: {store_id}")

            if category['subcategories']:
                add_categories_to_db(db, category['subcategories'], store_id, parent_id=db_category.category_id)
        else:
            logger.debug(f"Category '{category['name']}' already exists")


