import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

from app import schemas
from app.core.logger import setup_logger
from app.database.database import SessionLocal
from app.scraper.barbora_scraper import get_barbora_categories, get_barbora_items_by_category
from app.scraper.prisma_scraper import get_prisma_categories, get_prisma_items_by_category
from app.scraper.rimi_scraper import get_rimi_categories, get_rimi_items_by_category
from app.scraper.selver_scraper import get_selver_items_by_category, get_selver_categories
from app.services import (
    store_service,
    unit_service,
    product_store_data_service,
    product_price_history_service
)
from app.services.category_service import category_service
from app.utils.parse import get_conversion_factor

db = SessionLocal()

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

def get_all_barbora_items(store):
    logger.info("Fetching Barbora categories...")
    categories = get_barbora_categories(driver)

    logger.info(f"Found {len(categories)} Barbora categories")

    for category_index, category in enumerate(categories, 1):
        logger.info(f"Processing Barbora category {category_index}/{len(categories)}: {category['title']}")

        category_items = get_barbora_items_by_category(db, category['link'], store.store_id, driver)
        logger.info(f"Found {len(category_items)} items in category {category['title']}")

        for item_index, item in enumerate(category_items, 1):
            logger.info(f"Processing item {item_index}/{len(category_items)}: {item['name']}")
            process_item(store, item)


def get_all_rimi_items(store):
    logger.info("Fetching Rimi categories...")
    categories = get_rimi_categories(driver)
    add_categories_to_db(categories, store.store_id)
    top_categories = category_service.get_top_categories(db, store.store_id)
    logger.info(f"Found {len(top_categories)} Rimi categories")

    for category_index, category in enumerate(top_categories, 1):
        logger.info(f"Processing Rimi category {category_index}/{len(top_categories)}: {category.name}")
        category_items = get_rimi_items_by_category(category, driver)
        logger.info(f"Found {len(category_items)} items in category {category.name}")

        for item_index, item in enumerate(category_items, 1):
            logger.info(f"Processing item {item_index}/{len(category_items)}: {item['name']}")
            process_item(store, item)

def get_all_selver_items(store):
    logger.info("Fetching Selver categories...")
    categories = get_selver_categories()
    add_categories_to_db(categories, store.store_id)
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

        category_items = get_selver_items_by_category(category, category_ids)
        logger.info(f"Found {len(category_items)} items in category {category.name}")

        for item_index, item in enumerate(category_items, 1):
            logger.info(f"Processing item {item_index}/{len(category_items)}: {item['name']}")
            process_item(store, item)

def get_all_prisma_items(store):
    logger.info("Fetching Prisma categories...")
    add_categories_to_db(get_prisma_categories(), store.store_id)
    top_categories = category_service.get_top_categories(db, store.store_id)
    logger.info(f"Found {len(top_categories)} Prisma categories")

    for category_index, category in enumerate(top_categories, 1):
        logger.info(f"Processing Prisma category {category_index}/{len(top_categories)}: {category.name}")
        category_items = get_prisma_items_by_category(category)
        logger.info(f"Found {len(category_items)} items in category {category.name}")

        for item_index, item in enumerate(category_items, 1):
            logger.info(f"Processing item {item_index}/{len(category_items)}: {item['name']}")
            process_item(store, item)

def process_store(store):
    match store.name:
        case 'Barbora':
            logger.info("Starting Barbora scraping...")
            get_all_barbora_items(store)
            logger.info("Finished Barbora scraping")
        case 'Rimi':
            logger.info("Starting Rimi scraping...")
            get_all_rimi_items(store)
            logger.info("Finished Rimi scraping")
        case 'Selver':
            logger.info("Starting Selver scraping...")
            get_all_selver_items(store)
            logger.info("Finished Selver scraping")
        case 'Prisma':
            logger.info("Starting Prisma scraping...")
            get_all_prisma_items(store)
            logger.info("Finished Prisma scraping")
        case _:
            logger.info(f"No scraper for store {store} is implemented")


def scrape_store_products():
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
                
                process_store(store)
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

def process_item(store, item):
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

def add_categories_to_db(categories, store_id, parent_id=None):
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
                add_categories_to_db(category['subcategories'], store_id, parent_id=db_category.category_id)
        else:
            logger.debug(f"Category '{category['name']}' already exists")
