# app/scraper/scrape_store_products.py

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

logger = setup_logger("scraper")

def random_delay(min_seconds, max_seconds):
    return random.uniform(min_seconds, max_seconds)

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

    # Barbora scraping
    logger.info("Starting Barbora scraping...")
    get_all_barbora_items(db, store, headers, user_agent)
    logger.info("Finished Barbora scraping")

    # Rimi scraping
    logger.info("Starting Rimi scraping...")
    get_all_rimi_items(db, store, headers, user_agent)
    logger.info("Finished Rimi scraping")

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
    url = 'https://barbora.ee'
    if not is_allowed(url, user_agent):
        logger.warning(f"Access to {url} is forbidden according to robots.txt")
        return []
        
    try:
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        script_tags = soup.find_all('script', type='text/javascript')
        categories_data = None
        
        for script in script_tags:
            if script.string and 'window.b_categories' in script.string:
                json_str = script.string.split('window.b_categories = ')[1].split(';')[0]
                categories_data = json.loads(json_str)
                break
                
        if not categories_data:
            logger.error("Categories data not found in page source")
            return []
            
        result_categories = []
        for category in categories_data['categories']:
            result_categories.append({
                'title': category['title'],
                'link': f"/{category['url']}" if category['url'] else None
            })
            
        return result_categories
        
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.error(f"Error getting Barbora categories: {str(e)}")
        return []

def get_barbora_items_by_category(category, headers, user_agent):
    result_products = []
    current_page = 1
    
    while True:
        time.sleep(random_delay(3, 7))
        url = f"https://barbora.ee{category['link']}?page={current_page}&pageSize=48"
        
        if not is_allowed(url, user_agent):
            logger.warning(f"Доступ к {url} запрещен согласно robots.txt")
            break
            
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'
        
        if not response.ok:
            logger.error(f"Ошибка при запросе страницы: {url}")
            break
            
        soup = BeautifulSoup(response.text, 'html.parser')
        items = soup.select('div.product-item')
        
        if not items:
            break
            
        for item_html in items:
            try:
                name = item_html.select_one('.product-name').get_text(strip=True)
                price = item_html.select_one('.product-price').get_text(strip=True)
                image = item_html.select_one('img')['src']
                
                result_products.append({
                    'name': name,
                    'price': price,
                    'image': image,
                    'category': category['title']
                })
                
            except (AttributeError, KeyError) as e:
                logger.error(f"Ошибка при обработке товара: {e}")
                continue
                
        current_page += 1
        
    return result_products

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

def get_rimi_categories(headers, user_agent):
    url = 'https://www.rimi.ee/epood/ee'
    if not is_allowed(url, user_agent):
        logger.warning(f"Access to {url} is forbidden according to robots.txt")
        return []
        
    try:
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        categories = soup.select('.main-navigation__list > .main-navigation__item')
        
        result_categories = []
        for category in categories:
            link_elem = category.select_one('a.main-navigation__link')
            if not link_elem:
                continue
                
            title = link_elem.text.strip()
            link = link_elem.get('href', '')
            
            if title and link:
                result_categories.append({
                    'title': title,
                    'link': link
                })
                
        return result_categories
        
    except (requests.RequestException, Exception) as e:
        logger.error(f"Error getting Rimi categories: {str(e)}")
        return []

def get_rimi_items_by_category(category, headers, user_agent):
    result_products = []
    current_page = 1
    
    while True:
        time.sleep(random_delay(3, 7))
        url = f"https://www.rimi.ee/epood/ee/products/{category['id']}?page={current_page}&pageSize=100"
        
        if not is_allowed(url, user_agent):
            logger.warning(f"Доступ к {url} запрещен согласно robots.txt") 
            break
            
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'
        
        if not response.ok:
            logger.error(f"Ошибка при запросе страницы: {url}")
            break
            
        soup = BeautifulSoup(response.text, 'html.parser')
        items = soup.select('li.product-grid__item')
        
        if not items:
            break
            
        for item_html in items:
            try:
                data_gtm = item_html.select_one('div[data-gtm-eec-product]')
                if not data_gtm:
                    continue
                    
                item_data = data_gtm.get('data-gtm-eec-product')
                item = json.loads(item_data)
                
                price_int = item_html.select_one('.price__integer')
                price_dec = item_html.select_one('.price__decimal')
                
                if not price_int or not price_dec:
                    continue
                    
                euros = price_int.get_text(strip=True)
                cents = price_dec.get_text(strip=True)
                
                try:
                    product_price = round(float(f"{euros}.{cents}"), 2)
                except ValueError:
                    continue
                    
                weight_value, unit_name = parse_weight(item.get('measure', ''))
                
                result_products.append({
                    'name': item['name'],
                    'price': product_price,
                    'image': item_html.select_one('img')['src'],
                    'category': item['category'],
                    'weight_value': weight_value,
                    'unit_name': unit_name
                })
                
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Ошибка при обработке товара: {e}")
                continue
                
        current_page += 1
        
    return result_products

def process_item(db: Session, store, item):
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
    else:
        unit = None
    product_data = schemas.ProductCreate(
        name=item['name'],
        image_url=item['image'],
        weight_value=weight_value,
        unit_id=unit.unit_id if unit else None
    )
    product = product_service.get_by_name_and_unit(db, name=item['name'], unit_id=unit.unit_id if unit else None)
    if not product:
        product = product_service.create(db, product=product_data)
    psd_data = schemas.ProductStoreDataCreate(
        product_id=product.product_id,
        store_id=store.store_id,
        price=item['price'],
        store_product_name=item['name'],
        store_image_url=item['image'],
        store_weight_value=weight_value,
        store_unit_id=unit.unit_id if unit else None
    )
    existing_psd = product_store_data_service.get_by_product_and_store(db, product_id=product.product_id, store_id=store.store_id)
    if existing_psd:
        existing_psd.price = item['price']
        existing_psd.last_updated = time.strftime('%Y-%m-%d %H:%M:%S')
        db.commit()
    else:
        product_store_data_service.create(db, psd=psd_data)

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
