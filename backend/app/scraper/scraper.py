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
from app.core.logger import logger  # Импортируем настроенный логгер

def random_delay(min_seconds, max_seconds):
    return random.uniform(min_seconds, max_seconds)

def is_allowed(url, user_agent='Soodne/1.0'):
    domain = '/'.join(url.split('/')[:3])
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"{domain}/robots.txt")
    rp.read()
    return rp.can_fetch(user_agent, url)

def scrape_store_products():
    db: Session = SessionLocal()
    user_agent = 'Soodne/1.0 (+Daniil Šarin <nuacho@tlu.ee>)'
    headers = {'User-Agent': user_agent}

    barbora_store = store_service.get_by_name(db, name="Barbora")
    if not barbora_store:
        store_data = schemas.StoreCreate(
            name="Barbora",
            website_url="https://barbora.ee"
        )
        barbora_store = store_service.create(db, store=store_data)
    get_all_barbora_items(db, barbora_store, headers, user_agent)

    rimi_store = store_service.get_by_name(db, name="Rimi")
    if not rimi_store:
        store_data = schemas.StoreCreate(
            name="Rimi",
            website_url="https://www.rimi.ee"
        )
        rimi_store = store_service.create(db, store=store_data)
    get_all_rimi_items(db, rimi_store, headers, user_agent)

    db.close()

def get_all_barbora_items(db: Session, store, headers, user_agent):
    categories = get_barbora_categories(headers, user_agent)
    for category_index, category in enumerate(categories):
        logger.info(f"{category_index} - Парсинг Barbora категории: {category['title']}")
        if not category['link']:
            continue
        category_items = get_barbora_items_by_category(category, headers, user_agent)
        for item in category_items:
            process_item(db, store, item)

def get_barbora_categories(headers, user_agent):
    url = 'https://barbora.ee'
    if not is_allowed(url, user_agent):
        logger.warning(f"Доступ к {url} запрещен согласно robots.txt")
        return []
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'  # Указываем кодировку явно
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    categories = soup.select('li.b-categories-root-category > a')
    result_categories = []
    for category in categories:
        title = category.text.strip()
        link = category['href']
        result_categories.append({'title': title, 'link': link})
    return result_categories

def get_barbora_items_by_category(category, headers, user_agent):
    result_products = []
    current_page = 0
    while True:
        time.sleep(random_delay(3, 7))
        url = f"https://barbora.ee{category['link']}?page={current_page}"
        if not is_allowed(url, user_agent):
            logger.warning(f"Доступ к {url} запрещен согласно robots.txt")
            break
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'  # Указываем кодировку явно
        if not response.ok:
            logger.error(f"Ошибка при запросе страницы: {url}")
            break
        try:
            content = response.content.decode('utf-8', 'ignore')
        except UnicodeDecodeError as e:
            logger.error(f"Ошибка декодирования контента: {e}")
            break
        soup = BeautifulSoup(content, 'html.parser')
        items = soup.select('div.b-product--wrap[data-b-for-cart]')
        if not items:
            break
        for item_html in items:
            item_data = item_html.get('data-b-for-cart')
            if not item_data:
                continue
            try:
                item = json.loads(item_data)
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка декодирования JSON: {e}")
                continue
            product_name = item['title']
            product_price = round(float(item['price']), 2)
            product_image_url = item['image']
            product_weight = item.get('measure')
            weight_value, unit_name = parse_weight(product_weight)
            result_products.append({
                'name': product_name,
                'price': product_price,
                'image': product_image_url,
                'category': category['title'],
                'weight_value': weight_value,
                'unit_name': unit_name
            })
        current_page += 1
    return result_products

def get_all_rimi_items(db: Session, store, headers, user_agent):
    categories = get_rimi_categories(headers, user_agent)
    for category_index, category in enumerate(categories):
        logger.info(f"{category_index} - Парсинг Rimi категории: {category['title']}")
        if not category['link']:
            continue
        category_items = get_rimi_items_by_category(category, headers, user_agent)
        for item in category_items:
            process_item(db, store, item)

def get_rimi_categories(headers, user_agent):
    url = 'https://www.rimi.ee/epood/ee'
    if not is_allowed(url, user_agent):
        logger.warning(f"Доступ к {url} запрещен согласно robots.txt")
        return []
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'  # Указываем кодировку явно
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    categories = soup.select('div.category-menu a')
    result_categories = []
    for category in categories:
        title = category.get_text(strip=True)
        link = category['href']
        result_categories.append({'title': title, 'link': link})
    return result_categories

def get_rimi_items_by_category(category, headers, user_agent):
    result_products = []
    current_page = 1
    while True:
        time.sleep(random_delay(3, 7))
        url = f"https://www.rimi.ee{category['link']}?page={current_page}&pageSize=99"
        if not is_allowed(url, user_agent):
            logger.warning(f"Доступ к {url} запрещен согласно robots.txt")
            break
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'  # Указываем кодировку явно
        if not response.ok:
            logger.error(f"Ошибка при запросе страницы: {url}")
            break
        try:
            content = response.content.decode('utf-8', 'ignore')
        except UnicodeDecodeError as e:
            logger.error(f"Ошибка декодирования контента: {e}")
            break
        soup = BeautifulSoup(content, 'html.parser')
        items = soup.select('li.product-grid__item')
        if not items:
            break
        for item_html in items:
            data_gtm = item_html.select_one('div[data-gtm-eec-product]')
            if not data_gtm:
                continue
            item_data = data_gtm.get('data-gtm-eec-product')
            try:
                item = json.loads(item_data)
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка декодирования JSON: {e}")
                continue
            product_name = item['name']
            euros = item_html.select_one('.price__integer').get_text(strip=True)
            cents = item_html.select_one('.price__decimal').get_text(strip=True)
            try:
                product_price = round(float(f"{euros}.{cents}"), 2)
            except ValueError as e:
                logger.error(f"Ошибка конвертации цены: {e}")
                continue
            product_image_url = item_html.select_one('img')['src']
            product_weight = item.get('measure')
            weight_value, unit_name = parse_weight(product_weight)
            result_products.append({
                'name': product_name,
                'price': product_price,
                'image': product_image_url,
                'category': item['category'],
                'weight_value': weight_value,
                'unit_name': unit_name
            })
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
