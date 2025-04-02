import time

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from sqlalchemy.orm import Session

from app import schemas
from app.core.logger import setup_logger
from app.services import category_service
from app.utils.parse import parse_product_details, random_delay

logger = setup_logger("scraper")
logger.setLevel("DEBUG")

def get_barbora_categories(driver):
    try:
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        url = 'https://barbora.ee'
        logger.debug(f"Navigating to URL: {url}")
        driver.get(url)

        WebDriverWait(driver, 20).until(
            expected_conditions.presence_of_element_located((By.CLASS_NAME, "desktop-menu--parent-category-list"))
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


def get_barbora_items_by_category(db: Session, category_link, store_id, driver):
    result_products = []
    page = 1

    while True:
        url = f"https://barbora.ee{category_link}?page={page}"
        logger.debug(f"Loading page: {url}")

        try:
            driver.get(url)
            try:
                WebDriverWait(driver, 10).until(
                    expected_conditions.presence_of_element_located((By.ID, "CybotCookiebotDialog"))
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

            logger.debug(f"Valid products added from page {page}: {valid_products_count}")
            logger.debug(f"Current total products: {len(result_products)}")

            page += 1
            time.sleep(random_delay(1, 2))
        except Exception as e:
            logger.error(f"Error parsing products: {str(e)}")

    logger.debug(f"Final total products collected: {len(result_products)}")
    return result_products

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