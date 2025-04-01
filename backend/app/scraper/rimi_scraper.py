import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from tenacity import stop_after_attempt, wait_exponential, retry

from app.core.logger import setup_logger
from app.utils.parse import parse_product_details, random_delay

logger = setup_logger("scraper")
logger.setLevel("DEBUG")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def get_rimi_categories(driver):
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    url = 'https://www.rimi.ee/epood/ee/parimad-pakkumised'
    logger.debug(f"Navigating to URL: {url}")
    driver.get(url)

    WebDriverWait(driver, 20).until(
        expected_conditions.presence_of_element_located((By.TAG_NAME, "body"))
    )

    # Try to handle cookie dialog if it appears
    try:
        cookie_dialog = WebDriverWait(driver, 5).until(
            expected_conditions.element_to_be_clickable((By.ID, "CybotCookiebotDialogBodyButtonDecline"))
        )
        cookie_dialog.click()
    except Exception as e:
        logger.debug(f"Cookie dialog handling: {str(e)}")

    driver.find_element(By.CLASS_NAME, 'category-menu').click()

    categories_selector = "category-list-item"

    def find_rimi_categories_recursively(depth=0):
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
                category_dict["url"] = 'https://rimi.ee' + found_button_elements[0].get_attribute('href')
            found_link_elements = category.find_elements(By.TAG_NAME, 'a')
            if found_link_elements:
                category_dict["url"] = found_link_elements[0].get_attribute('href')
            nested_categories.append(category_dict)

        return nested_categories

    return find_rimi_categories_recursively()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def get_rimi_items_by_category(category, driver):

    result_products = []
    page = 1
    max_pages = 5  # Limit to prevent infinite loops

    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    logger.debug("Chrome driver successfully initialized")

    while page <= max_pages:  # Add a maximum page limit to prevent infinite loops
        url = f"{category.url}?pageSize=100&currentPage={page}"
        logger.debug(f"Loading page: {url}")

        try:
            driver.get(url)

            # Wait for page to at least partially load
            WebDriverWait(driver, 30).until(
                expected_conditions.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Handle cookie banner more reliably
            try:
                WebDriverWait(driver, 10).until(
                    expected_conditions.presence_of_element_located((By.ID, "CybotCookiebotDialog"))
                )
                driver.execute_script("document.getElementById('CybotCookiebotDialog').remove()")
                logger.debug("Cookie banner removed")
            except Exception as e:
                logger.debug(f"Cookie banner not found or couldn't be closed: {e}")

            # Try multiple extraction methods in sequence
            products = []
            script = '''
                let rawString = Array.from(document.querySelectorAll("script")).find(s => 
                    s.textContent.includes('dataLayer.push') && 
                    s.textContent.includes('"impressions"') &&
                    s.textContent.includes('ecommerce')
                ).textContent;

                let jsonStart = rawString.indexOf('dataLayer.push(') + 'dataLayer.push('.length;
                let jsonEnd = rawString.lastIndexOf('});');
                let jsonStr = rawString.slice(jsonStart, jsonEnd + 1);
                
                let data = JSON.parse(jsonStr);
                
                let products = data.ecommerce.impressions;
                return products;
                '''

            extracted_products = driver.execute_script(script)
            if extracted_products and len(extracted_products) > 0:
                products = extracted_products
            else:
                logger.debug(f"Found no products using script")

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
                    page += 1
                    continue
                else:
                    logger.info(f"No more products found after page {page - 1}, stopping pagination")
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
                    image_url = f"https://rimibaltic-res.cloudinary.com/image/upload/b_white,c_fit,f_auto,h_960,q_auto,w_960/d_ecommerce:backend-fallback.png/MAT_{product_id}_PCE_EE"
                    product_url = f"https://www.rimi.ee/epood/ee/tooted/p/{product_id}"

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
    logger.debug(f"Final total products collected: {len(result_products)}")
    return result_products
