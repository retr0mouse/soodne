import requests
from tenacity import stop_after_attempt, wait_exponential, retry

from app.core.logger import setup_logger
from app.utils.parse import parse_product_details

logger = setup_logger("scraper")
logger.setLevel("DEBUG")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def get_selver_categories():
    try:
        categories_api_url = "https://www.selver.ee/api/catalog/vue_storefront_catalog_et/category/_search?q=parent_id:3%20AND%20-_exists_:display_mode&_source_include=name,id,children_data,url_path&size=1000"
        logger.debug(f"Requesting Selver categories from: {categories_api_url}")

        response = requests.get(categories_api_url)
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
def get_selver_items_by_category(category, category_ids):
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

        response = requests.post(api_url, json=payload)

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