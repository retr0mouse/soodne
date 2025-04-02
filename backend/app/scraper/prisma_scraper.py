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
def get_prisma_categories():
    try:
        base_url = "https://graphql-api.prismamarket.ee"
        body = {
            "operationName": "RemoteNavigation",
            "variables": {
                "id": "542860184"
            },
            "extensions": {
                "persistedQuery": {
                    "version": 1,
                    "sha256Hash": "707a9c68de67bcde9992a5d135e696c61d48abe1a9c765ca73ecf07bd80c513f"
                }
            }
        }
        logger.debug(f"Requesting Prisma categories")

        response = requests.post(base_url, json=body)
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
                'subcategories': get_prisma_subcategories(child),  # Recursively get subcategories
            }
            subcategories.append(subcategory)

    return subcategories


def get_prisma_items_by_category(category):
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

            response = requests.post(base_url, json=body)
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