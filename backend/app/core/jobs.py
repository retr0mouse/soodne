import logging
from app.scraper.scraper import scrape_store_products

logger = logging.getLogger(__name__)

def scheduled_job():
    logger.info("Starting scheduled task: Store products parsing and AI matching.")
    try:
        scrape_store_products()
        logger.info("Store products parsing completed successfully.")
    except Exception as e:
        logger.error(f"Error during store products parsing: {e}")

    try:
        logger.info("AI matching completed successfully.")
    except Exception as e:
        logger.error(f"Error during AI matching: {e}")
