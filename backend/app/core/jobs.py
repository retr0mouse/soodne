import logging
from app.scraper.scraper import scrape_store_products
from app.ai.matcher import run_ai_matching
from fastapi.concurrency import Lock

logger = logging.getLogger(__name__)

scraping_lock = Lock()

async def scheduled_job():
    if not await scraping_lock.acquire(blocking=False):
        logger.warning("Another scraping job is already running")
        return
        
    try:
        scrape_store_products()
        run_ai_matching()
    finally:
        scraping_lock.release()
