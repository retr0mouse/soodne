import logging
from app.scraper.scraper import scrape_store_products
from app.ai.matcher import run_ai_matching

logger = logging.getLogger(__name__)

def scheduled_job():
    logger.info("Запуск запланированной задачи: Парсинг продуктов магазинов и запуск AI-сопоставления.")
    try:
        scrape_store_products()
        logger.info("Парсинг продуктов магазинов успешно завершен.")
    except Exception as e:
        logger.error(f"Ошибка при парсинге продуктов магазинов: {e}")

    try:
        run_ai_matching()
        logger.info("AI-сопоставление успешно завершено.")
    except Exception as e:
        logger.error(f"Ошибка при AI-сопоставлении: {e}")
