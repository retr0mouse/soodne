import logging
from fastapi import FastAPI
from app.api.v1.api import api_router
from app.core.scheduler import scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API для Сравнения Цен на Продукты",
    description="API для сравнения цен на продукты в различных магазинах",
    version="1.0.0",
)

app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
def on_startup():
    logger.info("Запуск приложения и планировщика...")
    if not scheduler.running:
        scheduler.start()
        logger.info("Планировщик запущен.")
    else:
        logger.warning("Планировщик уже запущен.")

@app.on_event("shutdown")
def on_shutdown():
    logger.info("Завершение работы приложения и планировщика...")
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Планировщик остановлен.")
    else:
        logger.warning("Планировщик не был запущен.")