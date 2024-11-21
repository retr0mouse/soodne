from fastapi import FastAPI
from app.api.v1.api import api_router
from app.core.scheduler import scheduler
from app.core.logger import setup_logger

logger = setup_logger("main")

app = FastAPI(
    title="Product Price Comparison API",
    description="API for comparing product prices across different stores",
    version="1.0.0",
)

app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
async def on_startup():
    logger.info("Starting application and scheduler...")
    if not scheduler.running:
        scheduler.start()
        logger.info("Scheduler started")
    else:
        logger.warning("Scheduler is already running")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down application and scheduler...")
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler stopped")
    else:
        logger.warning("Scheduler was not running")