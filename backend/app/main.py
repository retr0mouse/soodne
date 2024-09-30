import logging
from fastapi import FastAPI
from app.api.v1.api import api_router
from app.core.scheduler import scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API for Food Price Comparison",
    description="API for comparing food prices across different stores",
    version="1.0.0",
)

app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
def on_startup():
    logger.info("Starting up the application and scheduler...")
    scheduler.start()

@app.on_event("shutdown")
def on_shutdown():
    logger.info("Shutting down the application and scheduler...")
    scheduler.shutdown()
