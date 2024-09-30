from fastapi import FastAPI
from app.api.v1.api import api_router
from app.core.scheduler import scheduler

app = FastAPI(
    title="API сравнения цен на продукты питания",
    description="API для сравнения цен на продукты питания в различных магазинах",
    version="1.0.0",
)

app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
def on_startup():
    scheduler.start()

@app.on_event("shutdown")
def on_shutdown():
    scheduler.shutdown()
