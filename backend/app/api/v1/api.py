from fastapi import APIRouter
from app.api.v1.endpoints import products, stores, matching

api_router = APIRouter()
api_router.include_router(products.router, prefix="/products", tags=["products"])
api_router.include_router(stores.router, prefix="/stores", tags=["stores"])
api_router.include_router(matching.router, prefix="/matching", tags=["matching"])
