from fastapi import FastAPI
from .api.products import router as products_router
from .api.stores import router as stores_router
from .api.matching import router as matching_router

app = FastAPI(
    title="Food Price Comparison API",
    description="API Food Price Comparison",
    version="1.0.0",
)

app.include_router(products_router, prefix="/api/products", tags=["products"])
app.include_router(stores_router, prefix="/api/stores", tags=["stores"])
app.include_router(matching_router, prefix="/api/matching", tags=["matching"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Food Price Comparison API"}
