from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from app import schemas
from app.services import product_service
from app.api import deps
from app.core.logger import setup_logger
from app import models, schemas

router = APIRouter()

logger = setup_logger("api.products")

@router.get("/", response_model=List[schemas.Product])
def read_products(
    db: Session = Depends(deps.get_db),
    skip: int = 200,
    limit: int = 1000,
    name: Optional[str] = None,
    category_id: Optional[int] = None,
    min_weight: Optional[float] = None,
    max_weight: Optional[float] = None,
    unit_id: Optional[int] = None
):
    """
    Retrieve products with optional filtering.
    """
    try:
        products = product_service.get_multi(
            db, 
            skip=skip, 
            limit=limit,
            name=name,
            category_id=category_id,
            min_weight=min_weight,
            max_weight=max_weight,
            unit_id=unit_id
        )
        return products
    except Exception as e:
        logger.error(f"Error getting product list: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{product_id}", response_model=schemas.Product)
def read_product(product_id: int, db: Session = Depends(deps.get_db)):
    """
    Get a specific product by ID.
    """
    try:
        product = product_service.get(db, product_id=product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        return product
    except Exception as e:
        logger.error(f"Error getting product {product_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/", response_model=schemas.Product)
def create_product(product_in: schemas.ProductCreate, db: Session = Depends(deps.get_db)):
    """
    Create a new product.
    """
    try:
        existing_product = product_service.get_by_name_and_unit(
            db, name=product_in.name, unit_id=product_in.unit_id
        )
        if existing_product:
            raise HTTPException(status_code=400, detail="Product already exists")
        product = product_service.create(db, product=product_in)
        return product
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating product: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/with-prices/", response_model=List[schemas.ProductWithStoreData])
def read_products_with_prices(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    name: Optional[str] = None,
    category_id: Optional[int] = None,
    store_id: Optional[int] = None
):
    """
    Retrieve products with their store prices.
    """
    try:
        products = product_service.get_products_with_prices(
            db,
            skip=skip,
            limit=limit,
            name=name,
            category_id=category_id,
            store_id=store_id
        )
        return products
    except Exception as e:
        logger.error(f"Error getting products with prices: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")