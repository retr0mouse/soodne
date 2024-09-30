from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from app import schemas
from app.services import product_service
from app.api import deps

router = APIRouter()

@router.get("/", response_model=List[schemas.Product])
def read_products(
        skip: int = 0,
        limit: int = 100,
        name: Optional[str] = None,
        category_id: Optional[int] = None,
        min_weight: Optional[float] = None,
        max_weight: Optional[float] = None,
        unit_id: Optional[int] = None,
        db: Session = Depends(deps.get_db)
):
    products = product_service.get_multi(
        db, skip=skip, limit=limit,
        name=name, category_id=category_id,
        min_weight=min_weight, max_weight=max_weight,
        unit_id=unit_id
    )
    return products

@router.get("/{product_id}", response_model=schemas.Product)
def read_product(product_id: int, db: Session = Depends(deps.get_db)):
    product = product_service.get(db, product_id=product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product

@router.post("/", response_model=schemas.Product)
def create_product(product_in: schemas.ProductCreate, db: Session = Depends(deps.get_db)):
    existing_product = product_service.get_by_name_and_unit(
        db, name=product_in.name, unit_id=product_in.unit_id
    )
    if existing_product:
        raise HTTPException(status_code=400, detail="Product already exists")
    product = product_service.create(db, product=product_in)
    return product
