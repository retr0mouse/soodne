from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app import schemas
from app.services import store_service
from app.api import deps

router = APIRouter()

@router.get("/", response_model=List[schemas.Store])
def read_stores(skip: int = 0, limit: int = 100, db: Session = Depends(deps.get_db)):
    stores = store_service.get_all(db, skip=skip, limit=limit)
    return stores

@router.get("/{store_id}", response_model=schemas.Store)
def read_store(store_id: int, db: Session = Depends(deps.get_db)):
    store = store_service.get(db, store_id=store_id)
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    return store

@router.post("/", response_model=schemas.Store)
def create_store(store_in: schemas.StoreCreate, db: Session = Depends(deps.get_db)):
    existing_store = store_service.get_by_name(db, name=store_in.name)
    if existing_store:
        raise HTTPException(status_code=400, detail="Store already exists")
    store = store_service.create(db, store=store_in)
    return store
