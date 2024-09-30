from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from .. import models, schemas, crud
from ..database import SessionLocal

router = APIRouter(
    prefix="/stores",
    tags=["stores"],
    responses={404: {"description": "Not found"}},
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=schemas.Store)
def create_store(store: schemas.StoreCreate, db: Session = Depends(get_db)):
    db_store = db.query(models.Store).filter(models.Store.name == store.name).first()
    if db_store:
        raise HTTPException(status_code=400, detail="Store name already exists")
    return crud.create_store(db=db, store=store)

@router.get("/", response_model=List[schemas.Store])
def read_stores(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    stores = crud.get_stores(db, skip=skip, limit=limit)
    return stores

@router.get("/{store_id}", response_model=schemas.Store)
def read_store(store_id: int, db: Session = Depends(get_db)):
    db_store = crud.get_store(db, store_id=store_id)
    if db_store is None:
        raise HTTPException(status_code=404, detail="Store not found")
    return db_store