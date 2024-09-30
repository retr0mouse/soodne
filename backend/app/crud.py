from sqlalchemy.orm import Session
from . import models, schemas

def get_unit(db: Session, unit_id: int):
    return db.query(models.Unit).filter(models.Unit.unit_id == unit_id).first()

def get_units(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Unit).offset(skip).limit(limit).all()

def create_unit(db: Session, unit: schemas.UnitCreate):
    db_unit = models.Unit(**unit.dict())
    db.add(db_unit)
    db.commit()
    db.refresh(db_unit)
    return db_unit

def get_product(db: Session, product_id: int):
    return db.query(models.Product).filter(models.Product.product_id == product_id).first()

def get_products(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Product).offset(skip).limit(limit).all()

def get_product_by_barcode(db: Session, barcode: str):
    return db.query(models.Product).filter(models.Product.barcode == barcode).first()

def create_product(db: Session, product: schemas.ProductCreate):
    db_product = models.Product(**product.dict())
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return db_product

def get_product_store_data(db: Session, product_store_id: int):
    return db.query(models.ProductStoreData).filter(models.ProductStoreData.product_store_id == product_store_id).first()

def create_product_store_data(db: Session, psd: schemas.ProductStoreDataCreate):
    db_psd = models.ProductStoreData(**psd.dict())
    db.add(db_psd)
    db.commit()
    db.refresh(db_psd)
    return db_psd

def create_product_matching_log(db: Session, log: schemas.ProductMatchingLogCreate):
    db_log = models.ProductMatchingLog(**log.dict())
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log