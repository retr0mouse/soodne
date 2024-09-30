from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from . import models, schemas

def get_unit(db: Session, unit_id: int):
    return db.query(models.Unit).filter(models.Unit.unit_id == unit_id).first()

def get_units(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Unit).offset(skip).limit(limit).all()

def create_unit(db: Session, unit: schemas.UnitCreate):
    db_unit = models.Unit(**unit.dict())
    try:
        db.add(db_unit)
        db.commit()
        db.refresh(db_unit)
        return db_unit
    except IntegrityError:
        db.rollback()
        return None

def get_product(db: Session, product_id: int):
    return db.query(models.Product).filter(models.Product.product_id == product_id).first()

def get_products(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Product).offset(skip).limit(limit).all()

def get_product_by_barcode(db: Session, barcode: str):
    return db.query(models.Product).filter(models.Product.barcode == barcode).first()

def create_product(db: Session, product: schemas.ProductCreate):
    db_product = models.Product(**product.dict())
    try:
        db.add(db_product)
        db.commit()
        db.refresh(db_product)
        return db_product
    except IntegrityError:
        db.rollback()
        return None

def get_store(db: Session, store_id: int):
    return db.query(models.Store).filter(models.Store.store_id == store_id).first()

def get_stores(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Store).offset(skip).limit(limit).all()

def get_store_by_name(db: Session, name: str):
    return db.query(models.Store).filter(models.Store.name == name).first()

def create_store(db: Session, store: schemas.StoreCreate):
    db_store = models.Store(**store.dict())
    try:
        db.add(db_store)
        db.commit()
        db.refresh(db_store)
        return db_store
    except IntegrityError:
        db.rollback()
        return None

def get_product_store_data(db: Session, product_store_id: int):
    return db.query(models.ProductStoreData).filter(models.ProductStoreData.product_store_id == product_store_id).first()

def create_product_store_data(db: Session, psd: schemas.ProductStoreDataCreate):
    db_psd = models.ProductStoreData(**psd.dict())
    try:
        db.add(db_psd)
        db.commit()
        db.refresh(db_psd)
        return db_psd
    except IntegrityError:
        db.rollback()
        return None

def create_product_matching_log(db: Session, log: schemas.ProductMatchingLogCreate):
    db_log = models.ProductMatchingLog(**log.dict())
    try:
        db.add(db_log)
        db.commit()
        db.refresh(db_log)
        return db_log
    except IntegrityError:
        db.rollback()
        return None
