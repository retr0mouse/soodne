from sqlalchemy.orm import Session
from app import models, schemas

class ProductStoreDataService:
    def get(self, db: Session, product_store_id: int):
        return db.query(models.ProductStoreData).filter(
            models.ProductStoreData.product_store_id == product_store_id
        ).first()

    def get_by_product_and_store(self, db: Session, product_id: int, store_id: int):
        return db.query(models.ProductStoreData).filter(
            models.ProductStoreData.product_id == product_id,
            models.ProductStoreData.store_id == store_id
        ).first()

    def get_by_store_product_name_and_store(self, db: Session, store_product_name: str, store_id: int):
        return db.query(models.ProductStoreData).filter(
            models.ProductStoreData.store_product_name == store_product_name,
            models.ProductStoreData.store_id == store_id
        ).first()

    def get_by_store_product_url_and_store(self, db: Session, store_product_url: str, store_id: int):
        return db.query(models.ProductStoreData).filter(
            models.ProductStoreData.store_product_url == store_product_url,
            models.ProductStoreData.store_id == store_id
        ).first()

    def get_by_store_category(self, db: Session, store_category_id: int, store_id: int = None, skip: int = 0, limit: int = 100):
        query = db.query(models.ProductStoreData).filter(
            models.ProductStoreData.store_category_id == store_category_id
        )
        
        if store_id is not None:
            query = query.filter(models.ProductStoreData.store_id == store_id)
            
        return query.offset(skip).limit(limit).all()

    def create(self, db: Session, psd: schemas.ProductStoreDataCreate):
        db_psd = models.ProductStoreData(**psd.dict())
        db.add(db_psd)
        db.commit()
        db.refresh(db_psd)
        return db_psd

product_store_data_service = ProductStoreDataService()
