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

    def create(self, db: Session, psd: schemas.ProductStoreDataCreate):
        db_psd = models.ProductStoreData(**psd.dict())
        db.add(db_psd)
        db.commit()
        db.refresh(db_psd)
        return db_psd

product_store_data_service = ProductStoreDataService()