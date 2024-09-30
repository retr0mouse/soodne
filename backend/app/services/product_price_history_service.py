from sqlalchemy.orm import Session
from app import models, schemas

class ProductPriceHistoryService:
    def create(self, db: Session, pph: schemas.ProductPriceHistoryCreate):
        db_pph = models.ProductPriceHistory(**pph.dict())
        db.add(db_pph)
        db.commit()
        db.refresh(db_pph)
        return db_pph

product_price_history_service = ProductPriceHistoryService()
