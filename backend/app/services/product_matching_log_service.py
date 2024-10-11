from sqlalchemy.orm import Session
from app import models, schemas

class ProductMatchingLogService:
    def create(self, db: Session, log: schemas.ProductMatchingLogCreate):
        db_log = models.ProductMatchingLog(**log.dict())
        db.add(db_log)
        db.commit()
        db.refresh(db_log)
        return db_log

product_matching_log_service = ProductMatchingLogService()
