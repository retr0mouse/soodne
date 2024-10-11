from sqlalchemy.orm import Session
from app import models, schemas

class StoreService:
    def get(self, db: Session, store_id: int):
        return db.query(models.Store).filter(models.Store.store_id == store_id).first()

    def get_all(self, db: Session, skip: int = 0, limit: int = 100):
        return db.query(models.Store).offset(skip).limit(limit).all()

    def get_by_name(self, db: Session, name: str):
        return db.query(models.Store).filter(models.Store.name == name).first()

    def create(self, db: Session, store: schemas.StoreCreate):
        db_store = models.Store(**store.dict())
        db.add(db_store)
        db.commit()
        db.refresh(db_store)
        return db_store

store_service = StoreService()
