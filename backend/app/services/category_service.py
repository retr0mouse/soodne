from sqlalchemy.orm import Session
from app import models, schemas

class CategoryService:
    def get(self, db: Session, category_id: int):
        return db.query(models.Category).filter(models.Category.category_id == category_id).first()

    def create(self, db: Session, category: schemas.CategoryCreate):
        db_category = models.Category(**category.dict())
        db.add(db_category)
        db.commit()
        db.refresh(db_category)
        return db_category

category_service = CategoryService()