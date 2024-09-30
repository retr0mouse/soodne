from sqlalchemy.orm import Session
from app import models, schemas

class UnitService:
    def get(self, db: Session, unit_id: int):
        return db.query(models.Unit).filter(models.Unit.unit_id == unit_id).first()

    def get_by_name(self, db: Session, name: str):
        return db.query(models.Unit).filter(models.Unit.name == name).first()

    def create(self, db: Session, unit: schemas.UnitCreate):
        db_unit = models.Unit(**unit.dict())
        db.add(db_unit)
        db.commit()
        db.refresh(db_unit)
        return db_unit

unit_service = UnitService()
