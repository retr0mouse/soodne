from sqlalchemy.orm import Session, joinedload
from typing import Optional, List
from app import models, schemas

class ProductService:
    def get(self, db: Session, product_id: int):
        return db.query(models.Product).filter(models.Product.product_id == product_id).first()

    def get_by_name_and_unit(self, db: Session, name: str, unit_id: Optional[int]):
        return db.query(models.Product).filter(
            models.Product.name == name,
            models.Product.unit_id == unit_id
        ).first()

    def get_multi(
            self, db: Session, skip: int = 0, limit: int = 100,
            name: Optional[str] = None,
            category_id: Optional[int] = None,
            min_weight: Optional[float] = None,
            max_weight: Optional[float] = None,
            unit_id: Optional[int] = None
    ) -> List[models.Product]:
        query = db.query(models.Product)
        if name:
            query = query.filter(models.Product.name.ilike(f"%{name}%"))
        if category_id:
            query = query.filter(models.Product.category_id == category_id)
        if min_weight:
            query = query.filter(models.Product.weight_value >= min_weight)
        if max_weight:
            query = query.filter(models.Product.weight_value <= max_weight)
        if unit_id:
            query = query.filter(models.Product.unit_id == unit_id)
        return query.offset(skip).limit(limit).all()

    def create(self, db: Session, product: schemas.ProductCreate):
        db_product = models.Product(**product.dict())
        db.add(db_product)
        db.commit()
        db.refresh(db_product)
        return db_product

    def get_products_with_prices(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100,
        name: Optional[str] = None,
        category_id: Optional[int] = None,
        store_id: Optional[int] = None
    ) -> List[models.Product]:
        """
        Get products with their store prices and basic info.
        """
        query = (
            db.query(models.Product)
            .options(
                joinedload(models.Product.store_data)
                .joinedload(models.ProductStoreData.store)
            )
        )

        if name:
            query = query.filter(models.Product.name.ilike(f"%{name}%"))
        if category_id:
            query = query.filter(models.Product.category_id == category_id)
        if store_id:
            query = query.join(models.Product.store_data).filter(models.ProductStoreData.store_id == store_id)

        query = query.order_by(models.Product.product_id)
        return query.offset(skip).limit(limit).all()

product_service = ProductService()
