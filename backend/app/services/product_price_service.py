from sqlalchemy.orm import Session
from app import models, schemas

class ProductPriceService:
    def create(self, db: Session, product_id: int, store_id: int, price: float):
        new_price = models.ProductPrice(
            product_id=product_id,
            store_id=store_id,
            price=price
        )
        db.add(new_price)
        db.commit()
        db.refresh(new_price)
        return new_price

    def get_latest_price(self, db: Session, product_id: int, store_id: int):
        return db.query(models.ProductPrice).filter(
            models.ProductPrice.product_id == product_id,
            models.ProductPrice.store_id == store_id
        ).order_by(models.ProductPrice.created_at.desc()).first()

    def create_or_update(self, db: Session, product_id: int, store_id: int, price: float):
        latest_price = self.get_latest_price(db, product_id, store_id)
        
        # If no price exists or the price has changed, create a new record
        if not latest_price or float(latest_price.price) != float(price):
            return self.create(db, product_id, store_id, price)
        
        return latest_price

product_price_service = ProductPriceService() 