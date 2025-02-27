from sqlalchemy.orm import Session
from app import models, schemas
from decimal import Decimal
from tenacity import retry, stop_after_attempt, wait_exponential

class ProductPriceService:
    def create(self, db: Session, product_id: int, store_id: int, price: float):
        if price <= 0:
            raise ValueError("Price must be greater than 0")
            
        new_price = models.ProductPrice(
            product_id=product_id,
            store_id=store_id,
            price=Decimal(str(price)).quantize(Decimal('0.01'))
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def create_or_update(self, db: Session, product_id: int, store_id: int, price: float):
        if price <= 0:
            raise ValueError("Price must be greater than 0")
        
        latest_price = self.get_latest_price(db, product_id, store_id)
        
        # Convert to Decimal for accurate comparison
        new_price = Decimal(str(price)).quantize(Decimal('0.01'))
        
        if not latest_price or Decimal(str(latest_price.price)) != new_price:
            return self.create(db, product_id, store_id, float(new_price))
        
        return latest_price

product_price_service = ProductPriceService() 