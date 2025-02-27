from pydantic import BaseModel
from datetime import datetime

class ProductPriceBase(BaseModel):
    product_id: int
    store_id: int
    price: float

class ProductPriceCreate(ProductPriceBase):
    pass

class ProductPrice(ProductPriceBase):
    price_id: int
    created_at: datetime

    class Config:
        from_attributes = True 