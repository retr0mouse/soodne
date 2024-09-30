from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ProductBase(BaseModel):
    name: str
    description: Optional[str] = None
    image_url: Optional[str] = None
    weight_value: Optional[float] = None
    unit_id: Optional[int] = None
    category_id: Optional[int] = None
    barcode: Optional[str] = None

class ProductCreate(ProductBase):
    pass

class Product(ProductBase):
    product_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
