from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from decimal import Decimal

class StoreDataBase(BaseModel):
    product_store_id: int
    store_id: int
    price: Decimal
    price_per_unit: Optional[Decimal]
    store_product_name: str
    store_image_url: Optional[str]
    store_weight_value: Optional[Decimal]
    store_unit_id: Optional[int]
    ean: Optional[str]
    last_updated: datetime
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True

class ProductBase(BaseModel):
    name: str
    description: Optional[str]
    image_url: Optional[str]
    weight_value: Optional[Decimal]
    unit_id: Optional[int]
    category_id: Optional[int]
    barcode: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True

class ProductCreate(ProductBase):
    pass

class Product(ProductBase):
    product_id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True

class ProductStoreDataBase(BaseModel):
    price: Decimal
    price_per_unit: Optional[Decimal] = None
    store_product_name: Optional[str] = None
    store_image_url: Optional[str] = None
    store_weight_value: Optional[Decimal] = None
    store_unit_id: Optional[int] = None
    store_id: int
    ean: Optional[str] = None

class ProductWithStoreData(ProductBase):
    product_id: int
    store_data: List[StoreDataBase]

    class Config:
        orm_mode = True

class StorePrice(BaseModel):
    store_id: int
    store_name: str
    price: Decimal
    last_updated: datetime

    class Config:
        orm_mode = True

class ProductWithPrices(BaseModel):
    product_id: int
    name: str
    image_url: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    store_data: List[StorePrice]

    class Config:
        orm_mode = True