from pydantic import BaseModel
from typing import Optional, List
from enum import Enum
from datetime import datetime

class MatchingStatusEnum(str, Enum):
    unmatched = "unmatched"
    matched = "matched"
    pending = "pending"

class UnitBase(BaseModel):
    name: str
    conversion_factor: float

class UnitCreate(UnitBase):
    pass

class Unit(UnitBase):
    unit_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class StoreBase(BaseModel):
    name: str
    website_url: Optional[str] = None
    image_url: Optional[str] = None

class StoreCreate(StoreBase):
    pass

class Store(StoreBase):
    store_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class CategoryBase(BaseModel):
    name: str
    parent_id: Optional[int] = None

class CategoryCreate(CategoryBase):
    pass

class Category(CategoryBase):
    category_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

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

class ProductStoreDataBase(BaseModel):
    product_id: int
    store_id: int
    price: float
    price_per_unit: Optional[float] = None
    store_product_name: Optional[str] = None
    store_description: Optional[str] = None
    store_image_url: Optional[str] = None
    store_weight_value: Optional[float] = None
    store_unit_id: Optional[int] = None
    additional_attributes: Optional[dict] = None
    matching_status: Optional[MatchingStatusEnum] = MatchingStatusEnum.unmatched
    last_matched: Optional[datetime] = None

class ProductStoreDataCreate(ProductStoreDataBase):
    pass

class ProductStoreData(ProductStoreDataBase):
    product_store_id: int
    last_updated: datetime
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class ProductPriceHistoryBase(BaseModel):
    product_store_id: int
    price: float

class ProductPriceHistoryCreate(ProductPriceHistoryBase):
    pass

class ProductPriceHistory(ProductPriceHistoryBase):
    price_history_id: int
    recorded_at: datetime

    class Config:
        orm_mode = True

class ProductMatchingLogBase(BaseModel):
    product_store_id: int
    product_id: Optional[int] = None
    confidence_score: Optional[float] = None
    matched_by: Optional[str] = None

class ProductMatchingLogCreate(ProductMatchingLogBase):
    pass

class ProductMatchingLog(ProductMatchingLogBase):
    log_id: int
    matched_at: datetime

    class Config:
        orm_mode = True