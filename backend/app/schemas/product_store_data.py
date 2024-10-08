from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime
from .enums import MatchingStatusEnum

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
    additional_attributes: Optional[Dict] = None
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
