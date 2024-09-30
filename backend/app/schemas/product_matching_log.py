from pydantic import BaseModel
from typing import Optional
from datetime import datetime

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
