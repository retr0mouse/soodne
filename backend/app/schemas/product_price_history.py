
from pydantic import BaseModel
from datetime import datetime

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
