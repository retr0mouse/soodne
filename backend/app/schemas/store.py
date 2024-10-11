from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class StoreBase(BaseModel):
    name: str
    website_url: Optional[str] = None
    image_url: Optional[str] = None

class StoreCreate(StoreBase):
    pass

class Store(StoreBase):
    store_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True