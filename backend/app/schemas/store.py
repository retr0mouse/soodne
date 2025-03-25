from pydantic import BaseModel
from typing import Optional

class StoreBase(BaseModel):
    name: str
    website_url: Optional[str] = None
    image_url: Optional[str] = None

class StoreCreate(StoreBase):
    pass

class Store(StoreBase):
    store_id: int

    class Config:
        from_attributes = True