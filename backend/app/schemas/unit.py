from pydantic import BaseModel
from datetime import datetime

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
