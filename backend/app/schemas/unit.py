from pydantic import BaseModel

class UnitBase(BaseModel):
    name: str
    conversion_factor: float

class UnitCreate(UnitBase):
    pass

class Unit(UnitBase):
    unit_id: int

    class Config:
        orm_mode = True
