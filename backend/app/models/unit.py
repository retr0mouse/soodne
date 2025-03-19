from sqlalchemy import Column, Integer, String, DECIMAL, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.database import Base

class Unit(Base):
    __tablename__ = "units"

    unit_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)
    conversion_factor = Column(DECIMAL(10, 6), nullable=False)

    products = relationship("Product", back_populates="unit")
    product_store_data = relationship("ProductStoreData", back_populates="store_unit")
