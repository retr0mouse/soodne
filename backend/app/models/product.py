from sqlalchemy import Column, Integer, String, ForeignKey, Text, DECIMAL, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.models.base import Base

class Product(Base):
    __tablename__ = "products"

    product_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    image_url = Column(Text)
    weight_value = Column(DECIMAL(10, 2))
    unit_id = Column(Integer, ForeignKey("units.unit_id"))
    category_id = Column(Integer, ForeignKey("categories.category_id"))
    barcode = Column(String(50), unique=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=func.now())
    
    category = relationship("Category", backref="products")
    unit = relationship("Unit", back_populates="products")
    store_data = relationship("ProductStoreData", back_populates="product", cascade="all, delete-orphan")
