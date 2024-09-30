from sqlalchemy import Column, Integer, String, Text, DECIMAL, ForeignKey, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.database import Base

class Product(Base):
    __tablename__ = "products"

    product_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    image_url = Column(Text)
    weight_value = Column(DECIMAL(10, 2), nullable=True)
    unit_id = Column(Integer, ForeignKey("units.unit_id"))
    category_id = Column(Integer, ForeignKey("categories.category_id", ondelete="SET NULL"))
    barcode = Column(String(50), unique=True, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=func.now())

    unit = relationship("Unit", back_populates="products")
    category = relationship("Category", back_populates="products")
    product_store_data = relationship("ProductStoreData", back_populates="product")
