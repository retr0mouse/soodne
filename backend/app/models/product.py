from sqlalchemy import Column, Integer, String, ForeignKey, Text, DECIMAL, TIMESTAMP, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.models.base import Base
from app.schemas.enums import MatchingStatusEnum

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
    store_id = Column(Integer, ForeignKey("stores.store_id"))
    matching_status = Column(Enum(MatchingStatusEnum), default=MatchingStatusEnum.unmatched)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=func.now())
    
    category = relationship("Category", backref="products")
    unit = relationship("Unit", back_populates="products")
    store = relationship("Store", back_populates="products")
    prices = relationship("ProductPrice", back_populates="product", cascade="all, delete-orphan")
