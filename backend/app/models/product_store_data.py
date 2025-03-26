from sqlalchemy import (
    Column, Integer, DECIMAL, ForeignKey, String, Text, TIMESTAMP, JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy import func
from app.database.database import Base

class ProductStoreData(Base):
    __tablename__ = "product_store_data"

    product_store_id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.product_id", ondelete="CASCADE"), nullable=True)
    store_id = Column(Integer, ForeignKey("stores.store_id", ondelete="CASCADE"), nullable=False)
    price = Column(DECIMAL(10, 2), nullable=False)
    price_per_unit = Column(DECIMAL(10, 2), nullable=True)
    last_updated = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    store_product_name = Column(String(255))
    store_image_url = Column(Text, nullable=True)
    store_product_url = Column(Text, nullable=True)
    store_category_id = Column(Integer, ForeignKey("categories.category_id", ondelete="SET NULL"), nullable=True)
    store_weight_value = Column(DECIMAL(10, 2), nullable=True)
    store_unit_id = Column(Integer, ForeignKey("units.unit_id"))
    ean = Column(String(13), nullable=True, index=True)
    additional_attributes = Column(JSON, nullable=True)
    last_matched = Column(TIMESTAMP(timezone=True), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=func.now())

    product = relationship("Product", back_populates="store_data")
    store = relationship("Store", back_populates="product_store_data")
    store_unit = relationship("Unit", back_populates="product_store_data")
    store_category = relationship("Category", foreign_keys=[store_category_id])
    price_history = relationship("ProductPriceHistory", back_populates="product_store_data")
    matching_logs = relationship("ProductMatchingLog", back_populates="product_store_data")
