# app/models.py
from sqlalchemy import Column, Integer, String, DECIMAL, ForeignKey, Text, TIMESTAMP, Enum, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base
import enum

class MatchingStatusEnum(enum.Enum):
    unmatched = "unmatched"
    matched = "matched"
    pending = "pending"

class Unit(Base):
    __tablename__ = "Units"

    unit_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)
    conversion_factor = Column(DECIMAL(10, 6), nullable=False)
    created_at = Column(TIMESTAMP(timezone=False), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=False), server_default=func.now(), onupdate=func.now())

    products = relationship("Product", back_populates="unit")
    product_store_data = relationship("ProductStoreData", back_populates="store_unit")

class Store(Base):
    __tablename__ = "Stores"

    store_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)
    website_url = Column(Text)
    image_url = Column(Text)
    created_at = Column(TIMESTAMP(timezone=False), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=False), server_default=func.now(), onupdate=func.now())

    product_store_data = relationship("ProductStoreData", back_populates="store")

class Category(Base):
    __tablename__ = "Categories"

    category_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    parent_id = Column(Integer, ForeignKey("Categories.category_id", ondelete="SET NULL", onupdate="CASCADE"), nullable=True)
    created_at = Column(TIMESTAMP(timezone=False), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=False), server_default=func.now(), onupdate=func.now())

    parent = relationship("Category", remote_side=[category_id])
    products = relationship("Product", back_populates="category")

class Product(Base):
    __tablename__ = "Products"

    product_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    image_url = Column(Text)
    weight_value = Column(DECIMAL(10, 2), nullable=True)
    unit_id = Column(Integer, ForeignKey("Units.unit_id"))
    category_id = Column(Integer, ForeignKey("Categories.category_id", ondelete="SET NULL", onupdate="CASCADE"))
    barcode = Column(String(50), unique=True, nullable=True)
    created_at = Column(TIMESTAMP(timezone=False), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=False), server_default=func.now(), onupdate=func.now())

    unit = relationship("Unit", back_populates="products")
    category = relationship("Category", back_populates="products")
    product_store_data = relationship("ProductStoreData", back_populates="product")

class ProductStoreData(Base):
    __tablename__ = "ProductStoreData"

    product_store_id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("Products.product_id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    store_id = Column(Integer, ForeignKey("Stores.store_id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    price = Column(DECIMAL(10, 2), nullable=False)
    price_per_unit = Column(DECIMAL(10, 2), nullable=True)
    last_updated = Column(TIMESTAMP(timezone=False), server_default=func.now())
    store_product_name = Column(String(255))
    store_description = Column(Text)
    store_image_url = Column(Text)
    store_weight_value = Column(DECIMAL(10, 2), nullable=True)
    store_unit_id = Column(Integer, ForeignKey("Units.unit_id"))
    additional_attributes = Column(JSON, nullable=True)
    matching_status = Column(Enum(MatchingStatusEnum), default=MatchingStatusEnum.unmatched)
    last_matched = Column(TIMESTAMP(timezone=False), nullable=True)
    created_at = Column(TIMESTAMP(timezone=False), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=False), server_default=func.now(), onupdate=func.now())

    product = relationship("Product", back_populates="product_store_data")
    store = relationship("Store", back_populates="product_store_data")
    store_unit = relationship("Unit", back_populates="product_store_data")

    price_history = relationship("ProductPriceHistory", back_populates="product_store_data")
    matching_logs = relationship("ProductMatchingLog", back_populates="product_store_data")

class ProductPriceHistory(Base):
    __tablename__ = "ProductPriceHistory"

    price_history_id = Column(Integer, primary_key=True, index=True)
    product_store_id = Column(Integer, ForeignKey("ProductStoreData.product_store_id", ondelete="CASCADE"), nullable=False)
    price = Column(DECIMAL(10, 2), nullable=False)
    recorded_at = Column(TIMESTAMP(timezone=False), server_default=func.now())

    product_store_data = relationship("ProductStoreData", back_populates="price_history")

class ProductMatchingLog(Base):
    __tablename__ = "ProductMatchingLog"

    log_id = Column(Integer, primary_key=True, index=True)
    product_store_id = Column(Integer, ForeignKey("ProductStoreData.product_store_id"), nullable=False)
    product_id = Column(Integer, ForeignKey("Products.product_id"), nullable=True)
    confidence_score = Column(DECIMAL(5, 2), nullable=True)
    matched_at = Column(TIMESTAMP(timezone=False), server_default=func.now())
    matched_by = Column(String(50), nullable=True)

    product_store_data = relationship("ProductStoreData", back_populates="matching_logs")
    product = relationship("Product")
