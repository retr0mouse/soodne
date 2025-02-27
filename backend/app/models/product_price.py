from sqlalchemy import Column, Integer, DECIMAL, ForeignKey, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.database import Base

class ProductPrice(Base):
    __tablename__ = "product_prices"

    price_id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.product_id", ondelete="CASCADE"), nullable=False)
    store_id = Column(Integer, ForeignKey("stores.store_id", ondelete="CASCADE"), nullable=False)
    price = Column(DECIMAL(10, 2), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    product = relationship("Product", back_populates="prices")
    store = relationship("Store", back_populates="prices") 