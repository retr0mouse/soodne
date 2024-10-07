from sqlalchemy import Column, Integer, DECIMAL, ForeignKey, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.database import Base

class ProductPriceHistory(Base):
    __tablename__ = "productpricehistory"  # Соответствует таблице в БД

    price_history_id = Column(Integer, primary_key=True, index=True)
    product_store_id = Column(Integer, ForeignKey("productstoredata.product_store_id", ondelete="CASCADE"), nullable=False)
    price = Column(DECIMAL(10, 2), nullable=False)
    recorded_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    product_store_data = relationship("ProductStoreData", back_populates="price_history")
