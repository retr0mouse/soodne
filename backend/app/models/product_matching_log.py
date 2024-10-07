from sqlalchemy import Column, Integer, DECIMAL, ForeignKey, String, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.database import Base

class ProductMatchingLog(Base):
    __tablename__ = "productmatchinglog"  # Соответствует таблице в БД

    log_id = Column(Integer, primary_key=True, index=True)
    product_store_id = Column(Integer, ForeignKey("productstoredata.product_store_id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.product_id"), nullable=True)
    confidence_score = Column(DECIMAL(5, 2), nullable=True)
    matched_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    matched_by = Column(String(50), nullable=True)

    product_store_data = relationship("ProductStoreData", back_populates="matching_logs")
    product = relationship("Product")
