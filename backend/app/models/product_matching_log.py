from sqlalchemy import Column, Integer, DECIMAL, ForeignKey, String, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.database import Base

class ProductMatchingLog(Base):
    __tablename__ = "product_matching_log"

    log_id = Column(Integer, primary_key=True, index=True)
    product_id1 = Column(Integer, ForeignKey("products.product_id"), nullable=False)
    product_id2 = Column(Integer, ForeignKey("products.product_id"), nullable=False)
    confidence_score = Column(DECIMAL(5, 2), nullable=True)
    matched_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    matched_by = Column(String(50), nullable=True)

    product1 = relationship("Product", foreign_keys=[product_id1])
    product2 = relationship("Product", foreign_keys=[product_id2])
