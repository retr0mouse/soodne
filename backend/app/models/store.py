from sqlalchemy import Column, Integer, String, Text, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.database import Base

class Store(Base):
    __tablename__ = "stores"

    store_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)
    website_url = Column(Text)
    image_url = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=func.now())

    products = relationship("Product", back_populates="store")
    prices = relationship("ProductPrice", back_populates="store")
