import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

if not SQLALCHEMY_DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in environment variables")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"options": "-c client_encoding=utf8"}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_base_metadata():
    from app.models.store import Store
    from app.models.product import Product
    from app.models.unit import Unit
    from app.models.product_store_data import ProductStoreData
    from app.models.product_matching_log import ProductMatchingLog
    return Base.metadata
