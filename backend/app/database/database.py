# app/database/database.py

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# Получение DATABASE_URL из переменных окружения
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

if not SQLALCHEMY_DATABASE_URL:
    raise ValueError("DATABASE_URL не установлена в переменных окружения")

# Создание движка SQLAlchemy с явной установкой кодировки UTF-8
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"options": "-c client_encoding=utf8"}
)

# Создание сессии
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Базовый класс для всех моделей
Base = declarative_base()

# Отложенный импорт моделей при необходимости (например, в Alembic)
def get_base_metadata():
    # Импортируем модели только когда это необходимо (например, для миграций или инициализации базы данных)
    from app.models.store import Store
    from app.models.product import Product
    from app.models.unit import Unit
    from app.models.product_store_data import ProductStoreData
    from app.models.product_matching_log import ProductMatchingLog
    # Импортируйте другие модели здесь
    return Base.metadata
