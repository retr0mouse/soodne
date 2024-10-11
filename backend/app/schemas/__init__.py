# app/schemas/__init__.py

from .unit import Unit, UnitCreate
from .store import Store, StoreCreate
from .category import Category, CategoryCreate
from .product import Product, ProductCreate
from .product_store_data import ProductStoreData, ProductStoreDataCreate
from .product_price_history import ProductPriceHistory, ProductPriceHistoryCreate
from .product_matching_log import ProductMatchingLog, ProductMatchingLogCreate
from .enums import MatchingStatusEnum
from .job_status import JobStatus  # Добавлено

__all__ = [
    "Unit", "UnitCreate",
    "Store", "StoreCreate",
    "Category", "CategoryCreate",
    "Product", "ProductCreate",
    "ProductStoreData", "ProductStoreDataCreate",
    "ProductPriceHistory", "ProductPriceHistoryCreate",
    "ProductMatchingLog", "ProductMatchingLogCreate",
    "MatchingStatusEnum",
    "JobStatus",  # Добавлено
]
