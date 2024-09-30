from .products import router as products_router
from .stores import router as stores_router
from .matching import router as matching_router

__all__ = ["products_router", "stores_router", "matching_router"]
