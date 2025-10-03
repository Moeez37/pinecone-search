from enum import StrEnum

class OBJECT_TYPE(StrEnum):
    PRODUCTS = "products"
    BLOGS = "blogs"
    STORES = "stores"
    ALL = "all"

class STORE_TYPE(StrEnum):
    RECREATIONAL = "recreational"
    MEDICAL = "medical"
    BOTH = "both"