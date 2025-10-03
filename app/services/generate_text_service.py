from app.utils.constants import OBJECT_TYPE
from app.utils.common import collect_values, safe_get, is_nonempty
from pydantic import BaseModel
from enum import Enum
import re
from app.utils.logger import get_logger

logger = get_logger(__name__)
class GenerateTextService:
    def __init__(self):
        pass
    
    def generate_text(self, metadata: dict, object_type: OBJECT_TYPE) -> str:
        """Generate text for a single metadata, using cache if available"""
        try:
            match object_type:
                case OBJECT_TYPE.PRODUCTS:
                    return self.generate_product_text(metadata)
                case OBJECT_TYPE.BLOGS:
                    return self.generate_blog_text(metadata)
                case OBJECT_TYPE.STORES:
                    return self.generate_store_text(metadata)
                case _:
                    raise ValueError(f"Unknown object type: {object_type}")
        except Exception as e:
            logger.error("Error generating text", error=str(e))
            raise
    
    def generate_product_text(self, metadata: dict) -> str:
        """Generate product text for a single metadata, using cache if available"""
        try:
            return self.product_to_semantic_texts(metadata)
        except Exception as e:
            logger.error("Error generating product text", error=str(e))
            raise
    
    def generate_blog_text(self, metadata: dict) -> str:
        """Generate blog text for a single metadata, using cache if available"""
        try:
            return self.post_to_semantic_texts(metadata)
        except Exception as e:
            logger.error("Error generating blog text", error=str(e))
            raise
    
    def generate_store_text(self, metadata: dict) -> str:
        """Generate store text for a single metadata, using cache if available"""
        try:
            return self.store_to_text(metadata)
        except Exception as e:
            logger.error("Error generating store text", error=str(e))
            raise
    
    def store_to_semantic_texts(self, metadata: dict) -> str:
        """Generate store text for a single metadata, using cache if available"""
        try:
            pass
        except Exception as e:
            logger.error("Error generating store text", error=str(e))
            raise
        
    @staticmethod
    def product_to_semantic_texts(product: dict[str, any]) -> (str, str):
        """
        Given a product dict, returns:
        - natural language description
        - values-only string
        
        Missing/None/empty fields are skipped.
        """
        
        parts: List[str] = []
        
        name = product.get("name")
        if is_nonempty(name):
            parts.append(f"{name}")
        
        prod_type = product.get("type")
        if is_nonempty(prod_type):
            parts.append(f"is a {prod_type} product")
        
        brand = safe_get(product, "brand", "name")
        if is_nonempty(brand):
            parts.append(f"by {brand}")
        
        strain_prev = safe_get(product, "strain", "prevalence", "name")
        if is_nonempty(strain_prev):
            parts.append(f"of type {strain_prev}")
        
        category = safe_get(product, "category", "name")
        subcategory = safe_get(product, "subcategory", "name")
        if is_nonempty(category):
            if is_nonempty(subcategory):
                parts.append(f"in category {category} > {subcategory}")
            else:
                parts.append(f"in category {category}")
        
        # Variant / lab test / price
        variant_list = product.get("variants", [])
        if variant_list and isinstance(variant_list, list):
            var0 = variant_list[0]
            vname = var0.get("name")
            if is_nonempty(vname):
                parts.append(f"available as {vname}")
            
            # THC / CBD
            thc = safe_get(var0, "labTests", "thc", "value")
            # Usually a list for “value”, pick first if exists
            if isinstance(thc, list) and thc:
                parts.append(f"THC approx {thc[0]}%")
            elif is_nonempty(thc):
                parts.append(f"THC approx {thc}%")
            
            cbd = safe_get(var0, "labTests", "cbd", "value")
            if isinstance(cbd, list) and cbd:
                parts.append(f"CBD approx {cbd[0]}%")
            elif is_nonempty(cbd):
                parts.append(f"CBD approx {cbd}%")
            
            price = var0.get("price")
            if is_nonempty(price):
                parts.append(f"price {price}")
            
            promo = var0.get("promoPrice")
            if is_nonempty(promo):
                parts.append(f"promo {promo}")
        
        # Terpenes (top ones)
        terpenes = safe_get(product, "strain", "terpenes")
        if isinstance(terpenes, list) and terpenes:
            # take first few to avoid overly long text
            terp_names = [t.get("name") for t in terpenes if is_nonempty(t.get("name"))]
            if terp_names:
                top = terp_names[:5]  # pick first 5
                parts.append("terpenes: " + ", ".join(top))
        
        # Combine into description
        description = ". ".join(parts) + "."
        
        # Build values-only string with filtering
        value_parts: List[str] = []
        value_parts = collect_values(product)
        values_string = " ".join(value_parts)
        return description + " " + values_string
    @staticmethod
    def post_to_semantic_texts(post: dict[str, any]) -> (str, str):
        """
        Given a blog post dict, returns:
        - natural language description
        - values-only string
        
        Missing/None/empty fields are skipped.
        """
        parts: List[str] = []
        
        title = post.get("title")
        if is_nonempty(title):
            parts.append(f"Blog post titled '{title}'")
        
        excerpt = post.get("excerpt")
        if is_nonempty(excerpt):
            # strip HTML tags if needed
            import re
            clean_excerpt = re.sub(r"<[^>]*>", "", excerpt).strip()
            parts.append(f"Excerpt: {clean_excerpt}")
        
        author = safe_get(post, "author", "node", "nickname")
        if is_nonempty(author):
            parts.append(f"written by {author}")
        
        date = post.get("date")
        if is_nonempty(date):
            parts.append(f"published on {date}")
        
        category_edges = safe_get(post, "categories", "edges", default=[])
        if category_edges and isinstance(category_edges, list):
            cat_names = [safe_get(edge, "node", "name") for edge in category_edges]
            cat_names = [c for c in cat_names if is_nonempty(c)]
            if cat_names:
                parts.append(f"in categories: {', '.join(cat_names)}")
        
        featured_alt = safe_get(post, "featuredImage", "node", "altText")
        if is_nonempty(featured_alt):
            parts.append(f"featured image showing {featured_alt}")
        
        description = ". ".join(parts) + "."

        # Values-only collector
        value_parts: List[str] = []
        value_parts = collect_values(post)
        values_string = " ".join(value_parts)

        return description +" "+ values_string
    @staticmethod
    def store_to_text(store: dict) -> str:
        """Convert store dict into a clean string for embeddings (semantic search)."""
        parts = []

        # Title / Slug
        if store.get("title"):
            parts.append(f"Store: {store['title']}")
        if store.get("slug"):
            parts.append(f"Slug: {store['slug']}")

        # Status
        if store.get("location_status"):
            parts.append(f"Status: {store['location_status']}")

        # Address
        address_bits = []
        if store.get("address_1"):
            address_bits.append(store["address_1"])
        if store.get("city"):
            address_bits.append(store["city"])
        if store.get("state", {}).get("abbr"):
            address_bits.append(store["state"]["abbr"])
        if store.get("zip"):
            address_bits.append(store["zip"])
        if address_bits:
            parts.append("Address: " + ", ".join(address_bits))

        # Type
        if store.get("type"):
            parts.append(f"Type: {store['type']}")

        # IDs
        if store.get("medicalStoreId"):
            parts.append(f"Medical Store ID: {store['medicalStoreId']}")
        if store.get("recreationalStoreId"):
            parts.append(f"Recreational Store ID: {store['recreationalStoreId']}")

        # Hours (just include days that exist)
        if "hours" in store and isinstance(store["hours"], dict):
            hours_parts = []
            for day in [
                "monday", "tuesday", "wednesday",
                "thursday", "friday", "saturday", "sunday"
            ]:
                if day in store["hours"]:
                    open_t = store["hours"][day].get("open")
                    close_t = store["hours"][day].get("closed")
                    if open_t and close_t:
                        hours_parts.append(f"{day.title()} {open_t} - {close_t}")
            if hours_parts:
                parts.append("Hours: " + "; ".join(hours_parts))

        return " | ".join(parts)
