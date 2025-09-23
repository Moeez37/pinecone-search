import openai
from typing import List, Dict, Any, Optional
from config import settings
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingsService:
    def __init__(self):
        # Initialize OpenAI client
        if settings.OPENAI_API_KEY:
            self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            self.openai_client = None
            raise ValueError("OpenAI API key not configured")
            
        # Set OpenAI models
        self.embedding_model = settings.OPENAI_EMBEDDING_MODEL
        self.chat_model = settings.OPENAI_CHAT_MODEL
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using OpenAI"""
        try:
            # Clean and prepare text
            cleaned_text = text.strip().replace("\n", " ")
            if not cleaned_text:
                raise ValueError("Empty text provided")
            
            if not self.openai_client:
                raise ValueError("OpenAI API key not configured")
            
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=cleaned_text
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch:
                try:
                    embedding = self.generate_embedding(text)
                    batch_embeddings.append(embedding)
                    # Small delay to respect rate limits
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error generating embedding for text: {text[:50]}... Error: {e}")
                    # Use zero vector as fallback
                    batch_embeddings.append([0.0] * settings.EMBEDDING_DIMENSION)
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def generate_namespace(self, search_type: str, location_id: str = None) -> str:
        """Generate namespace based on search type and location"""
        if search_type == "products":
            return "product"
        elif search_type == "stores":
            return "store"
        elif search_type == "blogs":
            return "blog"
        else:
            return search_type

    from typing import Any, Dict, Union

    def _clean_metadata(self, metadata: Union[Dict[str, Any], list, Any]) -> Union[Dict[str, Any], list, Any]:
        """Clean metadata to ensure Pinecone compatibility:
        - Removes None, empty strings, empty lists/dicts
        - Keeps only allowed types: str, int, float, bool, list[str]
        """

        if isinstance(metadata, dict):
            cleaned = {}
            for key, value in metadata.items():
                cleaned_value = self._clean_metadata(value)

                # Skip None, empty, or invalid
                if cleaned_value in (None, "", [], {}):
                    continue

                cleaned[key] = cleaned_value
            return cleaned

        elif isinstance(metadata, list):
            cleaned_list = []
            for item in metadata:
                cleaned_item = self._clean_metadata(item)
                # Only allow strings in lists
                if isinstance(cleaned_item, str) and cleaned_item.strip():
                    cleaned_list.append(cleaned_item)

            return cleaned_list if cleaned_list else None

        # Allowed scalar types
        elif isinstance(metadata, (str, int, float, bool)):
            return metadata

        # Drop everything else
        return None

    
    def rewrite_query(self, query: str) -> str:
        """Use GPT to rewrite and expand user queries for better search"""
        try:
            system_prompt = """
You are an advanced semantic search-engine AI specialized in retrieving and synthesizing high-quality, personalized results from a database or knowledge base. Your mission: given a user query and optional context, return the most relevant, blended results (educational + product) in a compact JSON structure suitable for downstream processing.

INPUT VARIABLES (replace placeholders):
- query: {query}                 // raw user query, e.g. "best laptops for coding under $1000"
- user_context: { "location": "pakistan", "preferences": {...} } // optional personalization info
- max_results: 12                // limit to this many results
- filters: { ... }               // optional filters (date, category, price_range, etc.)

HIGH-PRIORITY CAPABILITIES
1. Semantic understanding: interpret intent, synonyms, and implied context (use embeddings/ontology-style expansions).
2. Multi-word logic: honor AND, OR, NOT, quoted exact phrases, wildcards (*), parentheses and precedence.
3. Educational + product blending: produce ~40–60% educational content and ~40–60% product items unless user intent strongly favors one type.
4. Unified location handling: treat addresses, cities, regions, and nearby results consistently; merge geo-data where relevant.
5. Personalization: boost results using user_context (location, past prefs). If no context, return globally relevant items and include a personalization suggestion.
6. Voice-readiness: normalize conversational/voice queries (remove filler words, expand “wanna” → “want to”, etc.).
7. Output must be machine-friendly JSON only (no additional text).

PROCESSING PIPELINE (step-by-step)
A. Parse & normalize:
   - Extract logical operators, quoted phrases, numeric constraints (e.g., "$1000", "under 5 km").
   - Detect voice/colloquial patterns and normalize.
   - If ambiguous (>=2 plausible intents), include a top-level "clarity_needed": true and propose 2–3 disambiguation queries in the JSON; still attempt best-effort results.

B. Expand semantically:
   - Map core terms to synonyms and related concepts (e.g., car → automobile, vehicle).
   - Expand multi-word entities (e.g., "python course" → "Python tutorial, bootcamp, video series").

C. Apply filters & personalization:
   - Apply explicit filters.
   - Apply personalization boosts (location proximity, preferred brands, price sensitivity).
   - Record which personalization rules affected ranking (for transparency).

D. Retrieve, rank & balance:
   - Retrieve candidate items (educational articles, tutorials, explainers; products with buy options; local venues).
   - Score by semantic similarity, freshness, quality, and personalization boost.
   - Enforce the educational/product blend target unless query intent indicates otherwise.

E. Construct output:
   - Return up to max_results items sorted by relevance_score (0.0–1.0).
   - Provide metadata for each item (type, concise summary, rationale for personalization, tags).
   - Provide a short "explainers" section describing how results were ranked (one-sentence summary).
   - Include "suggestions" (2–5 alternative queries or filters).

OUTPUT SCHEMA (strict JSON — no extra text)
{
  "query_normalized": "string",
  "clarity_needed": boolean,
  "clarification_queries": ["optional string", ...],
  "results": [
    {
      "title": "string",
      "type": "educational" | "product" | "local" | "mixed",
      "summary": "brief description (<=25 words)",
      "link": "URL or null",
      "location": "city/coords or null",
      "price": { "amount": number, "currency": "PKR|USD|..." } | null,
      "relevance_score": 0.0-1.0,
      "personalization_note": "why this matches user context",
      "tags": ["tutorial","beginner","laptop", ...]
    }
  ],
  "total_results": integer,
  "ranking_explainer": "short string (<=30 words)",
  "suggestions": ["alternate query 1", "alternate query 2", ...]
}

ADDITIONAL RULES & EXAMPLES
- Honor logical operators strictly: "apple AND fruit NOT tech" must demote tech-company results.
- For location-aware queries, include at least one local result when relevant and mark distance if known.
- If the query requests prices or purchase links, include the most relevant and indicate currency and approximate availability.
- If no suitable results found, return an empty results array and up to 3 alternative suggestions.
- Example: if query is voice-like "um best laptops for coding under a grand", normalize to "best laptops for coding under $1000" and set query_normalized accordingly.

FINAL INSTRUCTION:
Process the supplied input now and return ONLY the JSON structure that conforms to the schema above. Do not output any explanatory text outside the JSON.
"""
            
            if not self.openai_client:
                raise ValueError("OpenAI API key not configured")
            
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=100,
                temperature=0.3
            )
            rewritten_query = response.choices[0].message.content.strip()
            logger.info(f"Query rewritten: '{query}' → '{rewritten_query}'")
            
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            # Return original query if rewriting fails
            return query
    
    def prepare_product_text(self, product: Dict[str, Any]) -> str:
        """Prepare product data for embedding with key-value format"""
        text_parts = []
        # Add key-value pairs for better semantic understanding
        if product.get('unique_slug'):
            text_parts.append(f"slug: {product.get('unique_slug')},")
        if product.get('brand'):
            text_parts.append(f"brand: {product.get('brand')},")
        if product.get('kind'):
            text_parts.append(f"type: {product.get('kind')},")
        if product.get('kind_subtype'):
            text_parts.append(f"subtype: {product.get('kind_subtype')},")
        if product.get('custom_product_type'):
            text_parts.append(f"category: {product.get('custom_product_type')},")
        if product.get('custom_product_subtype'):
            text_parts.append(f"subcategory: {product.get('custom_product_subtype')},")
        if product.get('description'):
            text_parts.append(f"description: {product.get('description')},")
        if product.get('strain'):
            text_parts.append(f"strain: {product.get('strain')},")
        if product.get('effects'):
            text_parts.append(f"effects: {' '.join(product.get('effects', []))},")
        if product.get('flavors'):
            text_parts.append(f"flavour: {' '.join(product.get('flavors', []))},")
        if product.get('feelings'):
            text_parts.append(f"feelings: {' '.join(product.get('feelings', []))},")
        if product.get('activities'):
            text_parts.append(f"activities: {' '.join(product.get('activities', []))},")
        if product.get('name'):
            text_parts.append(f"name: {' '.join(product.get('name', []))}")
        if product.get('percent_thc'):
            text_parts.append(f"percent_thc: {product.get('percent_thc')}")
        if product.get('percent_cbd'):
            text_parts.append(f"percent_cbd: {product.get('percent_cbd')}")
        if product.get("available_for_pickup"):
            text_parts.append(f"available_for_pickup: {'this product is available for store pickup' if product.get('available_for_pickup') == 'true' else 'pickup not available'}")
        if product.get("available_for_delivery"):
            text_parts.append(f"available_for_delivery: {'this product can be delivered to your location' if product.get('available_for_delivery') == 'true' else 'delivery not available'}")
        if product.get("review_count"):
            text_parts.append(f"review_count: {product.get('review_count')}")
        if product.get("aggregate_rating"):
            text_parts.append(f"aggregate_rating: {product.get('aggregate_rating')}")
        return ' '.join(text_parts)
    
    def prepare_blog_text(self, blog: Dict[str, Any]) -> str:
        """Prepare blog data for embedding"""
        # Extract categories from the nested structure
        categories = []
        if blog.get('categories', {}).get('edges'):
            categories = [edge['node']['name'] for edge in blog['categories']['edges']]
        
        # Extract author name
        author_name = ''
        if blog.get('author', {}).get('node', {}).get('nickname'):
            author_name = blog['author']['node']['nickname']
        
        text_parts = []
        
        if blog.get('title'):
            text_parts.append(f"title: {blog['title']}")
        if blog.get('excerpt'):
            excerpt = blog['excerpt'].replace('<p>', '').replace('</p>', '').replace('\n', ' ')
            text_parts.append(f"excerpt: {excerpt}")
        if blog.get('slug'):
            text_parts.append(f"slug: {blog['slug']}")
        if author_name:
            text_parts.append(f"author: {author_name}")
        if categories:
            text_parts.append(f"categories: {' '.join(categories)}")
        
        return ' '.join(text_parts)
    
    def create_product_vector(self, product: Dict[str, Any], store_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a vector entry for a product"""
        text = self.prepare_product_text(product)
        embedding = self.generate_embedding(text)
        # Extract location from _geoloc if available
        location = None
        if product.get('_geoloc'):
            location = f"{product['_geoloc'].get('lat', '')},{product['_geoloc'].get('lng', '')}"

        metadata = {
            "image_url": product.get("image_url",""),
            "type": "product",
            "name": product.get('name', ''),
            "product_id": product['productId'] or product.get("productId"),
            "unique_slug": product.get('unique_slug', ''),
            "brand": product.get('brand', ''),
            "kind": product.get('kind', ''),
            "kind_subtype": product.get('kind_subtype', ''),
            "custom_product_type": product.get('custom_product_type', ''),
            "custom_product_subtype": product.get('custom_product_subtype', ''),
            "price": product.get('bucket_price') or product.get('sort_price') or product.get('price_gram'),
            "percent_thc": product.get('percent_thc', 0),
            "percent_cbd": product.get('percent_cbd', 0),
            "strain": product.get('strain', ''),
            "store_notes": product.get('store_notes') or product.get('description', ''),
            "effects": product.get('effects', []),
            "flavors": product.get('flavors', []),
            "feelings": product.get('feelings', []),
            "activities": product.get('activities', []),
            "root_types": product.get('root_types', []),
            "store_types": product.get('store_types', []),
            "available_for_delivery": product.get('available_for_delivery', False),
            "available_for_pickup": product.get('available_for_pickup', False),
            "location": location,
            "locationId": store_info.get('locationId', '') if store_info else '',
            "store_name": store_info.get('title', '') if store_info else '',
            "store_city": store_info.get('city', '') if store_info else '',
            "store_state": store_info.get('state', {}).get('abbr', '') if store_info else '',
            "storePlatform":store_info.get("storePlatform", ""),
            "store_id":product.get("store_id","")
        }
        return {
            "id": f"product_{product['productId']}",
            "values": embedding,
            "metadata": self._clean_metadata(metadata)
        }
    
    def create_blog_vector(self, blog: Dict[str, Any]) -> Dict[str, Any]:
        """Create a vector entry for a blog"""
        text = self.prepare_blog_text(blog)
        embedding = self.generate_embedding(text)
        
        # Extract categories from the nested structure
        categories = []
        if blog.get('categories', {}).get('edges'):
            categories = [edge['node']['name'] for edge in blog['categories']['edges']]
        
        # Extract author information
        author_name = ''
        author_slug = ''
        if blog.get('author', {}).get('node'):
            author_node = blog['author']['node']
            author_name = author_node.get('nickname', '')
            author_slug = author_node.get('slug', '')
        
        # Extract featured image URL
        featured_image_url = ''
        if blog.get('featuredImage', {}).get('node', {}).get('sourceUrl'):
            featured_image_url = blog['featuredImage']['node']['sourceUrl']
        
        # Clean excerpt
        excerpt = blog.get('excerpt', '').replace('<p>', '').replace('</p>', '').replace('\n', ' ').strip()
        
        return {
            "id": f"blog_{blog['databaseId']}",
            "values": embedding,
            "metadata": self._clean_metadata({
                "type": "blog",
                "title": blog.get('title', ''),
                "excerpt": excerpt,
                "author_name": author_name,
                "author_slug": author_slug,
                "featured_image_url": featured_image_url,
                "categories": categories,
                "slug": blog.get('slug', ''),
                "databaseId": blog.get('databaseId', ''),
                "createdAt": blog.get('createdAt', ''),
                "updatedAt": blog.get('updatedAt', ''),
            })
        }
    
    def prepare_store_text(self, store: Dict[str, Any]) -> str:
        """Prepare store data for embedding"""
        # Extract state information
        state_name = ''
        state_abbr = ''
        if store.get('state'):
            state_name = store['state'].get('post_title', '')
            state_abbr = store['state'].get('abbr', '')
        
        text_parts = []
        
        if store.get('title'):
            text_parts.append(f"store: {store['title']}")
        if store.get('post_title'):
            text_parts.append(f"name: {store['post_title']}")
        if store.get('slug'):
            text_parts.append(f"slug: {store['slug']}")
        if store.get('locationId'):
            text_parts.append(f"location_id: {store['locationId']}")
        if store.get('address_1'):
            text_parts.append(f"address: {store['address_1']}")
        if store.get('city'):
            text_parts.append(f"city: {store['city']}")
        if state_name:
            text_parts.append(f"state: {state_name}")
        if state_abbr:
            text_parts.append(f"state_code: {state_abbr}")
        if store.get('zip'):
            text_parts.append(f"zip: {store['zip']}")
        if store.get('location_status'):
            text_parts.append(f"status: {store['location_status']}")
        if store.get('storePlatform'):
            text_parts.append(f"platform: {store['storePlatform']}")
        
        return ' '.join(text_parts)
    
    def create_store_vector(self, store: Dict[str, Any]) -> Dict[str, Any]:
        """Create a vector entry for a store"""
        text = self.prepare_store_text(store)
        embedding = self.generate_embedding(text)
        
        # Extract state information
        state_name = ''
        state_abbr = ''
        if store.get('state'):
            state_name = store['state'].get('post_title', '')
            state_abbr = store['state'].get('abbr', '')
        
        return {
            "id": f"store_{store.get('locationId', store.get('sweedId', 'unknown'))}",
            "values": embedding,
            "metadata": self._clean_metadata({
                "type": "store",
                "name": store.get('title', ''),
                "locationId": store.get('locationId', ''),
                "city": store.get('city', ''),
                "state": state_name,
                "state_code": state_abbr,
                "zip": store.get('zip', ''),
                "storePlatform":store.get("storePlatform", ""),
                "slug":store.get("slug",""),
            })
        }

# Global instance
embeddings_service = EmbeddingsService()