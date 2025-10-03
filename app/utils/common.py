from typing import Any, Dict
from pydantic import BaseModel
from enum import Enum
def sanitize_metadata(data: Any, parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Recursively flattens and sanitizes metadata to ensure compatibility with Pinecone.
    
    Args:
        data: The metadata to be sanitized.
        parent_key: The base key to prepend to nested keys.
        sep: Separator for nested keys.
        
    Returns:
        A dictionary with sanitized metadata.
    """
    items = {}

    if isinstance(data, BaseModel):
        data = data.dict()  # Convert Pydantic model to dict

    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            items.update(sanitize_metadata(value, new_key, sep))
    elif isinstance(data, list):
        for i, value in enumerate(data):
            new_key = f"{parent_key}{sep}{i}"
            items.update(sanitize_metadata(value, new_key, sep))
    else:
        # Convert unsupported types to string or remove them
        if data is None:
            return {}  # Exclude None values
        elif isinstance(data, (dict, list)):
            return {parent_key: str(data)}  # Convert unsupported structures to string
        else:
            return {parent_key: data}

    return items
def collect_values(obj: any):
    value_parts: List[str] = []
    if obj is None:
        return value_parts
    if isinstance(obj, Enum):
        val = obj.value
        if is_nonempty(val):
            value_parts.append(str(val))
    elif isinstance(obj, BaseModel):
        collect_values(obj.dict(exclude_none=True))
    elif isinstance(obj, dict):
        for v in obj.values():
            collect_values(v)
    elif isinstance(obj, list):
        for v in obj:
            collect_values(v)
    else:
        # primitive
        if is_nonempty(obj):
            s = str(obj).strip()
            # filter out pure large IDs if desired
            if s.isdigit() and len(s) > 4:
                return
            value_parts.append(s)
    return value_parts
            
def safe_get(d: dict, *keys, default=None):
    """Drill down nested dicts; returns default if any missing or None."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur

def is_nonempty(v: any) -> bool:
    """Checks if value is meaningful (not None, not empty str/list/dict)."""
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip() != ""
    if isinstance(v, (list, dict)):
        return len(v) > 0
    return True
