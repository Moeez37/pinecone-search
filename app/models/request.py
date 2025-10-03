from pydantic import BaseModel,Field
from typing import List, Dict, Any, Optional
from app.utils.constants import OBJECT_TYPE,STORE_TYPE

class MetaData(BaseModel):
    type: STORE_TYPE
    class Config:
        extra = "allow"
class IngestRequest(BaseModel):
    location_id: Optional[str] = None
    id: Optional[str] = None
    type: OBJECT_TYPE
    metadata: MetaData
        
class BatchIngestRequest(BaseModel):
    location_id:Optional[str] = None
    id: Optional[str] = None
    type: OBJECT_TYPE
    metadata: List[MetaData]

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    location_id: Optional[str] = None
    namespace: Optional[OBJECT_TYPE] = OBJECT_TYPE.ALL

class StatusResponse(BaseModel):
    status: str
    message: str

class SearchMatch(BaseModel):
    id: Optional[str] = None
    metadata: Optional[MetaData] = None
    score: Optional[float] = None
    values: Optional[List[float]] = Field(default_factory=list)
    
class SearchResponse(BaseModel):
    results: List[SearchMatch]
    query_rewritten: Optional[str] = None
    total_results: int
