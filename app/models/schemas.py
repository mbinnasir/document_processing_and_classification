from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any, List
from uuid import UUID

class DocumentBase(BaseModel):
    document_name: str

class DocumentCreate(DocumentBase):
    content: str
    vector_embeddings: Optional[List[float]] = None

class DocumentResponse(DocumentBase):
    id: UUID
    content: Optional[str] = None
    processed_output: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)

class ChatQuery(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: Any

class ProcessingStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    current_file: str
