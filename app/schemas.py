from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    message: str

class Source(BaseModel):
    article_id: str
    article_title: str
    chunk_id: int
    snippet: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
