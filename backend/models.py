# models.py
from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    file_contents: Optional[List[str]] = None
    question: Optional[str] = None
    index_user: Optional[bool] = False

class UserKBUpload(BaseModel):
    user_id: str
    file_contents: List[str]