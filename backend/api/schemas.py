from pydantic import BaseModel
from typing import Dict, List, Any, Optional


class AnalyzeResponse(BaseModel):
    job_id: str
    status: str
    message: str


class ResultsResponse(BaseModel):
    job_id: str
    status: str
    summary: Optional[Dict[str, Any]] = None
    timeline_text: Optional[str] = None
    error: Optional[str] = None