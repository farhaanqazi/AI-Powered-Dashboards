from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from datetime import datetime

class DashboardResponse(BaseModel):
    """Guaranteed response structure - NEVER changes shape"""
    status: Literal["empty", "ready", "error"] = Field(
        ..., 
        description="System truth source for frontend"
    )
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict = Field(default_factory=dict)  # Pipeline version, source, etc.
    kpis: List[Dict] = Field(default_factory=list)
    charts: List[Dict] = Field(default_factory=list)
    eda: Dict = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)  # User-facing messages ONLY
    message: Optional[str] = Field(
        None,
        description="Contextual guidance (e.g., 'Connect ML pipeline to populate')"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {"status": "empty", "message": "No data yet. Check pipeline status.", "kpis": [], "charts": []},
                {"status": "ready", "kpis": [{"name": "Accuracy", "value": 0.95}], "charts": [...], "errors": []}
            ]
        }