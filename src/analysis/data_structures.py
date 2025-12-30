"""
Data Structures for the Analysis Pipeline

This module defines the structured data classes (e.g., using dataclasses or Pydantic)
that are passed between the layers of the analysis engine. This ensures a clear,
typed contract between each step of the analysis.
"""
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class SyntacticProfile:
    """Raw, objective facts about a single column."""
    name: str
    dtype: str
    null_count: int
    unique_count: int
    stats: Dict[str, Any] = field(default_factory=dict)
    top_categories: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class EnrichedProfile(SyntacticProfile):
    """A syntactic profile decorated with semantic meaning."""
    role: str = "unknown"
    confidence: float = 0.0
    semantic_tags: List[str] = field(default_factory=list)

@dataclass
class RelationalInsight:
    """Describes a single relationship found between columns."""
    type: str # e.g., 'correlation', 'time_series_pair'
    columns: List[str]
    details: Dict[str, Any]

@dataclass
class AnalysisOutput:
    """The final, consolidated output of the full analysis pipeline."""
    enriched_profiles: Dict[str, EnrichedProfile]
    relational_insights: List[RelationalInsight]
    kpis: List[Dict[str, Any]]
    charts: List[Dict[str, Any]]

@dataclass
class DashboardState:
    """Structured return type for the final dashboard state passed to the UI."""
    df: pd.DataFrame
    dataset_profile: Dict[str, Any]
    profile: List[Dict[str, Any]] # Deprecated, but kept for compatibility
    kpis: List[Dict[str, Any]]
    charts: List[Dict[str, Any]]
    primary_chart: Optional[Dict[str, Any]]
    category_charts: Dict[str, Any]
    all_charts: List[Dict[str, Any]]
    original_filename: Optional[str] = None
    errors: List[str] = field(default_factory=list)
