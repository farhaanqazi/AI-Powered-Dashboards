"""Pydantic response models for every HTTP endpoint in `main.py`.

These describe the wire contract. Frontend code (`frontend/src/services/api.js`)
relies on these field names exactly. Add new fields as `Optional` to keep
backward compatibility; never rename existing fields without a frontend change.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


_PERMISSIVE = ConfigDict(extra="allow")


class DashboardPayload(BaseModel):
    """The shared payload nested under `data` in upload/load responses and
    surfaced at the top level by GET /api/dashboard."""

    model_config = _PERMISSIVE

    dataset_profile: Dict[str, Any] = Field(default_factory=dict)
    kpis: List[Dict[str, Any]] = Field(default_factory=list)
    charts: List[Dict[str, Any]] = Field(default_factory=list)
    primary_chart: Optional[Dict[str, Any]] = None
    category_charts: Dict[str, Any] = Field(default_factory=dict)
    all_charts: List[Dict[str, Any]] = Field(default_factory=list)
    original_filename: str = ""
    errors: List[Any] = Field(default_factory=list)
    warnings: List[Any] = Field(default_factory=list)
    critical_totals: Dict[str, Any] = Field(default_factory=dict)
    critical_full_dataset_aggregates: Dict[str, Any] = Field(default_factory=dict)
    eda_summary: Dict[str, Any] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    model_config = _PERMISSIVE
    status: str
    trace_id: str
    data: DashboardPayload


class LoadExternalResponse(BaseModel):
    model_config = _PERMISSIVE
    status: str
    trace_id: str
    data: DashboardPayload


class ValidateExternalResponse(BaseModel):
    model_config = _PERMISSIVE
    ok: bool


class DashboardResponse(BaseModel):
    model_config = _PERMISSIVE

    status: str
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    kpis: List[Dict[str, Any]] = Field(default_factory=list)
    charts: List[Dict[str, Any]] = Field(default_factory=list)
    eda: Dict[str, Any] = Field(default_factory=dict)
    errors: List[Any] = Field(default_factory=list)
    warnings: List[Any] = Field(default_factory=list)
    message: Optional[str] = None
    dataset_profile: Dict[str, Any] = Field(default_factory=dict)
    primary_chart: Optional[Dict[str, Any]] = None
    category_charts: Dict[str, Any] = Field(default_factory=dict)
    all_charts: List[Dict[str, Any]] = Field(default_factory=list)
    original_filename: str = ""
    critical_totals: Dict[str, Any] = Field(default_factory=dict)
    critical_full_dataset_aggregates: Dict[str, Any] = Field(default_factory=dict)
    eda_summary: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    model_config = _PERMISSIVE
    message: str
    error_type: Optional[str] = None
    error_detail: Optional[str] = None
    errors: Optional[List[Any]] = None
