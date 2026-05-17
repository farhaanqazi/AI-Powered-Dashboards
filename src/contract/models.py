"""Phase 1 contract models — the typed result of the ingest gate.

These are deliberately lightweight (Pydantic v2). The frozen ``FieldContract``
/ ``DatasetContract`` of Phase 2 build on top of the cleaned frame and the
sensitivity verdict produced here.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class CleaningManifest(BaseModel):
    """An auditable record of every mutation the ingest gate applied.

    Nothing is cleaned silently: each transformation is recorded here so the
    downstream contract (and the HITL reviewer in Phase 7) can see exactly
    what changed between the uploaded bytes and the profiled frame.
    """

    model_config = ConfigDict(extra="forbid")

    original_shape: Tuple[int, int] = (0, 0)
    cleaned_shape: Tuple[int, int] = (0, 0)
    # column -> human-readable note about the numeric coercion applied.
    coerced_numeric: Dict[str, str] = Field(default_factory=dict)
    # column -> count of cells replaced with NA because they were sentinels.
    sentinels_nulled: Dict[str, int] = Field(default_factory=dict)
    # number of rows dropped because they were (near-)entirely null.
    dropped_null_rows: int = 0
    notes: List[str] = Field(default_factory=list)


class IngestResult(BaseModel):
    """Outcome of :func:`src.contract.ingest_gate.run_ingest_gate`.

    ``pii_blocked`` is the fail-closed egress switch: when True the LLM layer
    must never receive rows from this dataset, and (per the architectural
    invariant) no human approval re-opens it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    ok: bool = True
    rejected: bool = False
    reject_reason: Optional[str] = None

    df: Optional[pd.DataFrame] = None
    manifest: CleaningManifest = Field(default_factory=CleaningManifest)

    # "public" | "sensitive" — fail-closed to "sensitive" when undeterminable.
    sensitivity: str = "public"
    pii_blocked: bool = False
    # column -> sorted list of detected PII entity types.
    pii_columns: Dict[str, List[str]] = Field(default_factory=dict)
    pii_scan_engine: str = "none"  # "presidio" | "regex" | "unavailable"

    warnings: List[str] = Field(default_factory=list)


class FieldContract(BaseModel):
    """Immutable per-column contract produced by the compiler.

    Frozen: once compiled, a field's role/aggregation rules cannot mutate. The
    only mutation path is a HITL override that produces a *new* locked
    ``DatasetContract`` version (Phase 7).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    dtype: str
    role: str  # identifier|numeric|categorical|datetime|boolean|text|year|ratio
    # Finer semantic class: monetary|ratio|year|count|measure|dimension|...
    domain: str = "generic"
    confidence: float = 0.0
    alternatives: Tuple[Dict[str, float], ...] = ()
    # 'additive' (summable across grain), 'rate' (averaged), or 'none'.
    aggregation: str = "none"
    is_identifier: bool = False
    is_year: bool = False
    is_ratio: bool = False
    sensitivity: str = "public"  # public|sensitive
    pii_entities: Tuple[str, ...] = ()
    allowed_aggregations: Tuple[str, ...] = ()
    allowed_charts: Tuple[str, ...] = ()


class DatasetContract(BaseModel):
    """Immutable dataset-level contract. Serializes into the existing
    ``DashboardRecord`` (no new storage backend)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_fingerprint: str
    version: int = 1
    locked: bool = False
    n_rows: int = 0
    n_cols: int = 0
    # Minimal column set that uniquely identifies a row, or () if none found.
    grain: Tuple[str, ...] = ()
    has_aggregate_rows: bool = False
    aggregate_row_count: int = 0
    sensitivity: str = "public"
    pii_blocked: bool = False
    fields: Dict[str, FieldContract] = Field(default_factory=dict)

    def with_lock(self, *, bump_version: bool = True) -> "DatasetContract":
        """Return a new locked contract (frozen models never mutate in place)."""
        return self.model_copy(
            update={
                "locked": True,
                "version": self.version + 1 if bump_version else self.version,
            }
        )
