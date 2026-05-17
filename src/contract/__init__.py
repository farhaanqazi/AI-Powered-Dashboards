"""Semantic Contract Layer (Phases 1–8).

Phase 1 surface: the ingest gate and its typed result models.
"""
from src.contract.models import (
    CleaningManifest,
    IngestResult,
    FieldContract,
    DatasetContract,
)
from src.contract.ingest_gate import run_ingest_gate
from src.contract.compiler import compile_contract, schema_fingerprint
from src.contract.cache import ContractCache, get_contract_cache

__all__ = [
    "run_ingest_gate",
    "IngestResult",
    "CleaningManifest",
    "FieldContract",
    "DatasetContract",
    "compile_contract",
    "schema_fingerprint",
    "ContractCache",
    "get_contract_cache",
]
