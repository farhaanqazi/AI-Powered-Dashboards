"""Semantic Contract Layer (Phases 1–8).

Phase 1 surface: the ingest gate and its typed result models.
"""
from src.contract.models import (
    CleaningManifest,
    IngestResult,
    FieldContract,
    DatasetContract,
    LLMOutputContract,
)
from src.contract.dq_report import (
    DataQualityReport,
    build_dq_report,
    evaluate_acceptance,
)
from src.contract.ingest_gate import run_ingest_gate
from src.contract.compiler import compile_contract, schema_fingerprint
from src.contract.cache import ContractCache, get_contract_cache
from src.contract.invariant_critic import (
    critique,
    apply_vetoes,
    CritiqueResult,
    InvariantVeto,
    InvariantFlag,
)

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
    "critique",
    "apply_vetoes",
    "CritiqueResult",
    "InvariantVeto",
    "InvariantFlag",
    "LLMOutputContract",
    "DataQualityReport",
    "build_dq_report",
    "evaluate_acceptance",
]
