"""Phase 10 S10.3 — multi-format ingestion behind the parser interface.

Parquet / Excel / JSON / NDJSON join CSV as first-class inputs. Reading goes
through the S10.2 :mod:`src.data.engine` seam; this module owns format
detection and the post-read normalization (column-name cleanup + row cap)
that the legacy CSV path already applied, so every format produces an
identically-shaped frame for the unchanged pipeline.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from src import config
from src.data.engine import get_engine

_EXT_TO_FMT = {
    "csv": "csv",
    "parquet": "parquet",
    "pq": "parquet",
    "xlsx": "xlsx",
    "xls": "xls",
    "json": "json",
    "ndjson": "ndjson",
    "jsonl": "jsonl",
}


def detect_format(filename: Optional[str]) -> Optional[str]:
    """Map a filename to a canonical format token, or None if unsupported."""
    if not filename or "." not in filename:
        return None
    ext = filename.rsplit(".", 1)[1].strip().lower()
    fmt = _EXT_TO_FMT.get(ext)
    if fmt is None:
        return None
    if ext not in config.INGEST_ALLOWED_FORMATS and fmt not in config.INGEST_ALLOWED_FORMATS:
        return None
    return fmt


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Identical to the legacy CSV cleanup so downstream behaviour is stable."""
    df.columns = [
        str(c).strip().replace(" ", "_").replace("-", "_") for c in df.columns
    ]
    out, n = [], 0
    for c in df.columns:
        if c.startswith("Unnamed:"):
            out.append(f"unnamed_col_{n}")
            n += 1
        else:
            out.append(c)
    df.columns = out
    return df


def read_dataframe(
    raw: bytes,
    filename: str,
    *,
    max_rows: Optional[int] = None,
    encoding: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Parse ``raw`` for ``filename`` via the engine seam; normalize + cap.

    Raises ``ValueError`` for an unsupported/empty format (the caller maps
    that to a clean LoadResult error).
    """
    fmt = detect_format(filename)
    if fmt is None:
        raise ValueError(f"Unsupported file type: {filename!r}")
    warnings: List[str] = []
    df = get_engine().read(raw, fmt, encoding=encoding)
    if df is None or df.shape[1] == 0:
        raise ValueError("The file contained no columns.")
    df = _normalize_columns(df)

    cap = max_rows if max_rows is not None else config.MAX_ROWS
    if len(df) > cap:
        df = df.sample(n=cap, random_state=42).reset_index(drop=True)
        warnings.append(
            f"Dataset had more than {cap} rows; sampled down for processing."
        )
    return df, warnings
