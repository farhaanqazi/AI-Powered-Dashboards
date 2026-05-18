"""Phase 10 S10.2 — the DataFrame engine seam.

The entire analysis pipeline consumes :class:`pandas.DataFrame`. This seam
only abstracts *how raw bytes become that frame*. ``pandas`` is the default;
``polars`` / ``duckdb`` read larger-than-memory inputs efficiently and then
materialize to pandas so nothing downstream changes. A missing backend library
or an unknown engine name degrades gracefully to pandas — never an exception.

Every reader takes an in-memory ``bytes`` payload (the upload is already
size-capped upstream) plus a format string, and returns a pandas DataFrame.
"""
from __future__ import annotations

import abc
import io
from typing import Optional

import pandas as pd

from src import config

try:
    from src.logger import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


class DataFrameEngine(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def read(self, raw: bytes, fmt: str, *, encoding: Optional[str] = None) -> pd.DataFrame:
        """Parse ``raw`` (a ``fmt`` payload) into a pandas DataFrame."""


class PandasEngine(DataFrameEngine):
    name = "pandas"

    def read(self, raw, fmt, *, encoding=None):
        bio = io.BytesIO(raw)
        if fmt == "csv":
            return pd.read_csv(bio, encoding=encoding or "utf-8",
                               engine="python", on_bad_lines="skip")
        if fmt == "parquet":
            return pd.read_parquet(bio)
        if fmt in ("xlsx", "xls"):
            return pd.read_excel(bio)
        if fmt == "json":
            return pd.read_json(bio)
        if fmt in ("ndjson", "jsonl"):
            return pd.read_json(bio, lines=True)
        raise ValueError(f"Unsupported format '{fmt}'")


class PolarsEngine(DataFrameEngine):
    """Polars reader (memory-efficient), materialized to pandas."""

    name = "polars"

    def read(self, raw, fmt, *, encoding=None):
        try:
            import polars as pl
        except Exception:
            logger.info("polars unavailable; falling back to pandas engine")
            return PandasEngine().read(raw, fmt, encoding=encoding)
        bio = io.BytesIO(raw)
        try:
            if fmt == "csv":
                return pl.read_csv(bio, ignore_errors=True).to_pandas()
            if fmt == "parquet":
                return pl.read_parquet(bio).to_pandas()
            if fmt in ("ndjson", "jsonl"):
                return pl.read_ndjson(bio).to_pandas()
            if fmt == "json":
                return pl.read_json(bio).to_pandas()
        except Exception as exc:
            logger.warning("polars read failed (%s); pandas fallback", exc)
        # Excel + anything polars choked on → pandas.
        return PandasEngine().read(raw, fmt, encoding=encoding)


class DuckDBEngine(DataFrameEngine):
    """DuckDB reader for csv/parquet/json; materialized to pandas."""

    name = "duckdb"

    def read(self, raw, fmt, *, encoding=None):
        try:
            import duckdb
        except Exception:
            logger.info("duckdb unavailable; falling back to pandas engine")
            return PandasEngine().read(raw, fmt, encoding=encoding)
        try:
            con = duckdb.connect()
            con.register_filesystem  # noqa: B018 - probe attr existence
            if fmt == "csv":
                rel = con.read_csv(io.BytesIO(raw))
            elif fmt == "parquet":
                import tempfile, os

                with tempfile.NamedTemporaryFile(
                    suffix=".parquet", delete=False
                ) as t:
                    t.write(raw)
                    tmp = t.name
                try:
                    return con.execute(
                        f"SELECT * FROM read_parquet('{tmp}')"
                    ).df()
                finally:
                    os.unlink(tmp)
            elif fmt in ("json", "ndjson", "jsonl"):
                return PandasEngine().read(raw, fmt, encoding=encoding)
            else:
                return PandasEngine().read(raw, fmt, encoding=encoding)
            return rel.df()
        except Exception as exc:
            logger.warning("duckdb read failed (%s); pandas fallback", exc)
            return PandasEngine().read(raw, fmt, encoding=encoding)


_ENGINES = {
    "pandas": PandasEngine,
    "polars": PolarsEngine,
    "duckdb": DuckDBEngine,
}

_singleton: Optional[DataFrameEngine] = None


def get_engine() -> DataFrameEngine:
    """Config-selected engine singleton; unknown name ⇒ pandas."""
    global _singleton
    if _singleton is None:
        cls = _ENGINES.get(config.DATAFRAME_ENGINE, PandasEngine)
        _singleton = cls()
        if config.DATAFRAME_ENGINE not in _ENGINES:
            logger.info(
                "Unknown DATAFRAME_ENGINE '%s'; using pandas",
                config.DATAFRAME_ENGINE,
            )
    return _singleton


def reset_engine_for_tests() -> None:
    global _singleton
    _singleton = None
