"""Phase 14 S14.1 (Gap A) — the cleaned frame survives a container restart.

The transient tier is wiped on restart. The durable Parquet tier must let
get() rehydrate so Ask/Interact stop returning "working data has expired".
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.contract.df_cache import DataFrameCache


@pytest.fixture
def frame():
    return pd.DataFrame({"region": ["N", "S", "E"] * 5,
                         "revenue": [100, 250, 320] * 5})


def test_durable_rehydrate_after_transient_wipe(tmp_path, frame, monkeypatch):
    monkeypatch.setattr(config, "CLEANED_DF_DURABLE_ENABLED", True)
    monkeypatch.setattr(config, "CLEANED_DF_CACHE_ENABLED", True)

    writer = DataFrameCache(client=None, durable_dir=str(tmp_path))
    writer.put("fp-abc", frame)

    # Simulate a container restart: brand-new instance, empty in-process mem,
    # same durable dir (the filesystem spool persisted).
    reborn = DataFrameCache(client=None, durable_dir=str(tmp_path))
    got = reborn.get("fp-abc")
    assert got is not None
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), frame.reset_index(drop=True)
    )


def test_durable_disabled_means_no_survival(tmp_path, frame, monkeypatch):
    monkeypatch.setattr(config, "CLEANED_DF_DURABLE_ENABLED", False)
    monkeypatch.setattr(config, "CLEANED_DF_CACHE_ENABLED", True)

    writer = DataFrameCache(client=None, durable_dir=str(tmp_path))
    writer.put("fp-x", frame)
    reborn = DataFrameCache(client=None, durable_dir=str(tmp_path))
    assert reborn.get("fp-x") is None  # graceful: caller degrades as before


def test_durable_ttl_expiry(tmp_path, frame, monkeypatch):
    monkeypatch.setattr(config, "CLEANED_DF_DURABLE_ENABLED", True)
    monkeypatch.setattr(config, "CLEANED_DF_CACHE_ENABLED", True)
    monkeypatch.setattr(config, "CLEANED_DF_DURABLE_TTL_SECONDS", -1)

    writer = DataFrameCache(client=None, durable_dir=str(tmp_path))
    writer.put("fp-old", frame)
    reborn = DataFrameCache(client=None, durable_dir=str(tmp_path))
    assert reborn.get("fp-old") is None  # past TTL → evicted
