"""Phase 8 (S8.1) — property-based normalization & rejection tests.

Hypothesis drives the ingest gate with a wide space of inputs to assert the
*invariants* of normalization rather than hand-picked examples:

* purity — the caller's frame is never mutated;
* numeric coercion — currency/grouping-formatted numbers round-trip to floats;
* sentinel nulling — every configured sentinel token becomes ``pd.NA`` and is
  accounted for in the manifest;
* null-row rejection — an all-null dataset is rejected fail-closed.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src import config
from src.contract import run_ingest_gate

_SETTINGS = settings(
    max_examples=25,
    deadline=None,  # Windows CI timing is noisy; assertions are deterministic.
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

# Finite, JSON-safe money values; avoid NaN/inf and absurd magnitudes.
_money = st.floats(
    min_value=-1_000_000, max_value=1_000_000,
    allow_nan=False, allow_infinity=False,
).map(lambda v: round(v, 2))


@given(values=st.lists(_money, min_size=3, max_size=40))
@_SETTINGS
def test_currency_formatted_column_round_trips_to_numeric(values):
    formatted = [f"${v:,.2f}" for v in values]
    region = [("N", "S")[i % 2] for i in range(len(values))]
    df = pd.DataFrame({"region": region, "amount": formatted})
    before = df.copy(deep=True)

    res = run_ingest_gate(df)

    # Purity: the gate copies; the caller's frame is untouched.
    pd.testing.assert_frame_equal(df, before)

    assert res.ok and not res.rejected
    assert pd.api.types.is_numeric_dtype(res.df["amount"])
    assert "amount" in res.manifest.coerced_numeric
    for got, want in zip(res.df["amount"].tolist(), values):
        assert math.isclose(got, want, abs_tol=0.01)


@given(
    clean=st.lists(_money, min_size=2, max_size=10),
    sentinel=st.sampled_from(config.INGEST_SENTINELS),
)
@_SETTINGS
def test_every_sentinel_token_becomes_na_and_is_counted(clean, sentinel):
    col = [str(v) for v in clean] + [sentinel, sentinel.upper()]
    df = pd.DataFrame({"keep": list(range(len(col))), "val": col})

    res = run_ingest_gate(df)

    assert res.ok
    # The two sentinel cells (any case) were nulled and recorded.
    assert res.manifest.sentinels_nulled.get("val", 0) >= 2
    tail = res.df["val"].tail(2)
    assert tail.isna().all()


@given(
    n_rows=st.integers(min_value=1, max_value=25),
    n_cols=st.integers(min_value=1, max_value=6),
)
@_SETTINGS
def test_all_null_dataset_is_rejected_fail_closed(n_rows, n_cols):
    df = pd.DataFrame(
        {f"c{i}": [None] * n_rows for i in range(n_cols)}
    )
    res = run_ingest_gate(df)

    assert res.rejected and not res.ok
    assert res.reject_reason
    # Fail-closed: an unanalyzable dataset is never labelled "public".
    if config.SENSITIVITY_FAIL_CLOSED:
        assert res.sensitivity == "sensitive"


@given(data=st.data())
@_SETTINGS
def test_partially_null_rows_survive_when_any_value_present(data):
    n = data.draw(st.integers(min_value=3, max_value=20))
    keep_vals = data.draw(
        st.lists(_money, min_size=n, max_size=n)
    )
    df = pd.DataFrame({"id": list(range(n)), "amt": keep_vals})
    res = run_ingest_gate(df)
    assert res.ok and not res.rejected
    # No fully-null rows were introduced, so none are dropped.
    assert res.manifest.dropped_null_rows == 0
    assert res.df.shape[0] == n


def test_no_columns_is_rejected():
    res = run_ingest_gate(pd.DataFrame())
    assert res.rejected and not res.ok
    assert "no columns" in (res.reject_reason or "").lower()
