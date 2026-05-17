"""Phase 4 — invariant critic."""
import pandas as pd

from src.analysis.data_structures import EnrichedProfile
from src.contract.invariant_critic import apply_vetoes, critique


def _p(name, role, tags=None, stats=None):
    return EnrichedProfile(
        name=name, dtype="float64", role=role, semantic_tags=tags or [],
        null_count=0, unique_count=0, stats=stats or {}, top_categories=[],
    )


def test_unique_integer_numeric_vetoed_to_identifier():
    df = pd.DataFrame({"txn": list(range(1000, 1100)), "amt": [1.5] * 100})
    profiles = {"txn": _p("txn", "numeric"), "amt": _p("amt", "numeric", ["additive"])}
    res = critique(df, profiles)
    v = [v for v in res.vetoes if v.column == "txn"]
    assert v and v[0].to_role == "identifier"
    # additive measure is NOT vetoed even though here it is constant
    assert not any(v.column == "amt" for v in res.vetoes)


def test_apply_vetoes_does_not_mutate_original():
    df = pd.DataFrame({"txn": list(range(1000, 1100))})
    profiles = {"txn": _p("txn", "numeric")}
    res = critique(df, profiles)
    corrected = apply_vetoes(profiles, res)
    assert corrected["txn"].role == "identifier"
    assert profiles["txn"].role == "numeric"  # untouched


def test_fractional_identifier_vetoed_to_numeric():
    df = pd.DataFrame({"score_id": [1.5, 2.7, 3.9, 4.1, 5.5]})
    profiles = {"score_id": _p("score_id", "identifier")}
    res = critique(df, profiles)
    assert res.vetoes and res.vetoes[0].to_role == "numeric"


def test_total_vs_components_flagged():
    n = 20
    a = list(range(1, n + 1))
    b = [x * 2 for x in a]
    df = pd.DataFrame({"a": a, "b": b, "total": [x + y for x, y in zip(a, b)]})
    profiles = {c: _p(c, "numeric") for c in df.columns}
    res = critique(df, profiles)
    assert any(f.type == "total_vs_components" for f in res.flags)


def test_share_sum_flagged():
    n = 15
    s1 = [0.3] * n
    s2 = [0.7] * n
    df = pd.DataFrame({"share_a": s1, "share_b": s2})
    profiles = {
        "share_a": _p("share_a", "numeric", ["rate"]),
        "share_b": _p("share_b", "numeric", ["rate"]),
    }
    res = critique(df, profiles)
    assert any(f.type == "share_sum" for f in res.flags)


def test_std_much_greater_than_mean_flagged():
    profiles = {
        "spiky": _p("spiky", "numeric", stats={"mean": 1.0, "std": 50.0}),
    }
    df = pd.DataFrame({"spiky": [0, 0, 0, 250]})
    res = critique(df, profiles)
    assert any(f.type == "std_gg_mean" for f in res.flags)


def test_empty_inputs_safe():
    assert critique(pd.DataFrame(), {}).vetoes == []
    assert critique(None, {"x": _p("x", "numeric")}).flags == []
