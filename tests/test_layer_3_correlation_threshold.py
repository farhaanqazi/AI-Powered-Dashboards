"""Verify Layer 3 reads its threshold from config, not a literal."""
import numpy as np
import pandas as pd

from src.analysis.data_structures import EnrichedProfile
from src.analysis.layer_3_relational import run_relational_analysis
from src import config


def _make_profile(name: str, std: float) -> EnrichedProfile:
    return EnrichedProfile(
        name=name,
        dtype="float64",
        role="numeric",
        semantic_tags=[],
        null_count=0,
        unique_count=100,
        stats={"std": std, "count": 100},
        top_categories=[],
    )


def test_min_correlation_threshold_is_respected(monkeypatch):
    rng = np.random.default_rng(seed=42)
    base = rng.normal(0, 1, 200)
    weak_noise = rng.normal(0, 1, 200)
    # Weak correlation (~0.3) — should be excluded under default 0.5 threshold
    weak = 0.3 * base + 0.95 * weak_noise
    # Strong correlation (~0.9)
    strong = 0.9 * base + 0.1 * rng.normal(0, 1, 200)

    df = pd.DataFrame({"base": base, "weak": weak, "strong": strong})
    profiles = {
        "base": _make_profile("base", float(df["base"].std())),
        "weak": _make_profile("weak", float(df["weak"].std())),
        "strong": _make_profile("strong", float(df["strong"].std())),
    }

    monkeypatch.setattr(config, "MIN_CORRELATION", 0.5)
    insights = run_relational_analysis(df, profiles)
    pairs = [tuple(sorted(i.columns)) for i in insights]
    assert ("base", "strong") in pairs
    assert ("base", "weak") not in pairs

    # base-weak correlation lands at ~0.20 with this seed, so lower the
    # threshold to 0.1 to assert the weak pair is included once permitted.
    monkeypatch.setattr(config, "MIN_CORRELATION", 0.1)
    insights_low = run_relational_analysis(df, profiles)
    pairs_low = [tuple(sorted(i.columns)) for i in insights_low]
    assert ("base", "weak") in pairs_low
