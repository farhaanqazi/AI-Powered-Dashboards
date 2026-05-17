"""Phase 4 — the Invariant Critic.

A deterministic sanity pass over Layer-2 profiles + the cleaned frame. It
catches the misclassifications and double-counting traps that role heuristics
miss, emitting two kinds of finding:

* **vetoes** — a confident role correction the pipeline should apply before
  the contract is locked (e.g. an all-unique integer column heuristically
  called a measure is really an identifier; an "id" column with fractional
  values is really a measure).
* **flags** — non-fatal warnings surfaced to the data-quality report / HITL
  reviewer (total-vs-components, share-sum-to-one, std≫mean).

All thresholds come from ``src.config`` — no literals here (architectural
invariant). Wiring into the pipeline is Phase 5; this module is self-contained.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src import config


class InvariantVeto(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    from_role: str
    to_role: str
    reason: str


class InvariantFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str  # total_vs_components | share_sum | std_gg_mean
    columns: List[str]
    detail: str
    severity: str = "warning"


class CritiqueResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vetoes: List[InvariantVeto] = Field(default_factory=list)
    flags: List[InvariantFlag] = Field(default_factory=list)


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").dropna()


def _is_integer_like(s: pd.Series) -> bool:
    return not s.empty and bool((s == s.round()).all())


def _critique_roles(
    df: pd.DataFrame, profiles: Dict[str, Any]
) -> List[InvariantVeto]:
    vetoes: List[InvariantVeto] = []
    n = len(df)
    for name, prof in profiles.items():
        if name not in df.columns:
            continue
        role = getattr(prof, "role", "unknown")
        s = _numeric(df, name)

        # 1. unique-numeric -> identifier veto.
        if role == "numeric" and n > 0 and not s.empty:
            unique_ratio = s.nunique() / n
            tags = set(getattr(prof, "semantic_tags", []) or [])
            if (
                unique_ratio >= config.CRITIC_ID_UNIQUE_RATIO
                and _is_integer_like(s)
                and "additive" not in tags
                and "monetary" not in tags
            ):
                vetoes.append(
                    InvariantVeto(
                        column=name,
                        from_role=role,
                        to_role="identifier",
                        reason=(
                            f"Almost every value is a different whole number "
                            f"({unique_ratio * 100:.0f}% unique) — this looks "
                            f"like an ID, not a measurement."
                        ),
                    )
                )
                continue

        # 2. fractional-ID veto: identifiers are never fractional numbers.
        if role == "identifier" and not s.empty and len(s) == len(df[name].dropna()):
            if not _is_integer_like(s):
                vetoes.append(
                    InvariantVeto(
                        column=name,
                        from_role=role,
                        to_role="numeric",
                        reason="Has decimal values, so it isn’t an ID.",
                    )
                )
    return vetoes


def _critique_totals(
    df: pd.DataFrame, profiles: Dict[str, Any]
) -> List[InvariantFlag]:
    flags: List[InvariantFlag] = []
    if len(df) < config.CRITIC_MIN_ROWS:
        return flags

    numeric_cols = [
        c
        for c, p in profiles.items()
        if c in df.columns and getattr(p, "role", "") == "numeric"
    ]
    num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # total-vs-components: a column equal (row-wise) to the sum of >=2 others.
    for total in numeric_cols:
        others = [c for c in numeric_cols if c != total]
        if len(others) < 2:
            continue
        comp_sum = num_df[others].sum(axis=1)
        total_col = num_df[total]
        mask = total_col.abs() > 0
        if mask.sum() < config.CRITIC_MIN_ROWS:
            continue
        rel_err = ((comp_sum - total_col).abs() / total_col.abs())[mask]
        if rel_err.mean() <= config.CRITIC_TOTAL_TOLERANCE:
            flags.append(
                InvariantFlag(
                    type="total_vs_components",
                    columns=[total, *others],
                    detail=(
                        f"'{total}' equals the sum of {others} — using both "
                        f"the total and its parts would count the same "
                        f"numbers twice."
                    ),
                )
            )
            break  # one such relation is enough to warn

    # share-sum: >=2 ratio-ish columns summing to ~1.0 (or ~100) per row.
    share_cols = [
        c
        for c in numeric_cols
        if "rate" in (getattr(profiles[c], "semantic_tags", []) or [])
        or "ratio" in c.lower()
        or "share" in c.lower()
        or "pct" in c.lower()
    ]
    if len(share_cols) >= 2:
        row_sum = num_df[share_cols].sum(axis=1)
        for target in (1.0, 100.0):
            close = (row_sum - target).abs() <= config.CRITIC_SHARE_SUM_TOLERANCE * target
            if close.mean() >= 0.9:
                flags.append(
                    InvariantFlag(
                        type="share_sum",
                        columns=share_cols,
                        detail=(
                            f"{share_cols} add up to about {int(target)} per "
                            f"row — these look like percentages; don’t add "
                            f"them across rows."
                        ),
                    )
                )
                break
    return flags


def _critique_dispersion(
    df: pd.DataFrame, profiles: Dict[str, Any]
) -> List[InvariantFlag]:
    flags: List[InvariantFlag] = []
    for name, prof in profiles.items():
        if getattr(prof, "role", "") != "numeric" or name not in df.columns:
            continue
        stats = getattr(prof, "stats", {}) or {}
        mean = stats.get("mean")
        std = stats.get("std")
        if mean is None or std is None or mean == 0:
            continue
        if abs(std / mean) >= config.CRITIC_STD_MEAN_RATIO:
            flags.append(
                InvariantFlag(
                    type="std_gg_mean",
                    columns=[name],
                    detail=(
                        f"'{name}' varies a lot (spread is "
                        f"{abs(std / mean):.1f}× the average) — the average "
                        f"isn’t a good summary of a typical value."
                    ),
                    severity="info",
                )
            )
    return flags


def critique(df: pd.DataFrame, profiles: Dict[str, Any]) -> CritiqueResult:
    """Run all invariant checks. Pure — never mutates inputs."""
    if df is None or df.empty or not profiles:
        return CritiqueResult()
    return CritiqueResult(
        vetoes=_critique_roles(df, profiles),
        flags=_critique_totals(df, profiles) + _critique_dispersion(df, profiles),
    )


def apply_vetoes(
    profiles: Dict[str, Any], result: CritiqueResult
) -> Dict[str, Any]:
    """Return a shallow-copied profile map with veto role corrections applied.

    EnrichedProfile is a dataclass — copy + reassign ``role`` so the original
    (and any frozen contract derived from it) is untouched.
    """
    import copy

    if not result.vetoes:
        return profiles
    out = dict(profiles)
    for v in result.vetoes:
        if v.column in out:
            p = copy.copy(out[v.column])
            try:
                p.role = v.to_role
            except Exception:
                continue
            out[v.column] = p
    return out
