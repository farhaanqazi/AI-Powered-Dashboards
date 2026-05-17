"""Phase 7 — HITL registry override (pure logic + PATCH route)."""
import pandas as pd

from src.analysis.layer_1_profiler import run_syntactic_profiling
from src.analysis.layer_2_classifier import run_semantic_classification
from src.contract import compile_contract, run_ingest_gate
from src.contract.registry_patch import apply_registry_overrides


def _payload():
    df = pd.DataFrame({
        "region": ["N", "S", "E", "W"] * 8,
        "spend": [10.0, 22.0, 31.0, 9.0] * 8,
        "ticket": list(range(32)),
    })
    res = run_ingest_gate(df)
    profs = run_semantic_classification(run_syntactic_profiling(res.df), res.df)
    contract = compile_contract(res.df, profs, res)
    return {
        "dataset_profile": {
            "contract": contract.model_dump(mode="json"),
            "columns": [{"name": n, "role": p.role} for n, p in profs.items()],
            "data_quality": {"report": {"status": "review"}},
        },
        "all_charts": [
            {"x_field": "spend", "chart_type": "histogram"},
            {"x_field": "ticket", "chart_type": "histogram"},
        ],
        "kpis": [
            {"label": "Avg spend", "provenance": {"source": "column:spend"}},
            {"label": "Ticket", "provenance": {"source": "column:ticket"}},
        ],
    }


def test_override_locks_contract_and_bumps_version():
    p = apply_registry_overrides(_payload(), [{"name": "spend", "role": "ratio"}])
    c = p["dataset_profile"]["contract"]
    assert c["locked"] is True
    assert c["version"] == 2
    assert c["fields"]["spend"]["role"] == "ratio"
    assert "sum" not in c["fields"]["spend"]["allowed_aggregations"]


def test_override_to_identifier_drops_its_charts_and_kpis():
    p = apply_registry_overrides(_payload(), [{"name": "ticket", "role": "identifier"}])
    assert all(ch["x_field"] != "ticket" for ch in p["all_charts"])
    assert all(
        (k.get("provenance") or {}).get("source") != "column:ticket"
        for k in p["kpis"]
    )
    assert p["dataset_profile"]["data_quality"]["report"]["status"] == "ok"
    assert p["dataset_profile"]["data_quality"]["report"]["human_reviewed"] is True


def test_human_override_cannot_clear_pii_block():
    pay = _payload()
    c = pay["dataset_profile"]["contract"]
    c["pii_blocked"] = True
    c["fields"]["region"]["sensitivity"] = "sensitive"
    out = apply_registry_overrides(pay, [{"name": "region", "sensitivity": "public"}])
    oc = out["dataset_profile"]["contract"]
    assert oc["pii_blocked"] is True  # fail-closed: review never unblocks
    assert out["dataset_profile"]["data_quality"]["report"]["status"] == "blocked"


def test_missing_contract_raises():
    import pytest
    with pytest.raises(ValueError):
        apply_registry_overrides({"dataset_profile": {}}, [{"name": "x", "role": "text"}])


# ---- PATCH route ----

def test_patch_registry_requires_confirm(client, upload_files):
    up = client.post("/api/upload", files=upload_files)
    assert up.status_code == 200
    tid = up.json()["trace_id"]
    r = client.patch(f"/api/dashboard/{tid}/registry",
                      json={"overrides": [], "confirm": False})
    assert r.status_code == 400


def test_patch_registry_locks_and_persists(client, upload_files):
    up = client.post("/api/upload", files=upload_files)
    tid = up.json()["trace_id"]
    r = client.patch(
        f"/api/dashboard/{tid}/registry",
        json={"overrides": [{"name": "amount", "role": "ratio"}], "confirm": True},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["locked"] is True
    assert body["contract_version"] >= 2
    # Persisted: GET reflects the locked contract.
    g = client.get("/api/dashboard").json()
    assert g["dataset_profile"]["contract"]["locked"] is True


def test_patch_registry_404_without_dashboard(client):
    r = client.patch("/api/dashboard/none/registry",
                      json={"overrides": [], "confirm": True})
    assert r.status_code == 404


# ---- Finding ② fix: cached df → real re-render on override ----

def test_override_truly_rerenders_when_df_cached(client, upload_files):
    """An override of a real column re-runs L3→render (recomputed=True) and
    the dashboard charts reflect the corrected role."""
    up = client.post("/api/upload", files=upload_files)
    assert up.status_code == 200
    tid = up.json()["trace_id"]
    before = up.json()["data"]["all_charts"]

    r = client.patch(
        f"/api/dashboard/{tid}/registry",
        json={"overrides": [{"name": "amount", "role": "identifier"}],
              "confirm": True},
    )
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    report = data["dataset_profile"]["data_quality"]["report"]
    # df was cached by the pipeline → genuine recompute, not contract-only.
    assert report["recomputed"] is True
    # 'amount' is now an identifier → it must not drive any chart.
    for ch in data["all_charts"]:
        assert ch.get("x_field") != "amount" and ch.get("y_field") != "amount"
    assert isinstance(before, list)


def test_rebuild_dashboard_uses_contract_roles_not_reclassified():
    import pandas as pd
    from src.contract import compile_contract, run_ingest_gate
    from src.analysis.layer_1_profiler import run_syntactic_profiling
    from src.analysis.layer_2_classifier import run_semantic_classification
    from src.contract.rebuild import rebuild_dashboard

    df = pd.DataFrame({
        "region": ["N", "S", "E", "W"] * 8,
        "spend": [10.0, 22.0, 31.0, 9.0] * 8,
    })
    res = run_ingest_gate(df)
    profs = run_semantic_classification(run_syntactic_profiling(res.df), res.df)
    contract = compile_contract(res.df, profs, res)
    # Force 'spend' to identifier in the contract and rebuild.
    fc = contract.fields["spend"].model_copy(update={"role": "identifier",
                                                     "is_identifier": True})
    locked = contract.model_copy(
        update={"fields": {**contract.fields, "spend": fc}, "locked": True}
    )
    out = rebuild_dashboard(res.df, locked)
    roles = {c["name"]: c["role"] for c in out["columns"]}
    assert roles["spend"] == "identifier"  # contract role honored, not re-classified


def test_df_cache_disabled_falls_back(monkeypatch):
    import pandas as pd
    from src import config
    from src.contract.df_cache import get_df_cache, reset_df_cache_for_tests

    monkeypatch.setattr(config, "CLEANED_DF_CACHE_ENABLED", False)
    reset_df_cache_for_tests()
    c = get_df_cache()
    c.put("fp", pd.DataFrame({"a": [1]}))
    assert c.get("fp") is None  # disabled → no storage, graceful fallback
    reset_df_cache_for_tests()
