"""Regression: the generator's early-exit paths (empty / rejected / schema-
review) yield a valid DashboardState but historically did NOT assign the local
`state` var, so the `finally` block mislabeled the run CRITICAL_FAILURE with
"Pipeline (generator) failed to return a state object" — a false alarm seen in
production telemetry even though a dashboard was produced and persisted.
"""
import pandas as pd

import src.core.pipeline as pipeline


def test_generator_empty_df_is_not_a_false_critical_failure(monkeypatch):
    captured = {}
    # Force the finally-block to run (trace_id truthy) and capture the status.
    monkeypatch.setattr(pipeline.tracer, "record_initial_state",
                        lambda *a, **k: "t-test")
    monkeypatch.setattr(
        pipeline.tracer, "record_pipeline_end",
        lambda trace_id, status, errors: captured.update(
            status=status, errors=list(errors or [])),
    )

    events = list(pipeline.build_dashboard_from_df_generator(pd.DataFrame()))

    # A state WAS produced and delivered to the consumer.
    done = [e for e in events if e.get("phase") == "done"]
    assert done and done[-1].get("state") is not None
    # ...so the run must not be recorded as a critical failure.
    assert captured.get("status") != "CRITICAL_FAILURE"
    assert not any(
        "failed to return a state object" in e for e in captured.get("errors", [])
    )
