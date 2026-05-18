# SLOs & Error Budgets — Phase 12 S12.1

SLOs are defined over the metrics emitted by `src/observability/metrics.py`
(`http_requests_total`, `http_request_duration_seconds`,
`http_requests_in_flight`, `pipeline_layer_seconds`). The Prometheus rules in
`observability/prometheus/alerts.yml` alert when the budget burns too fast;
the Grafana board in `observability/grafana/dashboard.json` visualises them.

## Service Level Objectives

| # | SLO | SLI | Target | Window |
|---|-----|-----|--------|--------|
| 1 | API availability | `1 - (5xx / total)` over non-job paths | **99.5%** | 30d |
| 2 | API latency (fast paths) | p95 `http_request_duration_seconds` excluding the pipeline | **< 1.0s** | 30d |
| 3 | Upload submit latency | p95 of `POST /api/jobs/upload` (auth + spool only) | **< 2.0s** | 30d |
| 4 | Analysis success rate | jobs reaching `done` / jobs created | **99.0%** | 30d |
| 5 | Pipeline duration | p95 total `pipeline_layer_seconds` summed per run | **< 240s** | 30d |

> SLO 3 is the one that justified pulling S10.1 forward: the long pipeline
> moved off the request, so the auth-bearing call only needs to be fast and
> reliable for the sub-second submit.

## Error budgets

- **Availability (SLO 1):** 0.5% of 30d ≈ **3h 39m** of allowed error time.
- **Analysis success (SLO 4):** 1.0% of runs may fail terminally.

Burn-rate alerting (multi-window, Google SRE workbook): page on a **2%**
budget consumption in 1h (14.4× burn) AND 5% in 6h; ticket on slower burns.
Exact expressions live in `alerts.yml`.

## Review cadence

Error budget is reviewed at the Phase 12 weekly. If exhausted, feature work
pauses for reliability until the budget recovers (policy mirrors the "gate
production deploys" rule — reliability outranks new scope).
