# Observability & Reliability (Phase 12 S12.1)

Ops artifacts for the SLO/error-budget program. Code-side metrics are emitted
by `src/observability/` (Prometheus `/metrics`, OTel tracing, structlog).

| File | Purpose |
|------|---------|
| `slo.md` | SLO definitions, error budgets, burn-rate policy |
| `prometheus/alerts.yml` | Alerting rules (multi-window burn rate) over emitted metrics |
| `grafana/dashboard.json` | SLO overview board (import into Grafana, pick a Prometheus datasource) |
| `load/locustfile.py` | Load test for the SLO-critical submit/poll/ask paths |

Wiring (deployment-gated, per the "gate production deploys" rule):
1. Point Prometheus at the app `/metrics` and load `alerts.yml` as a rule file.
2. Import `grafana/dashboard.json`.
3. Run `locust -f observability/load/locustfile.py --host <url>` pre-release to
   validate SLO 3/4 under concurrency.

These artifacts are intentionally infra-side: enabling them is an ops action,
not a code change, so they ship without touching the runtime image.
