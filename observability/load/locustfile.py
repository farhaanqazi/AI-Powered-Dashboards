"""Phase 12 S12.1 — Locust load test.

Drives the SLO-critical paths: the fast auth-bearing upload submit, status
polling, and the deterministic Ask follow-up. Run:

    locust -f observability/load/locustfile.py --host http://localhost:8000

Validates SLO 3 (submit p95 < 2s) and SLO 4 (analysis success rate) under
concurrency. Guest mode keeps it auth-free for load purposes.
"""
from __future__ import annotations

import io
import time

from locust import HttpUser, between, task

_CSV = b"region,revenue,units\n" + b"\n".join(
    f"R{i % 5},{100 + i},{i % 9}".encode() for i in range(500)
)


class DashboardUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.client.headers.update(
            {"X-Guest-Mode": "1", "X-Guest-Session-Id": f"load-{id(self)}"}
        )

    @task(3)
    def submit_and_poll(self):
        files = {"dataset": ("load.csv", io.BytesIO(_CSV), "text/csv")}
        with self.client.post("/api/jobs/upload", files=files,
                              catch_response=True) as r:
            if r.status_code != 202:
                r.failure(f"submit not 202: {r.status_code}")
                return
            job_id = r.json().get("job_id")
        deadline = time.time() + 90
        while time.time() < deadline:
            s = self.client.get(f"/api/jobs/{job_id}").json()
            if s["status"] in ("done", "failed", "cancelled"):
                if s["status"] != "done":
                    self.environment.events.request.fire(
                        request_type="JOB", name="analysis",
                        response_time=0, response_length=0,
                        exception=RuntimeError(s["status"]),
                    )
                return
            time.sleep(1)

    @task(1)
    def ask(self):
        self.client.post("/api/ask", json={"question": "average revenue?"})

    @task(1)
    def health(self):
        self.client.get("/healthz")
