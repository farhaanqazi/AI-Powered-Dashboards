# Local development — async analysis jobs (Option B)

The analysis pipeline is long-running. It now runs as a **job**, not inside the
upload request. There are two execution strategies, chosen by one env flag.

## The two modes

| `JOB_QUEUE_ENABLED` | Where work runs | Needs | Durability | Use for |
|---|---|---|---|---|
| `false` (default) | in-process `asyncio` task | nothing | lost on restart | Hugging Face / quick local |
| `true` | out-of-process **Arq** worker | Redis + a worker process | survives API restarts | local "production-correct", real hosts |

Either way the upload request authenticates and returns in <1 s with a
`job_id`; progress streams from `GET /api/jobs/{id}/events`. The Clerk token
only has to survive the submit, so a long analysis can no longer 401.

## Run locally — production-correct (Option B, out-of-process worker)

Prereqs: a local Redis (e.g. `docker run -p 6379:6379 redis:7-alpine`).

```
# .env
JOB_QUEUE_ENABLED=true
REDIS_URL=redis://localhost:6379
```

Then run **two processes** (the helper script does both):

```powershell
./scripts/run-local.ps1
```

or manually:

```powershell
# terminal 1 — API
./venv/Scripts/python.exe -m uvicorn main:app --reload --port 8000
# terminal 2 — worker
./venv/Scripts/arq src.jobs.worker.WorkerSettings
# terminal 3 — frontend
cd frontend; npm run dev
```

Restarting the API no longer loses an in-flight analysis — the worker owns it.

## Run locally — full production-like stack (Docker Compose)

One command brings up the real out-of-process topology — API, a **separate**
Arq worker, Redis, and Postgres (shared DB so the worker's result is visible
to the API; a per-container SQLite would not be):

```
docker compose up --build
# app on http://localhost:7860
```

This is the closest local mirror of a correct production deployment — use it
to validate Option B end-to-end before trusting it.

## Run locally — quick (in-process, no Redis/worker)

Leave `JOB_QUEUE_ENABLED=false` (default). Just run the API + frontend; jobs
run on a background thread inside the API. This is identical to how Hugging
Face will behave.

## The test gate

`pytest` exercises the in-process path (no Redis/worker needed):

```
./venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
```

## Hugging Face deployment — what to do with the existing code

**Nothing is deleted or rewritten.** The decision was deliberate:

* `JOB_QUEUE_ENABLED` defaults to **false**, so the HF Space keeps running as a
  single container with no Redis and no worker — exactly as before — yet still
  gets the incident fix (auth on the fast submit, pipeline off the request
  thread). No Dockerfile/topology change is required to ship it.
* The legacy `POST /api/upload` and `POST /api/upload/stream` endpoints are
  **kept** (back-compat + tests). The frontend now uses the job flow; the old
  endpoints remain callable.
* The pipeline itself (`src/core/pipeline.py`) is **unchanged** — the job layer
  only changed *who calls it*.

### The Docker image already supports both modes

`docker/entrypoint.sh` (wired into the Dockerfile `CMD`) runs **only uvicorn
by default**, so HF is byte-for-byte unchanged unless you opt in. When
`JOB_QUEUE_ENABLED=true` it also starts the Arq worker in-container (unless
`RUN_WORKER=false`, used by the compose `worker` service).

### To make HF durable later (deliberate, not automatic)

The image plumbing is done — no Dockerfile edits remain. What's left is an
infrastructure decision only:

1. Set Space secrets: `JOB_QUEUE_ENABLED=true` and `REDIS_URL` pointing at an
   **external/managed Redis** (an in-container Redis dies with the Space and
   defeats the durability point — this is the real cost, not code).
2. (Optional) point `DATABASE_URL` at managed Postgres so results survive
   restarts too.

Until you do that, leave the flag unset: HF keeps working exactly as before,
still with the incident fix (auth on fast submit, pipeline off the request).
