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

### Later (Phase 10 hardening, not now)

To make HF durable (survive container restarts) you would, deliberately:

1. Set `JOB_QUEUE_ENABLED=true` and point `REDIS_URL` at an **external**
   managed Redis (in-container Redis dies with the Space and defeats the
   purpose).
2. Run the worker as a second process. The single-container way is a process
   manager in the image; append to the Dockerfile CMD, e.g.:

   ```dockerfile
   # supervisord/honcho running BOTH:
   #   uvicorn main:app --host 0.0.0.0 --port 7860
   #   arq src.jobs.worker.WorkerSettings
   ```

   This is a real deployment-topology change — validate the Space boots both
   processes before relying on it. It is intentionally out of scope for the
   incident fix.
