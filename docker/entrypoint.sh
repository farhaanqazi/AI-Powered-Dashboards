#!/bin/sh
# Container entrypoint.
#
# Default (JOB_QUEUE_ENABLED unset/false): runs ONLY the API — byte-for-byte
# the previous behaviour, so Hugging Face is unaffected.
#
# JOB_QUEUE_ENABLED=true: also starts the Arq worker in the same container
# (the production-correct out-of-process path). Requires REDIS_URL to point at
# a reachable Redis — on HF that MUST be an external/managed Redis, because an
# in-container Redis dies with the Space and defeats the durability point.
set -eu

PORT="${PORT:-7860}"

# Start an in-container worker only when the queue is on AND we weren't told
# a dedicated worker process exists elsewhere (RUN_WORKER=false — e.g. the
# docker-compose 'worker' service runs it separately).
_jq="$(printf '%s' "${JOB_QUEUE_ENABLED:-false}" | tr '[:upper:]' '[:lower:]')"
_rw="$(printf '%s' "${RUN_WORKER:-true}" | tr '[:upper:]' '[:lower:]')"
case "$_jq" in
  1|true|yes|on)
    case "$_rw" in
      0|false|no|off)
        echo "[entrypoint] queue on; worker runs as a separate process" ;;
      *)
        echo "[entrypoint] JOB_QUEUE_ENABLED -> starting Arq worker (background)"
        arq src.jobs.worker.WorkerSettings & ;;
    esac
    ;;
  *)
    echo "[entrypoint] in-process jobs (no separate worker)"
    ;;
esac

# exec => uvicorn becomes PID 1 and receives stop signals directly; the
# backgrounded worker is a child and is torn down with the container.
exec python -m uvicorn main:app --host 0.0.0.0 --port "$PORT"
