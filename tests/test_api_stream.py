"""Tests for POST /api/upload/stream (Server-Sent Events)."""
import json


def _parse_sse(body_text: str):
    events = []
    for chunk in body_text.split("\n\n"):
        chunk = chunk.strip()
        if not chunk.startswith("data: "):
            continue
        events.append(json.loads(chunk[len("data: "):]))
    return events


def test_stream_emits_expected_phases_in_order(client, upload_files):
    with client.stream("POST", "/api/upload/stream", files=upload_files) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = response.read().decode("utf-8")

    events = _parse_sse(body)
    phases = [e["phase"] for e in events]
    assert phases[0] in ("reading", "preparing", "profiling")
    assert phases[-1] == "done"
    expected_phase_set = {
        "reading", "preparing", "ingest_gate", "profiling", "classifying",
        "relating", "eda", "kpis", "rendering", "done",
    }
    assert set(phases).issubset(expected_phase_set | {"error"})
    assert events[-1]["percent"] == 100
    assert "data" in events[-1]
    assert "trace_id" in events[-1]


def test_stream_rejects_non_csv(client):
    import io
    files = {"dataset": ("x.txt", io.BytesIO(b"a,b\n1,2\n"), "text/plain")}
    response = client.post("/api/upload/stream", files=files)
    assert response.status_code == 400
