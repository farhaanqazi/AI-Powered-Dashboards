#!/usr/bin/env python3
"""
frontend/scripts/analyze_traces.py

Copy of analyze_traces utility to be tracked inside the git repo (frontend).
This script behaves the same as the top-level one and defaults to looking in
`../pipeline_traces` and writes into `./traces_analysis` inside the frontend
folder so that it is easily committed and inspected in CI.

Usage:
  # When run from frontend/ folder:
  python scripts/analyze_traces.py

"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional

PIPE_PREFIX = "PIPELINE_TRACE::"
# When running under frontend/, parent folder contains pipeline_traces
DEFAULT_TRACE_DIR = os.path.join("..", "pipeline_traces")
DEFAULT_OUTPUT_DIR = "traces_analysis"


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)[:200]


def parse_log_lines(lines: List[str]) -> List[Dict[str, Any]]:
    entries = []
    for ln in lines:
        if PIPE_PREFIX in ln:
            try:
                payload = ln.split(PIPE_PREFIX, 1)[1].strip()
                obj = json.loads(payload)
                entries.append(obj)
            except Exception:
                m = re.search(r"(\{.*\})", ln)
                if m:
                    try:
                        obj = json.loads(m.group(1))
                        entries.append(obj)
                    except Exception:
                        continue
    return entries


def read_jsonl_trace_file(path: str) -> List[Dict[str, Any]]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                entries.append(obj)
            except Exception:
                continue
    return entries


def collect_entries_from_trace_dir(trace_dir: str) -> List[Dict[str, Any]]:
    all_entries = []
    if not os.path.isdir(trace_dir):
        return all_entries
    for fname in os.listdir(trace_dir):
        if not fname.endswith(".jsonl") and not fname.endswith(".json"):
            continue
        path = os.path.join(trace_dir, fname)
        all_entries.extend(read_jsonl_trace_file(path))
    return all_entries


# --- Hugging Face Spaces log fetch helper ---
def _try_import_requests():
    try:
        import requests
        return requests
    except Exception:
        return None


def fetch_space_logs(space: str, token: Optional[str] = None, timeout: int = 10) -> List[str]:
    """Try fetching logs from a Hugging Face Space.

    Attempts several endpoints and returns raw text lines (not parsed JSON).
    If the space is private, provide a token (or set HF_TOKEN env var).
    """
    requests_mod = _try_import_requests()

    # Normalize space argument: accept 'owner/space' or full URL
    if space.startswith("http"):
        # Extract owner/space from URL
        m = re.search(r"/spaces/([^/]+/[^/\s]+)", space)
        if m:
            space_path = m.group(1)
        else:
            space_path = space
    else:
        space_path = space

    candidate_urls = [
        f"https://huggingface.co/spaces/{space_path}/logs",
        f"https://huggingface.co/spaces/{space_path}/+/logs",
        f"https://huggingface.co/api/spaces/{space_path}/logs",
    ]

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Try using requests if available
    for url in candidate_urls:
        try:
            if requests_mod:
                resp = requests_mod.get(url, headers=headers or None, timeout=timeout)
                if resp.status_code == 200 and resp.text:
                    return resp.text.splitlines()
            else:
                # Fallback to urllib
                from urllib.request import Request, urlopen
                req = Request(url)
                if token:
                    req.add_header("Authorization", f"Bearer {token}")
                with urlopen(req, timeout=timeout) as r:
                    text = r.read().decode("utf-8", errors="ignore")
                    if text:
                        return text.splitlines()
        except Exception:
            # Try next endpoint
            continue

    # As a last resort, try the Hugging Face Hub API to list runs (if available)
    # and gather logs from run outputs (best-effort, non-critical)
    return []


def group_by_trace_id(entries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        tid = e.get("trace_id")
        if not tid:
            continue
        groups[tid].append(e)
    for tid, evs in groups.items():
        evs.sort(key=lambda x: x.get("timestamp", ""))
    return groups


def iso_to_dt(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def analyze_trace(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    trace_id = events[0].get("trace_id") if events else "unknown"
    first_ts = iso_to_dt(events[0].get("timestamp", "")) if events else None
    last_ts = None
    dataset_name = "unknown"
    status = "UNKNOWN"
    errors: List[str] = []
    event_summaries: List[str] = []

    kpi_count = 0
    chart_count = 0

    for ev in events:
        et = ev.get("event_type", "<unknown>")
        payload = ev.get("payload", {}) or {}
        ts = iso_to_dt(ev.get("timestamp", ""))
        if ts:
            last_ts = ts
        if et == "initial_state":
            dataset_name = payload.get("source_name") or dataset_name
        if et == "profiling_decision":
            dp = payload.get("dataset_profile") or {}
            dataset_name = dp.get("source_name") or dataset_name
        if et == "kpi_generation":
            kpi_count = len((payload.get("kpis") or []))
        if et == "chart_selection":
            chart_count = payload.get("charts_suggested_count") or len(payload.get("specs") or [])
        if et == "pipeline_end":
            status = (payload.get("final_status") or status)
            if payload.get("errors"):
                errors.extend([str(x) for x in payload.get("errors")])
        if isinstance(payload, dict) and ("error" in payload or "errors" in payload):
            if payload.get("error"):
                errors.append(str(payload.get("error")))
        summary = f"{ev.get('timestamp','?')} | {et}"
        if et == "initial_state":
            shape = payload.get("shape")
            if shape:
                summary += f" | shape={shape}"
        if et == "pipeline_end":
            summary += f" | status={payload.get('final_status')}"
        event_summaries.append(summary)

    duration_s = None
    if first_ts and last_ts:
        duration_s = (last_ts - first_ts).total_seconds()

    missing_eda = (chart_count == 0 and kpi_count == 0)

    analysis = {
        "trace_id": trace_id,
        "dataset_name": dataset_name or "unknown",
        "status": status,
        "errors": errors,
        "first_timestamp": first_ts.isoformat() if first_ts else None,
        "last_timestamp": last_ts.isoformat() if last_ts else None,
        "duration_seconds": duration_s,
        "kpi_count": kpi_count,
        "chart_count": chart_count,
        "missing_eda": missing_eda,
        "events": event_summaries,
    }
    return analysis


def format_analysis_text(a: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"Trace ID: {a.get('trace_id')}")
    lines.append(f"Dataset: {a.get('dataset_name')}")
    lines.append(f"Status: {a.get('status')}")
    if a.get("duration_seconds") is not None:
        lines.append(f"Duration: {a.get('duration_seconds'):.2f}s")
    lines.append(f"KPIs generated: {a.get('kpi_count')}")
    lines.append(f"Charts suggested: {a.get('chart_count')}")
    lines.append(f"Missing EDA outputs: {a.get('missing_eda')}")
    lines.append("")
    lines.append("Errors:")
    if a.get('errors'):
        for e in a.get('errors'):
            lines.append(f" - {e}")
    else:
        lines.append(" - None")
    lines.append("")
    lines.append("Event timeline (most relevant events):")
    for ev in a.get('events', [])[:20]:
        lines.append(f" - {ev}")
    return "\n".join(lines)


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR, help="Directory with trace .jsonl files")
    parser.add_argument("--logs-file", default=None, help="A log file (stdout) to parse for PIPELINE_TRACE lines")
    parser.add_argument("--hf-space", default=None, help="Hugging Face Space (owner/space or URL) to fetch logs from")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token or set HF_TOKEN env var")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Where to write analyses")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")

    args = parser.parse_args(argv)

    entries: List[Dict[str, Any]] = []

    if os.path.isdir(args.trace_dir):
        if not args.quiet:
            print(f"Reading trace files from: {args.trace_dir}")
        entries.extend(collect_entries_from_trace_dir(args.trace_dir))

    if args.logs_file:
        if not os.path.isfile(args.logs_file):
            print(f"Logs file not found: {args.logs_file}")
        else:
            if not args.quiet:
                print(f"Parsing logs file: {args.logs_file}")
            with open(args.logs_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            entries.extend(parse_log_lines(lines))

    if args.hf_space:
        if not args.quiet:
            print(f"Fetching logs from Hugging Face Space: {args.hf_space}")
        try:
            hf_lines = fetch_space_logs(args.hf_space, args.hf_token)
            if hf_lines:
                entries.extend(parse_log_lines(hf_lines))
            else:
                if not args.quiet:
                    print("No logs fetched from HF Space or fetch failed.")
        except Exception as e:
            print(f"Failed to fetch logs from HF Space: {e}")

    if not entries:
        print("No trace entries found. Make sure TRACE_TO_STDOUT is false (to write files) or pass --logs-file with exported logs.")
        return

    groups = group_by_trace_id(entries)

    ensure_dir(args.output_dir)

    summary_rows = []

    for tid, evs in groups.items():
        analysis = analyze_trace(evs)
        dataset = analysis.get('dataset_name') or 'unknown'
        safe_name = sanitize_filename(dataset)
        out_base = f"{safe_name}__{tid}"
        txt_path = os.path.join(args.output_dir, out_base + ".txt")
        json_path = os.path.join(args.output_dir, out_base + ".json")
        txt_content = format_analysis_text(analysis)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(txt_content)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)

        if not args.quiet:
            print(f"Wrote analysis: {txt_path}")

        summary_rows.append({
            "trace_id": tid,
            "dataset": dataset,
            "status": analysis.get('status'),
            "errors": len(analysis.get('errors') or []),
            "charts": analysis.get('chart_count'),
            "kpis": analysis.get('kpi_count')
        })

    index_path = os.path.join(args.output_dir, "index.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(summary_rows, f, indent=2)
    if not args.quiet:
        print(f"Wrote index: {index_path}")
        print(f"Analyses written to {args.output_dir} (count: {len(summary_rows)})")


if __name__ == "__main__":
    main()
