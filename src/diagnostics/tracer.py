import pandas as pd
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

# --- Configuration ---
# Use environment variable for control. Default to 'false'.
# This allows enabling/disabling without code changes, e.g., PIPELINE_TRACE_ENABLED=true
TRACE_ENABLED = os.environ.get("PIPELINE_TRACE_ENABLED", "false").lower() == "true"
TRACE_DIR = "pipeline_traces" # Top-level directory for traces

# Ensure trace directory exists if tracing is enabled
if TRACE_ENABLED and not os.path.exists(TRACE_DIR):
    os.makedirs(TRACE_DIR)

def _get_trace_file_path(trace_id: str) -> str:
    """Returns the full path for a given trace_id."""
    return os.path.join(TRACE_DIR, f"{trace_id}.json")

def _read_trace_file(trace_id: str) -> Dict[str, Any]:
    """Reads an existing trace file."""
    file_path = _get_trace_file_path(trace_id)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

def _write_trace_file(trace_id: str, data: Dict[str, Any]):
    """Writes (or overwrites) a trace file."""
    file_path = _get_trace_file_path(trace_id)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

# --- Public API ---

def record_initial_state(df: pd.DataFrame, source_name: str) -> str:
    """
    Records the initial state of the DataFrame and generates a unique trace_id.
    Returns the generated trace_id.
    """
    if not TRACE_ENABLED:
        return ""
    
    trace_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    
    # Collect initial DF metadata
    initial_state_data = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        "head_sample": df.head(5).to_dict('records'),
        "tail_sample": df.tail(5).to_dict('records'),
        "unique_counts": df.nunique().to_dict()
    }
    
    trace_data = {
        "trace_id": trace_id,
        "timestamp_start": datetime.now().isoformat(),
        "source_name": source_name,
        "status": "IN_PROGRESS",
        "initial_state": initial_state_data,
        "events": [] # To store subsequent events
    }
    _write_trace_file(trace_id, trace_data)
    return trace_id

def record_profiling_decision(trace_id: str, dataset_profile: Dict[str, Any]):
    """
    Records the complete dataset profile output by the analyser.
    """
    if not TRACE_ENABLED or not trace_id:
        return
    
    trace_data = _read_trace_file(trace_id)
    trace_data["profiling_decision"] = dataset_profile
    _write_trace_file(trace_id, trace_data)

def record_kpi_generation(trace_id: str, kpis: List[Dict[str, Any]]):
    """
    Records the list of generated KPIs.
    """
    if not TRACE_ENABLED or not trace_id:
        return
    
    trace_data = _read_trace_file(trace_id)
    trace_data["kpi_generation"] = kpis
    _write_trace_file(trace_id, trace_data)

def record_chart_selection(trace_id: str, charts: List[Dict[str, Any]]):
    """
    Records the list of suggested chart specifications.
    """
    if not TRACE_ENABLED or not trace_id:
        return
    
    trace_data = _read_trace_file(trace_id)
    trace_data["chart_selection"] = {
        "charts_suggested_count": len(charts),
        "specs": charts # Store the full specs
    }
    _write_trace_file(trace_id, trace_data)

def record_pipeline_end(trace_id: str, status: str = "SUCCESS", errors: Optional[List[str]] = None):
    """
    Records the end of the pipeline run and its final status.
    """
    if not TRACE_ENABLED or not trace_id:
        return
    
    trace_data = _read_trace_file(trace_id)
    trace_data["timestamp_end"] = datetime.now().isoformat()
    trace_data["status"] = status
    if errors:
        trace_data["errors"] = errors
    _write_trace_file(trace_id, trace_data)

def record_custom_event(trace_id: str, event_name: str, details: Dict[str, Any]):
    """
    Records a custom event or arbitrary data point in the trace.
    """
    if not TRACE_ENABLED or not trace_id:
        return
    
    trace_data = _read_trace_file(trace_id)
    trace_data["events"].append({
        "timestamp": datetime.now().isoformat(),
        "event_name": event_name,
        "details": details
    })
    _write_trace_file(trace_id, trace_data)
