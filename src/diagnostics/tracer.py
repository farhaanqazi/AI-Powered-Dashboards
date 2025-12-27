import pandas as pd
import json
import os
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# --- Configuration ---
TRACE_ENABLED = os.environ.get("PIPELINE_TRACE_ENABLED", "false").lower() == "true"
TRACE_TO_STDOUT = os.environ.get("PIPELINE_TRACE_TO_STDOUT", "true").lower() == "true" # Default to stdout for cloud environments
TRACE_DIR = "pipeline_traces"

# --- JSON Serialization Helper ---
def _universal_serializer(obj):
    """Handles non-serializable types like pandas/numpy objects."""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return json.loads(obj.to_json(orient='split', date_format='iso'))
    try:
        return str(obj) # Fallback to string representation
    except Exception:
        return f"Unserializable type: {type(obj).__name__}"

def _log_event(trace_id: str, event_type: str, payload: Dict[str, Any]):
    """Internal function to log a trace event to either stdout or a file."""
    if not TRACE_ENABLED:
        return

    log_entry = {
        "trace_id": trace_id,
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "payload": payload
    }

    try:
        log_line = json.dumps(log_entry, default=_universal_serializer)
        
        if TRACE_TO_STDOUT:
            # Prefix to make the log line easily searchable in cloud logs
            print(f"PIPELINE_TRACE::{log_line}")
        else:
            # Write to a JSON Lines file if not in stdout mode
            if not os.path.exists(TRACE_DIR):
                os.makedirs(TRACE_DIR)
            file_path = os.path.join(TRACE_DIR, f"{trace_id}.jsonl")
            with open(file_path, 'a') as f:
                f.write(log_line + '\n')

    except Exception as e:
        # If logging fails, print an error to stderr to avoid crashing the app
        error_log = {
            "trace_id": trace_id,
            "event_type": "LOGGING_FAILURE",
            "error": str(e)
        }
        print(f"PIPELINE_TRACE_ERROR::{json.dumps(error_log)}")

# --- Public API ---

def record_initial_state(df: pd.DataFrame, source_name: str) -> str:
    """
    Records the initial state of the DataFrame, generates a unique trace_id, and logs it.
    Returns the generated trace_id.
    """
    trace_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    if not TRACE_ENABLED:
        return trace_id # Return ID even if not enabled, so caller doesn't fail

    try:
        payload = {
            "source_name": source_name,
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "head_sample": df.head(5), # Serializer will handle DataFrame
            "tail_sample": df.tail(5),
            "unique_counts": df.nunique()
        }
    except Exception as e:
        payload = {"error": f"Could not serialize initial state: {e}"}
    
    _log_event(trace_id, "initial_state", payload)
    return trace_id

def record_profiling_decision(trace_id: str, dataset_profile: Dict[str, Any]):
    """Records the complete dataset profile output by the analyser."""
    _log_event(trace_id, "profiling_decision", {"dataset_profile": dataset_profile})

def record_kpi_generation(trace_id: str, kpis: List[Dict[str, Any]]):
    """Records the list of generated KPIs."""
    _log_event(trace_id, "kpi_generation", {"kpis": kpis})

def record_chart_selection(trace_id: str, charts: List[Dict[str, Any]]):
    """Records the list of suggested chart specifications."""
    _log_event(trace_id, "chart_selection", {
        "charts_suggested_count": len(charts),
        "specs": charts
    })

def record_pipeline_end(trace_id: str, status: str = "SUCCESS", errors: Optional[List[str]] = None):
    """Records the end of the pipeline run and its final status."""
    payload = {"final_status": status}
    if errors:
        payload["errors"] = errors
    _log_event(trace_id, "pipeline_end", payload)

def record_custom_event(trace_id: str, event_name: str, details: Dict[str, Any]):
    """Records a custom event or arbitrary data point in the trace."""
    _log_event(trace_id, f"custom_event_{event_name}", {"details": details})
