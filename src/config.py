import os
import tempfile


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# --- General Configuration ---
APP_TITLE = "AI-Powered Dashboard Generator"
APP_DESCRIPTION = "Upload a CSV or point to a URL/Kaggle dataset and instantly get smart column roles, KPIs, correlations, and ready-to-use visuals with zero manual setup."
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

# --- Pipeline Configuration ---
MAX_ROWS = int(os.environ.get("MAX_ROWS", 500000))  # Increased for financial data analysis
MAX_COLS = int(os.environ.get("MAX_COLS", 50))
MAX_CATEGORIES = int(os.environ.get("MAX_CATEGORIES", 10))
MAX_CHARTS = int(os.environ.get("MAX_CHARTS", 20))
MEMORY_LIMIT_MB = int(os.environ.get("MEMORY_LIMIT_MB", 1000))  # Increased for financial data
TIMEOUT_SECONDS = int(os.environ.get("TIMEOUT_SECONDS", 600))  # Increased for larger datasets

# --- Analyser Configuration ---
UNIQUENESS_CUTOFF = float(os.environ.get("UNIQUENESS_CUTOFF", 0.95))
AVG_LENGTH_CUTOFF = int(os.environ.get("AVG_LENGTH_CUTOFF", 30))
MIN_DATE = int(os.environ.get("MIN_DATE", 1900))
MAX_DATE = int(os.environ.get("MAX_DATE", 2100))

# --- KPI Generator Configuration ---
KPI_TOP_K = int(os.environ.get("KPI_TOP_K", 10))
MIN_VARIABILITY_THRESHOLD = float(os.environ.get("MIN_VARIABILITY_THRESHOLD", 0.01))
MIN_UNIQUE_RATIO = float(os.environ.get("MIN_UNIQUE_RATIO", 0.01))
MAX_UNIQUE_RATIO = float(os.environ.get("MAX_UNIQUE_RATIO", 0.9))

# --- Correlation Engine Configuration ---
MIN_CORRELATION = float(os.environ.get("MIN_CORRELATION", 0.5))
MIN_VARIANCE = float(os.environ.get("MIN_VARIANCE", 0.001))

# --- HTTP Configuration ---
_default_origins = "http://localhost:5173,http://localhost:8000"
CORS_ALLOW_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ALLOW_ORIGINS", _default_origins).split(",")
    if o.strip()
]

# --- AI Analyst (LLM) Configuration ---
# The LLM narrates over and selects from deterministically-computed numbers.
# It never computes or invents figures. If the key is absent or the call fails,
# the pipeline silently falls back to the heuristic Layer 4.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TIMEOUT_SECONDS = int(os.environ.get("GROQ_TIMEOUT_SECONDS", 30))
# Defaults ON only when a key is present; explicit env override always wins.
AI_ANALYST_ENABLED = os.environ.get(
    "AI_ANALYST_ENABLED", "true" if GROQ_API_KEY else "false"
).lower() == "true"

# Phase 9 S9.1 — provider-agnostic AI. `llm_analyst` depends only on the
# `src.analysis.llm.LLMProvider` interface; the concrete provider and model are
# selected here, never hardcoded in analysis code.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
# Deterministic response cache keyed on the ground-truth hash (provider+model+
# system+user). In-process + TTL-bound — no new storage backend (mirrors the
# df_cache invariant). Identical ground truth ⇒ no duplicate paid LLM call.
LLM_RESPONSE_CACHE_ENABLED = _env_bool("LLM_RESPONSE_CACHE_ENABLED", True)
LLM_RESPONSE_CACHE_TTL_SECONDS = int(
    os.environ.get("LLM_RESPONSE_CACHE_TTL_SECONDS", 3600)
)
LLM_RESPONSE_CACHE_MAX_ENTRIES = int(
    os.environ.get("LLM_RESPONSE_CACHE_MAX_ENTRIES", 256)
)

# --- Persistence Configuration ---
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///./_local/dashboards.db",
)
REDIS_URL = os.environ.get("REDIS_URL", "")
DASHBOARD_TTL_SECONDS = int(os.environ.get("DASHBOARD_TTL_SECONDS", 86400))

# --- Async analysis jobs (Phase 10 S10.1, pulled forward) ---
# Decouples auth from the long pipeline: the upload request authenticates and
# returns in <1s; the analysis runs as a job and progress is streamed by id.
#
# JOB_QUEUE_ENABLED=True  → out-of-process Arq worker on Redis (the production-
#                           correct path; run a separate `arq` worker process).
# JOB_QUEUE_ENABLED=False → in-process asyncio task (HF single-container
#                           fallback: still fixes the 401 + the blocked
#                           request; in-flight jobs are lost on restart).
# Either way the frontend uses ONE flow (submit → stream by job id); the
# backend picks the strategy. Legacy /api/upload[/stream] stay intact.
JOB_QUEUE_ENABLED = _env_bool("JOB_QUEUE_ENABLED", False)
# Where the uploaded file is spooled so a separate worker process can read it
# (the worker can't see the request's in-memory bytes).
JOB_SPOOL_DIR = os.environ.get(
    "JOB_SPOOL_DIR", os.path.join(tempfile.gettempdir(), "di_job_spool")
)
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", 86400))
# SSE poll cadence (seconds) for the job event stream.
JOB_STREAM_POLL_SECONDS = float(os.environ.get("JOB_STREAM_POLL_SECONDS", 0.5))

# --- Statistical depth (Phase 9 S9.2) ---
# Deterministic advanced analytics (Spearman/MI/Cramér's V/η, Mann-Kendall +
# STL, IsolationForest/LOF, KMeans/HDBSCAN, skew/kurtosis/normality, RandomForest
# driver analysis). Every figure is computed here, never by the LLM. sklearn/
# statsmodels are optional: each block degrades to {} when a dep is absent so
# the deployed image stays correct (and slim) without them.
# --- DataFrame engine seam (Phase 10 S10.2) ---
# The whole pipeline consumes pandas; this only chooses HOW bytes become a
# frame. "polars"/"duckdb" read larger-than-memory inputs efficiently then
# materialize to pandas (downstream unchanged). Unknown/missing backend ⇒
# graceful pandas fallback.
DATAFRAME_ENGINE = os.environ.get("DATAFRAME_ENGINE", "pandas").strip().lower()

# --- Multi-format ingestion (Phase 10 S10.3) ---
# Extensions accepted by the upload endpoints, behind the parser interface.
INGEST_ALLOWED_FORMATS = [
    s.strip().lower()
    for s in os.environ.get(
        "INGEST_ALLOWED_FORMATS",
        "csv,parquet,xlsx,xls,json,ndjson,jsonl",
    ).split(",")
    if s.strip()
]

# --- Ask Your Data (Phase 11) ---
# Conversational follow-up. The LLM only PROPOSES a deterministic query/stat
# from a fixed tool catalog and narrates the returned numbers — it never
# computes a figure. The bounded agent caps tool iterations so it always
# terminates and every number stays traceable.
ASK_DATA_ENABLED = _env_bool("ASK_DATA_ENABLED", True)
ASK_MAX_ITERATIONS = int(os.environ.get("ASK_MAX_ITERATIONS", 3))

STATISTICAL_DEPTH_ENABLED = _env_bool("STATISTICAL_DEPTH_ENABLED", True)
# Row sample cap (deterministic, seeded) so heavy estimators stay bounded.
STAT_DEPTH_MAX_ROWS = int(os.environ.get("STAT_DEPTH_MAX_ROWS", 20000))
STAT_DEPTH_MAX_COLS = int(os.environ.get("STAT_DEPTH_MAX_COLS", 40))
STAT_DEPTH_RANDOM_STATE = int(os.environ.get("STAT_DEPTH_RANDOM_STATE", 0))

# --- Security Hardening (Phase 0) ---
# Fail-closed posture for the Semantic Contract Layer. When sensitivity cannot
# be determined, treat the dataset as sensitive rather than open. When PII is
# detected, block all LLM egress (no human override unblocks it).
SENSITIVITY_FAIL_CLOSED = _env_bool("SENSITIVITY_FAIL_CLOSED", True)
PII_BLOCK_EGRESS = _env_bool("PII_BLOCK_EGRESS", True)

# Upload hard caps (S0.4). Enforced on the request stream, before full load.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 100 * 1024 * 1024))
# A full Excel worksheet is 1,048,576 rows incl. header. The row cap must
# admit a completely-filled sheet of real data, so default to that size.
EXCEL_MAX_SHEET_ROWS = 1_048_576
MAX_UPLOAD_ROWS = int(os.environ.get("MAX_UPLOAD_ROWS", EXCEL_MAX_SHEET_ROWS))
MAX_UPLOAD_COLS = int(os.environ.get("MAX_UPLOAD_COLS", 512))

# SSRF hardening for URL ingestion (S0.3).
URL_FETCH_ALLOWED_SCHEMES = tuple(
    s.strip().lower()
    for s in os.environ.get("URL_FETCH_ALLOWED_SCHEMES", "http,https").split(",")
    if s.strip()
)
URL_FETCH_MAX_BYTES = int(os.environ.get("URL_FETCH_MAX_BYTES", 100 * 1024 * 1024))
URL_FETCH_TIMEOUT_SECONDS = int(os.environ.get("URL_FETCH_TIMEOUT_SECONDS", 30))
URL_FETCH_MAX_REDIRECTS = int(os.environ.get("URL_FETCH_MAX_REDIRECTS", 3))
# Default deny: never fetch private/link-local/loopback/metadata addresses.
ALLOW_PRIVATE_URL_FETCH = _env_bool("ALLOW_PRIVATE_URL_FETCH", False)

# Per-IP rate limiting (S0.5). slowapi limit strings, e.g. "30/minute".
RATE_LIMIT_ENABLED = _env_bool("RATE_LIMIT_ENABLED", True)
RATE_LIMIT_DEFAULT = os.environ.get("RATE_LIMIT_DEFAULT", "120/minute")
RATE_LIMIT_UPLOAD = os.environ.get("RATE_LIMIT_UPLOAD", "20/minute")

# Diagnostic endpoints are off unless explicitly enabled AND authenticated.
ENABLE_DEBUG_ENDPOINTS = _env_bool("ENABLE_DEBUG_ENDPOINTS", False)

# Admin allow-list for diagnostic endpoints (S0.5). Comma-separated Clerk user
# ids (the JWT 'sub' claim). Empty => nobody is admin, so the debug endpoints
# stay closed even to authenticated users (fail-closed).
ADMIN_USER_IDS = frozenset(
    s.strip()
    for s in os.environ.get("ADMIN_USER_IDS", "").split(",")
    if s.strip()
)

# Secret used to HMAC-sign guest session ids so one guest cannot forge another
# guest's id and read their history (S10.4 IDOR fix). When unset a random
# per-process secret is generated — secure, but guest history will not survive
# a restart or span containers. Set GUEST_SESSION_SECRET in the environment for
# persistent, cross-container guest history.
GUEST_SESSION_SECRET = (
    os.environ.get("GUEST_SESSION_SECRET", "").strip() or os.urandom(32).hex()
)

# --- Ingest Contract Gate (Phase 1) ---
# Case-insensitive exact-match cell values coerced to NA before profiling.
INGEST_SENTINELS = [
    s.strip()
    for s in os.environ.get(
        "INGEST_SENTINELS",
        "na,n/a,nan,null,none,nil,-,--,?,#n/a,#na,#value!,#ref!,unknown,not available",
    ).split(",")
    if s.strip()
]
# A row with at least this fraction of NA cells is dropped as a null row.
INGEST_NULL_ROW_FRACTION = float(os.environ.get("INGEST_NULL_ROW_FRACTION", 1.0))
# An object column is coerced to numeric only if at least this fraction of its
# non-null values parse as numbers after stripping currency/grouping symbols.
INGEST_NUMERIC_COERCE_FRACTION = float(
    os.environ.get("INGEST_NUMERIC_COERCE_FRACTION", 0.95)
)
# Rows sampled per column for PII scanning (bounds Presidio cost).
PII_SAMPLE_ROWS = int(os.environ.get("PII_SAMPLE_ROWS", 200))
# Presidio entity types that count as PII / sensitive.
PII_ENTITY_TYPES = [
    s.strip()
    for s in os.environ.get(
        "PII_ENTITY_TYPES",
        "EMAIL_ADDRESS,PHONE_NUMBER,CREDIT_CARD,US_SSN,IBAN_CODE,IP_ADDRESS,PERSON,US_PASSPORT,US_DRIVER_LICENSE,US_BANK_NUMBER",
    ).split(",")
    if s.strip()
]
# Minimum Presidio confidence for an entity to count.
PII_SCORE_THRESHOLD = float(os.environ.get("PII_SCORE_THRESHOLD", 0.5))

# --- Invariant Critic (Phase 4) — config-driven tolerances ---
# A numeric column this unique (and integer-like) is vetoed to identifier.
CRITIC_ID_UNIQUE_RATIO = float(os.environ.get("CRITIC_ID_UNIQUE_RATIO", 0.99))
# Row-wise |sum(components) - total| / |total| within this is a total match.
CRITIC_TOTAL_TOLERANCE = float(os.environ.get("CRITIC_TOTAL_TOLERANCE", 0.01))
# Per-row share columns summing to 1.0 (or 100) within this tolerance.
CRITIC_SHARE_SUM_TOLERANCE = float(os.environ.get("CRITIC_SHARE_SUM_TOLERANCE", 0.02))
# std / |mean| at or above this flags extreme dispersion.
CRITIC_STD_MEAN_RATIO = float(os.environ.get("CRITIC_STD_MEAN_RATIO", 3.0))
# Minimum rows before the critic trusts a statistical signal.
CRITIC_MIN_ROWS = int(os.environ.get("CRITIC_MIN_ROWS", 12))

# --- Pipeline Wiring (Phase 5) ---
# The contract layer is always compiled, vetoed, cached and threaded into the
# dashboard. The HITL schema-review GATE (halt before L3/L4/EDA/LLM/render
# pending human approval) activates when a dataset does NOT auto-accept (Phase
# 6 S6.3: minimum per-field confidence ≥ threshold ∧ grain ∧ ¬pii_blocked).
# Enabled by default 2026-05-18 — the safety net the upgrade promised is now
# live; set SCHEMA_REVIEW_ENABLED=0 only for a deliberate bypass.
SCHEMA_REVIEW_ENABLED = _env_bool("SCHEMA_REVIEW_ENABLED", True)

# --- LLM Output Validation + Auto-Accept (Phase 6) ---
# Calibrated 2026-05-18 — a compiled contract auto-accepts (auto-locks, no
# human review) only when ALL hold: a grain was detected, the dataset is not
# pii_blocked, the MEAN per-field confidence ≥ AUTO_ACCEPT_CONFIDENCE (overall
# quality bar), AND no single column is below CRITICAL_FIELD_CONFIDENCE_FLOOR
# (the "one catastrophically mis-typed column" guard the audit flagged).
# Two gates instead of a raw minimum: the mean keeps the overall bar while the
# floor only trips on worse-than-coin-flip columns — so normal fuzzy text
# columns (~0.6) no longer force review, but a 0.10 column hidden among 0.95s
# still does.
AUTO_ACCEPT_CONFIDENCE = float(os.environ.get("AUTO_ACCEPT_CONFIDENCE", 0.70))
CRITICAL_FIELD_CONFIDENCE_FLOOR = float(
    os.environ.get("CRITICAL_FIELD_CONFIDENCE_FLOOR", 0.50)
)

# --- Cleaned-DataFrame cache (Phase 7 fix for HITL re-render) ---
# When True, the post-ingest cleaned DataFrame is held in a transient,
# auto-expiring cache (Redis if configured, else in-process), keyed by schema
# fingerprint. This lets a HITL role override actually re-run L3→chart-render
# with the corrected roles. It is NOT a durable store: TTL-bound, fingerprint-
# keyed, and fully disableable here. Disabling falls back to the deterministic
# contract-only re-derivation (override still locks, charts just aren't redrawn).
CLEANED_DF_CACHE_ENABLED = _env_bool("CLEANED_DF_CACHE_ENABLED", True)
CLEANED_DF_CACHE_TTL_SECONDS = int(
    os.environ.get("CLEANED_DF_CACHE_TTL_SECONDS", 3600)
)

# --- Durable cleaned-frame tier (Phase 14 S14.1 — Gap A) ---
# The transient cache above is wiped on container restart, which kills
# follow-up Ask/Interact ("working data has expired"). When enabled, the
# cleaned frame is ALSO written once as a fingerprint-keyed Parquet file to a
# local spool dir — the same filesystem-spool pattern already accepted for
# JOB_SPOOL_DIR, NOT a new storage backend. df_cache.get() falls back
# mem → client → parquet so interactivity survives a restart. Self-pruning by
# age; fully disableable.
CLEANED_DF_DURABLE_ENABLED = _env_bool("CLEANED_DF_DURABLE_ENABLED", True)
CLEANED_DF_DURABLE_DIR = os.environ.get(
    "CLEANED_DF_DURABLE_DIR",
    os.path.join(tempfile.gettempdir(), "di_clean_frames"),
)
CLEANED_DF_DURABLE_TTL_SECONDS = int(
    os.environ.get("CLEANED_DF_DURABLE_TTL_SECONDS", DASHBOARD_TTL_SECONDS)
)

# --- Interactive Dashboard (Phase 14 S14.2) ---
# Structured, non-LLM interactions reuse the Phase 11 deterministic Ask tool
# catalogue over the (now durable) cleaned frame. No WASM, no new engine. The
# AI is not in this path at all. Results are memoised in-process keyed on
# sha256(schema_fingerprint + canonical(spec)) with LRU eviction so repeated
# filter states are instant and the single container memory stays bounded.
INTERACT_ENABLED = _env_bool("INTERACT_ENABLED", True)
INTERACT_RESULT_CACHE_MAX_ENTRIES = int(
    os.environ.get("INTERACT_RESULT_CACHE_MAX_ENTRIES", 256)
)
# Hard cap on filter predicates per interaction request (abuse / cost guard).
INTERACT_MAX_FILTERS = int(os.environ.get("INTERACT_MAX_FILTERS", 12))
