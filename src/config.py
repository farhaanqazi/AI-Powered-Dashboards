import os


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

# --- Persistence Configuration ---
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///./_local/dashboards.db",
)
REDIS_URL = os.environ.get("REDIS_URL", "")
DASHBOARD_TTL_SECONDS = int(os.environ.get("DASHBOARD_TTL_SECONDS", 86400))

# --- Security Hardening (Phase 0) ---
# Fail-closed posture for the Semantic Contract Layer. When sensitivity cannot
# be determined, treat the dataset as sensitive rather than open. When PII is
# detected, block all LLM egress (no human override unblocks it).
SENSITIVITY_FAIL_CLOSED = _env_bool("SENSITIVITY_FAIL_CLOSED", True)
PII_BLOCK_EGRESS = _env_bool("PII_BLOCK_EGRESS", True)

# Upload hard caps (S0.4). Enforced on the request stream, before full load.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 100 * 1024 * 1024))
MAX_UPLOAD_ROWS = int(os.environ.get("MAX_UPLOAD_ROWS", 1_000_000))
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
# pending human approval) is shipped inert here: its activation criterion is
# the auto-accept confidence rule (Phase 6 S6.3) and its UI is Phase 7. Until
# then this stays False so behaviour — and the local pytest merge gate — is
# unchanged.
SCHEMA_REVIEW_ENABLED = _env_bool("SCHEMA_REVIEW_ENABLED", False)

# --- LLM Output Validation + Auto-Accept (Phase 6) ---
# A compiled contract auto-accepts (auto-locks, no human review) only when the
# minimum per-field role confidence is at least this AND the dataset is not
# pii_blocked AND a grain was detected. Below it → needs schema review.
AUTO_ACCEPT_CONFIDENCE = float(os.environ.get("AUTO_ACCEPT_CONFIDENCE", 0.70))

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
