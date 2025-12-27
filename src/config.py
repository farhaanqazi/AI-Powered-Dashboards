import os

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
MIN_CORRELATION = float(os.environ.get("MIN_CORRELATION", 0.1))
MIN_VARIANCE = float(os.environ.get("MIN_VARIANCE", 0.001))
