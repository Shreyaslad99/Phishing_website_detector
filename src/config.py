"""Central configuration constants."""
import os

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "reports")

# Data caps (to fit 8 GB RAM / CPU-only)
MAX_URL_ROWS = 50_000       # URL-only processing cap (lowered from 100k due to streaming speed)
MAX_HTML_ROWS = 20_000       # Rows where we also fetch/parse HTML
STREAM_CHUNK = 2_000         # Streaming chunk size

# HuggingFace dataset
HF_DATASET_ID = "phreshphish/phreshphish"

# UCI cross-test dataset
UCI_REPO_ID = 967  # PhiUSIIL

# Training time guard
MAX_TRAIN_MINUTES = 30

# API cascade thresholds (tuned later, defaults)
STAGE1_LOW = 0.10   # below this -> legitimate (tuned)
STAGE1_HIGH = 0.6236  # above this -> phishing (tuned)

# Fetch limits
FETCH_TIMEOUT_S = 5
FETCH_MAX_BYTES = 2 * 1024 * 1024  # 2 MB
FETCH_MAX_REDIRECTS = 3

# Random seed
SEED = 42
