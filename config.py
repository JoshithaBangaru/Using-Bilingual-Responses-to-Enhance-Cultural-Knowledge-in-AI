"""Central configuration. Every other module imports its settings from here."""
import os
from pathlib import Path

try:  # python-dotenv is optional; a .env file is read if it is installed
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------- LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Change the model in ONE place (env var or .env). Run `python check_setup.py`
# to list the models your key can actually use.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
USE_CACHE = os.getenv("LLM_CACHE", "1") == "1"      # LLM_CACHE=0 for latency measurements
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))

# ---------------------------------------------------------------- paths
DATA_PATH = str(BASE_DIR / "cultural_data.json")
INDEX_PATH = str(BASE_DIR / "faiss_index.bin")
METADATA_PATH = str(BASE_DIR / "metadata.json")
DB_PATH = str(BASE_DIR / "pipeline_records.db")      # all pipeline / evaluation logs
CACHE_DB_PATH = str(BASE_DIR / "llm_cache.db")       # cached LLM responses

# ---------------------------------------------------------------- retrieval
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"    # KB embeddings (d=1024)
SIMILARITY_MODEL_NAME = "intfloat/multilingual-e5-base"    # answer-vs-answer similarity
TOP_K = 3

# ---------------------------------------------------------------- disagreement
# Tune this on a dev split with analyze.py (threshold sweep), then fix it.
DISAGREEMENT_THRESHOLD = float(os.getenv("DISAGREEMENT_THRESHOLD", "0.85"))

# ---------------------------------------------------------------- countries
# These five countries are all BLEnD countries and all present in the KB.
COUNTRY_LANGUAGE = {
    "South Korea": "Korean",
    "Ethiopia": "Amharic",
    "China": "Chinese",
    "Indonesia": "Indonesian",
    "Azerbaijan": "Azerbaijani",
}
COUNTRIES = list(COUNTRY_LANGUAGE)

# Resource level of each local language (the BLEnD paper's motivation: local language helps for
# high-resource cultures, English helps for low-resource ones). VERIFY these against the
# BLEnD paper's language table before reporting.
RESOURCE_LEVEL = {
    "China": "high",
    "South Korea": "mid",
    "Indonesia": "mid",
    "Ethiopia": "low",
    "Azerbaijan": "low",
}

# Pipeline variants used for the evaluation / ablation study
MODES = [
    "english_only",      # baseline 1: English answer only
    "local_only",        # baseline 2: local-language answer only
    "dual_judge_nokb",   # ablation:   judge on disagreement, WITHOUT KB facts
    "dual_kb",           # main system: judge on disagreement, WITH KB facts
    "dual_kb_always",    # ablation:   judge + KB facts on every question
]


COUNTRY_CODE = {          # BLEnD country codes (used as split names on Hugging Face)
    "South Korea": "KR",
    "Ethiopia": "ET",
    "China": "CN",
    "Indonesia": "ID",
    "Azerbaijan": "AZ",
}


def normalize_country(name):
    """Map dataset spellings ('South_Korea', 'KR', 'china') to the names used here."""
    raw = str(name).strip()
    key = raw.lower().replace("_", " ")
    for country, code in COUNTRY_CODE.items():
        if key == country.lower() or raw.upper() == code:
            return country
    return raw
