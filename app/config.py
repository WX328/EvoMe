import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
EVOME_DB_URL = os.getenv("EVOME_DB_URL", "sqlite:///./evome.db")

EVOME_QDRANT_PATH = os.getenv("EVOME_QDRANT_PATH", "./evome_qdrant")
HYPOTHESIS_TOP_K = int(os.getenv("HYPOTHESIS_TOP_K", "5"))
HYPOTHESIS_SCORE_THRESHOLD = float(os.getenv("HYPOTHESIS_SCORE_THRESHOLD", "0.60"))
HYPOTHESIS_RELATION_CONFIDENCE = float(os.getenv("HYPOTHESIS_RELATION_CONFIDENCE", "0.60"))

REVISION_ANALYSIS_MIN_INDEPENDENT_SIGNALS = int(os.getenv("REVISION_ANALYSIS_MIN_INDEPENDENT_SIGNALS", "2"))

REVISION_ANALYSIS_MAX_PER_RUN = int(os.getenv("REVISION_ANALYSIS_MAX_PER_RUN", "2"))
