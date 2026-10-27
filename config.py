"""Central settings for the SEC filing Q&A project.

Everything here can be overridden with environment variables or a `.env` file
(copy `.env.example` to `.env`). Secrets are read through functions so they are
looked up at call time and never imported as constants.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# ---------------------------------------------------------------- chunking
# ~1000 characters fits inside the 256-token window of all-MiniLM-L6-v2.
CHUNK_SIZE = _int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)
MIN_CHUNK_CHARS = 30
MIN_TEXT_LENGTH = 2000  # filings shorter than this are treated as failed downloads

# ------------------------------------------------------------ vector store
# Named "chroma_db" (not "vector_store") so it never clashes with vector_store.py.
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", str(BASE_DIR / "chroma_db"))
COLLECTION_NAME = "sec_filings"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = _int("TOP_K", 6)
BATCH_SIZE = _int("BATCH_SIZE", 200)  # chunks embedded and written per call

# --------------------------------------------------------------------- LLM
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
LLM_TEMPERATURE = 0.0
LLM_TIMEOUT_SECONDS = _int("LLM_TIMEOUT_SECONDS", 90)
LLM_MAX_RETRIES = _int("LLM_MAX_RETRIES", 3)
# Maximum characters of filing text sent to the model per question.
MAX_CONTEXT_CHARS = _int("MAX_CONTEXT_CHARS", 12000)

# ------------------------------------------------------------- SEC EDGAR
DEFAULT_TICKERS = [t.strip().upper() for t in os.getenv("TICKERS", "AAPL,MSFT,TSLA").split(",") if t.strip()]
FORMS = ("10-K", "10-Q")
FILINGS_PER_FORM = {"10-K": _int("FILINGS_10K", 2), "10-Q": _int("FILINGS_10Q", 2)}
# SEC allows at most 10 requests/second; stay well below that.
SEC_REQUEST_DELAY = 0.25
SEC_MAX_RETRIES = 3


def google_api_key() -> str:
    """Gemini API key (set GOOGLE_API_KEY in .env)."""
    return os.getenv("GOOGLE_API_KEY", "").strip()


def sec_user_agent() -> str:
    """SEC requires a User-Agent that names you and gives a contact email."""
    return os.getenv("SEC_USER_AGENT", "").strip()
