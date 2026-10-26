"""Central settings for NoteWise AI.

Every value can be overridden with an environment variable, so nothing
machine-specific is hard-coded and no secrets live in the source code.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("NOTEWISE_DATA_DIR", BASE_DIR / "data"))
CHROMA_DIR = DATA_DIR / "chroma"          # vector index (ChromaDB)
UPLOAD_DIR = DATA_DIR / "uploads"         # copies of uploaded files
DB_PATH = DATA_DIR / "notewise.db"        # SQLite chat history (Phase 3)

# ---------- Ollama / models ----------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL = os.getenv("NOTEWISE_CHAT_MODEL", "gemma4:e4b")
EMBED_MODEL = os.getenv("NOTEWISE_EMBED_MODEL", "embeddinggemma")

# ---------- Upload limits ----------
MAX_FILE_MB = _env_int("NOTEWISE_MAX_FILE_MB", 20)
MAX_PAGES = _env_int("NOTEWISE_MAX_PAGES", 30)

# ---------- Extraction / OCR ----------
# A page with fewer characters than this is treated as "scanned" and sent to OCR.
MIN_CHARS_FOR_TEXT_PAGE = _env_int("NOTEWISE_MIN_CHARS_FOR_TEXT_PAGE", 40)
OCR_DPI = _env_int("NOTEWISE_OCR_DPI", 300)
OCR_LANG = os.getenv("NOTEWISE_OCR_LANG", "eng")

# ---------- Chunking ----------
CHUNK_WORDS = _env_int("NOTEWISE_CHUNK_WORDS", 90)
CHUNK_OVERLAP_WORDS = _env_int("NOTEWISE_CHUNK_OVERLAP_WORDS", 20)

# ---------- Retrieval ----------
TOP_K = _env_int("NOTEWISE_TOP_K", 5)
# Cosine similarity (0..1) the best passage must reach before we treat a question
# as "covered by the notes". Calibrated with scripts/check_retrieval.py on a
# MacBook Air M2 (90-word chunks): off-topic questions scored 0.05-0.17,
# real ones 0.41-0.67.
MIN_RELEVANCE = _env_float("NOTEWISE_MIN_RELEVANCE", 0.29)

# ---------- Generation (Gemma 4) ----------
# Ollama's default context for Gemma 4 is small (4K). RAG sends ~5 passages plus
# a few chat turns, so 8K is plenty and keeps answers fast on 16 GB RAM.
NUM_CTX = _env_int("NOTEWISE_NUM_CTX", 8192)
TEMPERATURE = _env_float("NOTEWISE_TEMPERATURE", 0.2)   # low = sticks closer to the notes
KEEP_ALIVE = os.getenv("NOTEWISE_KEEP_ALIVE", "30m")    # keep the model loaded between questions
MAX_HISTORY_TURNS = _env_int("NOTEWISE_MAX_HISTORY_TURNS", 6)

# ---------- Quiz ----------
QUIZ_QUESTIONS = _env_int("NOTEWISE_QUIZ_QUESTIONS", 5)
# Second Gemma call per question that answers it without the key; mismatches are
# regenerated. Slower (~5s per question) but catches wrong keys and ambiguous options.
QUIZ_VERIFY = os.getenv("NOTEWISE_QUIZ_VERIFY", "true").lower() not in ("0", "false", "no")
