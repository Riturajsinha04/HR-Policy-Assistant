"""
Configuration module for the HR Policy Assistant.
Loads settings from .env and provides centralised constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env (if present) ──────────────────────────────────────────
load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_PATH = DATA_DIR / "hr_manual.pdf"
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "hr_manual"

# ── LLM ──────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# ── Embeddings ───────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ── Chunking ─────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# ── Retrieval ────────────────────────────────────────────────────────
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "6"))
# Minimum relevance score (0-1) below which a chunk is discarded
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.25"))

# ── Pages with important tables (leave-policy table spans these) ────
TABLE_PAGES = list(range(59, 68))  # 0-indexed pages 59-67 = PDF pages 60-68
