"""Central configuration. Values come from environment variables / a .env file in the repo root."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
SKILLS_TAXONOMY_PATH = DATA_DIR / "skills_taxonomy.json"
JOBS_CSV_PATH = Path(os.getenv("JOBS_CSV_PATH", DATA_DIR / "jobs_sample.csv"))
EMBEDDING_CACHE_DIR = DATA_DIR / ".cache"

# --- LLM (optional). Without a key the app silently uses the rule-based rewriter. ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- Matching ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _weights() -> dict[str, float]:
    """Hybrid weights (embedding / tfidf / skills). Re-normalised so they always sum to 1."""
    w = {
        "embedding": float(os.getenv("WEIGHT_EMBEDDING", 0.5)),
        "tfidf": float(os.getenv("WEIGHT_TFIDF", 0.5)),
        "skills": float(os.getenv("WEIGHT_SKILLS", 0.0)),
    }
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


DEFAULT_WEIGHTS = _weights()
