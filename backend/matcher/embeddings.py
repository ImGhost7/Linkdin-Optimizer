"""Sentence-transformer embeddings (all-MiniLM-L6-v2) with an on-disk cache for the job corpus.

all-MiniLM-L6-v2 maps text to a 384-d vector; cosine similarity between two vectors measures
semantic closeness (so "built ML pipelines" ~ "machine learning engineer" even with no shared
words). It reads at most 256 word-pieces, so very long texts are truncated - which is why we put
the most informative text (title, skills) first when building job/profile texts.
The model is downloaded once from the Hugging Face Hub, then runs fully offline.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

from .. import config


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer     # heavy import: only when first needed
    return SentenceTransformer(config.EMBEDDING_MODEL)


def encode(texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
    """L2-normalised embeddings, shape (n, 384). Normalised => cosine similarity is a plain dot product."""
    return _model().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=show_progress, convert_to_numpy=True)


def corpus_embeddings(texts: list[str], show_progress: bool = True) -> np.ndarray:
    """Embeddings for the job corpus, cached in data/.cache keyed by (model, texts) so edits invalidate it."""
    key = hashlib.sha1((config.EMBEDDING_MODEL + "\x00" + "\x00".join(texts)).encode("utf-8")).hexdigest()[:16]
    path = config.EMBEDDING_CACHE_DIR / f"jobs_{key}.npy"
    if path.exists():
        return np.load(path)
    emb = encode(texts, show_progress=show_progress).astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb
