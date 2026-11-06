"""Process-wide singleton so the API loads jobs + embeddings once, not per request."""
from __future__ import annotations

from functools import lru_cache

from .hybrid import HybridMatcher
from .jobs import load_jobs


@lru_cache(maxsize=1)
def get_matcher() -> HybridMatcher:
    return HybridMatcher(load_jobs())


def warm_up() -> HybridMatcher:
    """Load jobs AND the embedding model now, so the first user request is as fast as the rest.
    (Job embeddings come from the disk cache, but the model itself is loaded lazily on first use.)"""
    matcher = get_matcher()
    matcher.encode(["warm up"])
    return matcher
