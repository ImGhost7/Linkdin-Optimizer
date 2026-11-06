"""Ranking metrics. Input is always `grades`: the relevance grade of each item in RANKED order
(best-scored first); grade 0 = irrelevant, 1 = adjacent/partly relevant, 2 = clearly relevant.
`total_relevant` / `all_grades` describe the whole candidate pool (needed for recall and ideal DCG).
"""
from __future__ import annotations

import math
from typing import Sequence


def precision_at_k(grades: Sequence[int], k: int) -> float:
    """Fraction of the top-k that is relevant (grade >= 1)."""
    return sum(g > 0 for g in grades[:k]) / k


def recall_at_k(grades: Sequence[int], k: int, total_relevant: int) -> float:
    """Fraction of ALL relevant items that appear in the top-k."""
    return sum(g > 0 for g in grades[:k]) / total_relevant if total_relevant else 0.0


def reciprocal_rank(grades: Sequence[int]) -> float:
    """1 / rank of the first relevant item (0 if none). Averaged over queries this is MRR."""
    for i, g in enumerate(grades, start=1):
        if g > 0:
            return 1.0 / i
    return 0.0


def dcg_at_k(grades: Sequence[int], k: int) -> float:
    """Discounted cumulative gain with exponential gain (2^g - 1): a grade-2 hit is worth 3x a grade-1 hit,
    and a hit at rank i is discounted by log2(i + 1)."""
    return sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(grades[:k], start=1))


def ndcg_at_k(grades: Sequence[int], k: int, all_grades: Sequence[int]) -> float:
    """DCG normalised by the best possible DCG for this pool (ideal ordering) -> in [0, 1]."""
    ideal = dcg_at_k(sorted(all_grades, reverse=True), k)
    return dcg_at_k(grades, k) / ideal if ideal > 0 else 0.0


def all_metrics(grades: Sequence[int], all_grades: Sequence[int]) -> dict[str, float]:
    total_rel = sum(g > 0 for g in all_grades)
    return {
        "P@5": precision_at_k(grades, 5),
        "R@10": recall_at_k(grades, 10, total_rel),
        "MRR": reciprocal_rank(grades),
        "NDCG@10": ndcg_at_k(grades, 10, all_grades),
    }
