"""Ranking metrics for offline recommendation evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable


def recall_at_k(
    recommended_ids: Iterable[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Fraction of a user's relevant items retrieved in the first k results."""
    if not relevant_ids:
        return 0.0
    recommendations = _deduplicate(recommended_ids, k)
    hits = sum(item_id in relevant_ids for item_id in recommendations)
    return hits / len(relevant_ids)


def ndcg_at_k(
    recommended_ids: Iterable[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Binary-relevance normalized discounted cumulative gain at k."""
    if not relevant_ids:
        return 0.0

    recommendations = _deduplicate(recommended_ids, k)
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item_id in enumerate(recommendations)
        if item_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def _deduplicate(item_ids: Iterable[str], k: int) -> list[str]:
    """Keep ranking order while preventing duplicate recommendations from scoring."""
    unique_ids = []
    seen = set()
    for item_id in item_ids:
        item_id = str(item_id)
        if item_id in seen:
            continue
        seen.add(item_id)
        unique_ids.append(item_id)
        if len(unique_ids) == k:
            break
    return unique_ids
