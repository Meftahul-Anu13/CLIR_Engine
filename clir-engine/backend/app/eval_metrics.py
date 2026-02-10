from __future__ import annotations

import math
from typing import Iterable, List


def precision_at_k(relevances: Iterable[int], k: int) -> float:
    rel = list(relevances)[:k]
    if not rel or k <= 0:
        return 0.0
    return sum(rel) / float(k)


def recall_at_k(relevances: Iterable[int], k: int, total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    rel = list(relevances)[:k]
    return sum(rel) / float(total_relevant)


def dcg_at_k(relevances: Iterable[int], k: int) -> float:
    rel = list(relevances)[:k]
    return sum((rel_i / math.log2(idx + 2)) for idx, rel_i in enumerate(rel))


def ndcg_at_k(relevances: Iterable[int], k: int) -> float:
    rel = list(relevances)[:k]
    ideal = sorted(rel, reverse=True)
    denom = dcg_at_k(ideal, k)
    if denom == 0:
        return 0.0
    return dcg_at_k(rel, k) / denom


def mrr(relevances: Iterable[int]) -> float:
    for idx, rel in enumerate(relevances, start=1):
        if rel:
            return 1.0 / idx
    return 0.0
