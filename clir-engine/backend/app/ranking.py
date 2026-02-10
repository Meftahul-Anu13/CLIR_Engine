from __future__ import annotations

from typing import Dict, Iterable


def normalize_channel(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    min_v = min(values)
    max_v = max(values)
    if max_v - min_v < 1e-9:
        return {doc_id: 0.0 for doc_id in scores}
    scale = max_v - min_v
    return {doc_id: max(0.0, min(1.0, (score - min_v) / scale)) for doc_id, score in scores.items()}


def fuse_scores(
    normalized_channels: Dict[str, Dict[int, float]],
    weights: Dict[str, float],
) -> Dict[int, float]:
    combined: Dict[int, float] = {}
    for channel, scores in normalized_channels.items():
        weight = weights.get(channel, 0.0)
        if weight <= 0 or not scores:
            continue
        for doc_id, score in scores.items():
            combined[doc_id] = combined.get(doc_id, 0.0) + weight * score
    # clamp to [0,1]
    for doc_id, score in list(combined.items()):
        combined[doc_id] = max(0.0, min(1.0, score))
    return combined
