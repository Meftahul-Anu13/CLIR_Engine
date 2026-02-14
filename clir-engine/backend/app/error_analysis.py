from __future__ import annotations

from typing import Dict, List, Tuple

SUPPORTED_ERROR_TYPES = [
    "Translation Failures",
    "Named Entity Mismatch",
    "Semantic Gap",
    "Cross-Script Ambiguity",
    "Code-Switching",
]

_ERROR_ALIASES = {
    "translation failures": "Translation Failures",
    "translation failure": "Translation Failures",
    "translation": "Translation Failures",
    "named entity mismatch": "Named Entity Mismatch",
    "named-entity mismatch": "Named Entity Mismatch",
    "semantic gap": "Semantic Gap",
    "semantic gaps": "Semantic Gap",
    "cross-script ambiguity": "Cross-Script Ambiguity",
    "cross script ambiguity": "Cross-Script Ambiguity",
    "code-switching": "Code-Switching",
    "code switching": "Code-Switching",
}

for canonical in SUPPORTED_ERROR_TYPES:
    _ERROR_ALIASES.setdefault(canonical.lower(), canonical)


def normalize_error_type(value: str) -> str:
    """Normalize user-provided label to canonical error categories."""
    if not value:
        raise ValueError("error_type is required.")
    value_clean = value.strip().lower()
    if value_clean not in _ERROR_ALIASES:
        raise ValueError(
            f"Unsupported error_type '{value}'. "
            f"Expected one of: {', '.join(SUPPORTED_ERROR_TYPES)}"
        )
    return _ERROR_ALIASES[value_clean]


def _fractional_parts(counts: Dict[str, float]) -> List[Tuple[str, float]]:
    parts: List[Tuple[str, float]] = []
    for key, value in counts.items():
        frac = value - int(value)
        parts.append((key, frac))
    parts.sort(key=lambda item: (-item[1], item[0]))
    return parts


def stable_round_percentages(raw_counts: Dict[str, int]) -> Dict[str, int]:
    """Round percentages so they sum to 100 using largest remainder."""
    total = sum(raw_counts.values())
    if total == 0:
        return {k: 0 for k in raw_counts}

    raw_percentages = {
        key: (value / total) * 100.0 for key, value in raw_counts.items()
    }
    floored = {key: int(value) for key, value in raw_percentages.items()}
    remainder = 100 - sum(floored.values())
    if remainder <= 0:
        return floored

    for key, _ in _fractional_parts(raw_percentages):
        if remainder <= 0:
            break
        floored[key] += 1
        remainder -= 1

    return floored
