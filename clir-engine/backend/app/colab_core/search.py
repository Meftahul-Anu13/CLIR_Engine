from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from .config_loader import get_config
from .dataset import DatasetStore
from .indexes import IndexStore
resultsfrom .query_pipeline import QueryProcessor
from .retrieval import RetrievalManager

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
STORAGE_DIR = BASE_DIR / "storage"

_COMPONENTS: Optional[
    Tuple[DatasetStore, IndexStore, QueryProcessor, RetrievalManager]
] = None


def _build_components():
    cfg = get_config()
    dataset = DatasetStore(DATA_DIR)
    index_store = IndexStore(dataset, STORAGE_DIR)
    index_store.ensure_ready()
    query_processor = QueryProcessor()
    retrieval_manager = RetrievalManager(
        dataset,
        index_store,
        lexical_topn=cfg["lexical_topn"],
        semantic_topn=cfg["semantic_topn"],
        weights=cfg["fusion_weights"],
    )
    return dataset, index_store, query_processor, retrieval_manager


def get_components():
    global _COMPONENTS
    if _COMPONENTS is None:
        _COMPONENTS = _build_components()
    return _COMPONENTS


def search(q: str, lang: str = "all", k: int = 10, debug: bool = False) -> Dict:
    query = (q or "").strip()
    if not query:
        raise ValueError("Query is required.")

    dataset, _, query_processor, retrieval_manager = get_components()

    search_start = time.time()
    bundle = query_processor.process(query)
    retrieval = retrieval_manager.search(
        bundle, lang, k, include_internal_ids=debug, debug=debug
    )
    total_ms = int((time.time() - search_start) * 1000)

    results = retrieval["results"]
    top_score = results[0]["score"] if results else 0.0
    warning = None
    threshold = 0.20
    if top_score < threshold:
        warning = {
            "threshold": threshold,
            "top_score": round(float(top_score), 4),
            "message": "Low confidence match. Try different keywords or a different language filter.",
        }

    timing_ms = {
        "total": total_ms,
        "translation": bundle["timing"]["translation_ms"],
        "lexical": retrieval["timing_ms"]["lexical"],
        "semantic": retrieval["timing_ms"]["semantic"],
        "fuzzy": retrieval["timing_ms"]["fuzzy"],
        "ranking": retrieval["timing_ms"]["ranking"],
    }

    response = {
        "query": query,
        "detected_lang": bundle["detected_lang"],
        "normalized": bundle["normalized"],
        "query_variants": bundle["query_variants"],
        "k": k,
        "language_filter": lang,
        "timing_ms": timing_ms,
        "warning": warning,
        "results": results,
    }
    if debug:
        response["debug"] = {
            "translation_error": bundle.get("translation_error"),
            "expansions": bundle.get("expansions"),
            "named_entities": bundle.get("named_entities"),
            "queries_used": retrieval.get("queries_used"),
        }
    return response
