from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from starlette.middleware.cors import CORSMiddleware

from .dataset import DatasetStore
from .index_store import IndexStore
from .query import QueryProcessor
from .retrieval import RetrievalManager

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
STORAGE_DIR = BASE_DIR / "storage"

dataset = DatasetStore(DATA_DIR)
index_store = IndexStore(dataset, STORAGE_DIR)
index_store.ensure_ready()
query_processor = QueryProcessor(remove_stopwords=False)
retrieval_manager = RetrievalManager(dataset, index_store)


def _allowed_origins() -> list[str]:
    extra = os.getenv("BACKEND_ALLOWED_ORIGINS", "")
    origins = {"http://localhost:3000"}
    for origin in extra.split(","):
        origin = origin.strip()
        if origin:
            origins.add(origin)
    return sorted(origins)


app = FastAPI(title="CLIR Engine", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "documents": dataset.size}


@app.get("/meta")
def meta():
    return {
        "documents": {
            "total": dataset.size,
            "by_language": dataset.language_counts(),
        },
        "models": {
            "lexical": ["bm25", "tfidf"],
            "semantic": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "fuzzy": "rapidfuzz.token_set_ratio",
        },
    }


@app.get("/search")
def search(
    q: str = Query(..., description="User query"),
    lang: str = Query("all", pattern="^(all|en|bn)$", description="Language filter"),
    k: int = Query(10, ge=1, le=50, description="Number of results"),
):
    query = (q or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")

    search_start = time.time()
    bundle = query_processor.process(query)
    retrieval = retrieval_manager.search(bundle, lang, k)
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

    return {
        "query": query,
        "query_variants": bundle["query_variants"],
        "k": k,
        "language_filter": lang,
        "timing_ms": timing_ms,
        "warning": warning,
        "results": results,
    }
