from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from starlette.middleware.cors import CORSMiddleware

from .colab_core.search import get_components, search as core_search

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
STORAGE_DIR = BASE_DIR / "storage"

dataset, index_store, _, _ = get_components()


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


@app.on_event("startup")
def _startup_warmup() -> None:
    get_components()


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
    debug: bool = Query(False, description="Return extra debug data"),
):
    query = (q or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")
    try:
        return core_search(query, lang=lang, k=k, debug=debug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/debug/index_integrity")
def debug_index_integrity():
    embeddings, faiss_index = index_store.ensure_ready()
    probes = []
    sample = min(3, dataset.size)
    for doc_id in range(sample):
        doc = dataset.get_document(doc_id)
        vec = embeddings[doc_id : doc_id + 1]
        _, idxs = faiss_index.search(vec, 1)
        probes.append(
            {
                "doc_id": doc_id,
                "title": doc.title,
                "url": doc.url,
                "language": doc.language,
                "faiss_top_id": int(idxs[0][0]),
            }
        )
    return {
        "documents_sampled": sample,
        "fingerprint": index_store.current_fingerprint(),
        "probes": probes,
    }
