# Hybrid Cross-Lingual Retriever
 Every query travels through the exact same preprocessing, multi-model retrieval, and fusion stages that were prototyped in `Query_pipeline_Module_B.ipynb`, `210041227_Retrieval_Module_C.ipynb`, and `210041227_Retrieval_evaluation_Module_D.ipynb`. This document summarizes that pipeline so you can reason about it without opening the notebooks.

---

## Pipeline Overview

1. **Query Processing (Module B alignment)**
   - Language detection via Bangla Unicode range.
   - Normalization + stopword trimming (English only).
   - Marian MT translations (`Helsinki-NLP/opus-mt-en-bn`, `bn-en`) plus named-entity injection (e.g., “tarek zia” → “তারেক রহমান”).
   - Query expansion tables add lightweight synonyms (“election” → “vote”, “polls”, …).
   - Output bundle contains both `query_variants` (`en`, `bn`), expansions, NE mappings, and timing.

2. **Hybrid Retrieval (Module C alignment)**
   - **BM25** (rank-bm25, unigram tokenization) on concatenated title+body text.
   - **TF-IDF** (scikit-learn, ngram_range=(1,2)) acting as an additional sparse signal.
   - **Char N-gram Fuzzy** (scikit-learn vectorizer + cosine) to handle transliteration/typos.
   - **Semantic** search using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (“LaBSE-style” dense vectors). Embeddings are precomputed once (`storage/doc_emb.npy`) and indexed with FAISS (`faiss.index`).
   - Candidate pooling: top-N hits from BM25/TF-IDF/semantic are unioned; fuzzy scoring is only run on this small set.

3. **Score Normalization + Fusion (Module D alignment)**
   - Each channel’s scores are min-max normalized to `[0,1]`.
   - Weighted sum fusion with notebook weights: **BM25 0.3**, **TF-IDF 0.2**, **Semantic 0.4**, **Fuzzy 0.1**.
   - Final ranked list is truncated to `k` and returned with per-channel debug scores.

---

## Key Features

### Multi-Modal Retrieval
- Can surface Bangla or English documents for either language of query thanks to translation + NE injection.
- Candidate pooling keeps semantic + fuzzy work bounded while still letting lexical models influence the final score.

### Efficiency Notes
- Document embeddings and FAISS index are computed once via `python scripts/build_indexes.py`; the backend only encodes the *query* at runtime.
- Fuzzy scoring runs on the union of the top lexical/semantic hits (default ≤100 docs), cutting latency to tens of milliseconds.
- Startup warmup calls the semantic search once to load the transformer model into memory.

### Debuggability
- `GET /search?...&debug=true` exposes the exact inputs/outputs per channel (query variants, expansions, `_doc_id`s, raw cosine scores).
- `GET /debug/index_integrity` reports fingerprint metadata so you can confirm FAISS ↔ dataset alignment before trusting results.

---

## Typical Latency (CPU)

| Stage                               | Latency (ms) |
|-------------------------------------|--------------|
| Query normalization + translation   | 1–20 (cached)|
| BM25 / TF-IDF scoring               | 30–50        |
| Semantic encode + FAISS top-N       | 50–80        |
| Fuzzy scoring on pooled candidates  | 20–40        |
| Score fusion + formatting           | < 2          |
| **End-to-end (k=10)**               | **~120–190** |

---

## Usage Examples

### Python (shared module)

```python
from app.colab_core.search import search

response = search("tarek zia", lang="all", k=10, debug=True)
for hit in response["results"]:
    print(hit["rank"], hit["language"], hit["score"], hit["title"])
```

### FastAPI endpoint

```
GET /search?q=tarek%20zia&lang=all&k=10&debug=true
```

Returns the same JSON structure as above (the API simply calls the shared `search()` helper).

---

## Dependencies

- `rank-bm25`, `scikit-learn`, `numpy`
- `sentence-transformers`, `faiss-cpu`, `torch`, `sentencepiece`, `sacremoses`
- `rapidfuzz` for fuzzy token-set scoring
- `transformers` (Marian MT)

---

## Sync with Colab

The notebooks and the backend load the same code paths:

- Notebooks import the modules in `app/colab_core/` directly, or you can export their results and run `python scripts/compare_with_colab.py --colab-results results.json` to verify overlap@k.
- When updating the dataset (`bangla_news.jsonl` / `english_news.jsonl`), delete the cached embeddings (`storage/doc_emb.npy`, `faiss.index`, `doc_fingerprint.json`) and rerun `python scripts/build_indexes.py` before comparing to Colab outputs.
