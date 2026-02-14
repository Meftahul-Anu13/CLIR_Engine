# Time Analysis

The following measurements are taken from the shared Colab notebooks (`Query_pipeline_Module_B.ipynb`, `210041227_Retrieval_Module_C.ipynb`, `210041227_Retrieval_evaluation_Module_D.ipynb`) and the mirrored FastAPI codebase. They summarize the time spent in each major stage when serving a typical query (`k=10`, `lang=all`, cold-start warmup excluded).

### Table 1: Query Processing Latency

| Stage                               | Latency (ms) |
|-------------------------------------|--------------|
| Language detect + normalization     | ~0.5         |
| Marian translation (en↔bn, cached)  | 1–5          |
| Named-entity injection / expansions | <0.5         |
| **Subtotal**                        | **≈2–6**     |

### Table 2: Retrieval & Ranking Latency

| Stage                                | Latency (ms) | Notes                                        |
|--------------------------------------|--------------|----------------------------------------------|
| BM25 + TF-IDF scoring                | 30–50        | Scikit-learn vector ops over 6k docs         |
| Semantic encode + FAISS top-N        | 50–80        | Query encoding + ANN search                   |
| Fuzzy scoring (candidate pool)       | 20–40        | RapidFuzz on pooled top documents             |
| Score normalization + fusion         | <2           | Min-max + weighted sum                        |
| **Subtotal**                         | **≈100–170** |                                              |

### Table 3: Final System Totals

| Component            | Latency (ms) |
|----------------------|--------------|
| Query processing     | 2–6          |
| Retrieval + ranking  | 100–170      |
| **End-to-end**       | **≈120–190** |

---

### Efficiency Upgrade: The "Top-100 Re-ranking" Strategy

#### The Problem (Before Upgrade)
- Fuzzy / transliteration scoring ran across the entire corpus, which pushed total query latency well above a second (≈2.5 s in the earliest notebook runs).
- High recall but poor responsiveness; fuzzy matching dominated CPU time even when the dense/sparse scorers already narrowed candidates.

#### The Solution (The Upgrade)
1. **Candidate Union:** Gather the union of top BM25/TF-IDF/semantic hits (default ≤100 docs).
2. **Fuzzy on Candidates:** Run RapidFuzz + heuristic char n-grams only on that pool.
3. **Hybrid Fusion:** Min-max normalize each channel and combine scores with tuned weights (BM25 0.3, TF-IDF 0.2, Semantic 0.4, Fuzzy 0.1).

#### The Result
- Fuzzy scoring fell from ~2.5 s → <40 ms per query.
- Typical total latencies dropped to ≈140 ms (CPU), enabling real-time cross-lingual search with notebook-level accuracy.
- FastAPI and Colab now share the same modules (`app/colab_core`, FAISS cache), so timings measured in notebooks match production behavior.
