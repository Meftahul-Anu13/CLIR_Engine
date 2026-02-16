# Module D — Ranking, Scoring & Evaluation (Metrics + Error Analysis)


## Datasets used
This project uses a multilingual Bangla–English news dataset with a shared schema:

- `title`, `body`, `url`, `date`, `language` (required by the assignment)
- `text = title + body`



### Cell 1: pip -q install rank-bm25 scikit-learn sentence-transformers rapidfuzz
*Type:* `code`

**Purpose**

- Installs the Python libraries required for preprocessing, translation, retrieval, and evaluation in this notebook.

**Key snippet**

```python
!pip -q install rank-bm25 scikit-learn sentence-transformers rapidfuzz
```

### Cell 2: pip install -q spacy
*Type:* `code`

**Purpose**

- Installs the Python libraries required for preprocessing, translation, retrieval, and evaluation in this notebook.

**Key snippet**

```python
!pip install -q spacy
!python -m spacy download en_core_web_sm
!pip install -q transformers torch
!pip -q install sacremoses
```

### Cell 3: Loading the DB[link text](https://drive.google.com/drive/folders/14ybm
*Type:* `markdown`

**Context**

# Loading the DB[link text](https://drive.google.com/drive/folders/14ybm3SgzLffL7s5C1OYMSk5gm_gziMkg?usp=sharing)

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 4: pip -q install gdown
*Type:* `code`

**Purpose**

- Installs the Python libraries required for preprocessing, translation, retrieval, and evaluation in this notebook.
- Pulls processed dataset files into the runtime (often from Google Drive).

**Key snippet**

```python
!pip -q install gdown
```

### Cell 5: # !pip install deep-translator
*Type:* `code`

**Purpose**

- Installs the Python libraries required for preprocessing, translation, retrieval, and evaluation in this notebook.

**Key snippet**

```python
# !pip install deep-translator
```

### Cell 6: pip install FlagEmbedding
*Type:* `code`

**Purpose**

- Installs the Python libraries required for preprocessing, translation, retrieval, and evaluation in this notebook.

**Key snippet**

```python
!pip install FlagEmbedding
```

### Cell 7: import gdown
*Type:* `code`

**Purpose**

- Pulls processed dataset files into the runtime (often from Google Drive).

**Key snippet**

```python
import gdown
import os
DATA_DIR = "data/processed"
os.makedirs(DATA_DIR, exist_ok=True)
gdown.download_folder(
    url="https://drive.google.com/drive/folders/14ybm3SgzLffL7s5C1OYMSk5gm_gziMkg",
    output=DATA_DIR,
    quiet=False,
```

### Cell 8: import json
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
import json
import pandas as pd
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)
```

### Cell 9: def attach_docs(df_lang, pairs, lang, model_name):
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
def attach_docs(df_lang, pairs, lang, model_name):
    out = []
    for idx, score in pairs:
        row = df_lang.iloc[int(idx)]
        out.append({
            "model": model_name,
            "lang": lang,
            "score": float(score),
```

### Cell 10: from deep_translator import GoogleTranslator
*Type:* `code`

**Purpose**

- Translates queries between Bangla and English (OPUS-MT or Google Translate). This is the core CLIR bridge.

**Key snippet**

```python
from deep_translator import GoogleTranslator
def return_translated(keyword:str):
    try:
        if keyword[0].isascii():
            query_en = keyword
            query_bn = GoogleTranslator(source='auto', target='bn').translate(keyword)
        else:
            query_bn = keyword
```

**Gotchas / tips**

- Translation is a common failure source; log both original and translated queries for error analysis.
- Online translation can fail; the pipeline includes a fallback path so execution continues without crashing.

### Cell 11: qqq= return_translated("Bangladesh Election")
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
qqq= return_translated("Bangladesh Election")
qqq
```

### Cell 12: Model 1: Lexical Retrieval
*Type:* `markdown`

**Context**

# Model 1: Lexical Retrieval

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 13: import numpy as np
*Type:* `code`

**Purpose**

- Builds a TF‑IDF baseline to retrieve documents by lexical overlap in a sparse vector space.

**Key snippet**

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
class TFIDFRetriever:
    def __init__(self, texts):
        self.vectorizer = TfidfVectorizer(
            lowercase=False,
            ngram_range=(1,2),
            max_features=200000
```

### Cell 14: from rank_bm25 import BM25Okapi
*Type:* `code`

**Purpose**

- Implements Okapi BM25 over tokenized documents; a classic and competitive IR baseline.

**Key snippet**

```python
from rank_bm25 import BM25Okapi
def simple_tokenize(text: str):
    return (text or "").split()
class BM25Retriever:
    def __init__(self, texts):
        self.corpus_tokens = [simple_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.corpus_tokens)
    def search(self, query, topk=10):
```

### Cell 15: Fuzzy / Transliteration Matching
*Type:* `markdown`

**Context**

# Fuzzy / Transliteration Matching

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 16: from sklearn.feature_extraction.text import TfidfVectorizer
*Type:* `code`

**Purpose**

- Builds a TF‑IDF baseline to retrieve documents by lexical overlap in a sparse vector space.
- Uses character n‑gram TF‑IDF to match misspellings and cross-script variants (transliteration-friendly).

**Key snippet**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
class CharNgramFuzzyRetriever:
    def __init__(self, texts):
        self.vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2)
        self.X = self.vec.fit_transform(texts)
    def search(self, query, topk=10):
        qv = self.vec.transform([query])
        scores = (self.X @ qv.T).toarray().ravel()
```

### Cell 17: Semantic Matching
*Type:* `markdown`

**Context**

# Semantic Matching

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 18: # from sentence_transformers import SentenceTransformer
*Type:* `code`

**Purpose**

- Encodes queries/documents using a multilingual embedding model and retrieves by similarity.

**Key snippet**

```python
# from sentence_transformers import SentenceTransformer
# class SemanticRetriever:
#     def __init__(self, texts, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
#         self.model = SentenceTransformer(model_name)
#         self.emb = self.model.encode(
#             texts,
#             batch_size=64,
#             show_progress_bar=True,
```

**Gotchas / tips**

- Semantic embedding steps can be slow; cache document embeddings to disk and reuse across runs.
- FP16/float16 errors occur on CPU-only runtimes; fp16 is disabled and batch size is reduced.

### Cell 19: BGEM3FlagModel
*Type:* `markdown`

**Context**

# BGEM3FlagModel

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 20: from FlagEmbedding import BGEM3FlagModel
*Type:* `code`

**Purpose**

- Encodes queries/documents using a multilingual embedding model and retrieves by similarity.

**Key snippet**

```python
from FlagEmbedding import BGEM3FlagModel
import numpy as np
class SemanticRetriever:
    def __init__(self, texts, model_name='BAAI/bge-m3'):
        self.model = BGEM3FlagModel(model_name, use_fp16=True)
        embeddings = self.model.encode(
            texts,
            batch_size=12,
```

**Gotchas / tips**

- Semantic embedding steps can be slow; cache document embeddings to disk and reuse across runs.
- FP16/float16 errors occur on CPU-only runtimes; fp16 is disabled and batch size is reduced.

### Cell 21: Hybrid fusion (BM25 + Semantic + Fuzzy)
*Type:* `markdown`

**Context**

# Hybrid fusion (BM25 + Semantic + Fuzzy)

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 22: def minmax_norm(pairs):
*Type:* `code`

**Purpose**

- Normalizes scores to [0,1] and combines multiple signals (BM25 + semantic + fuzzy) via weighted fusion.

**Key snippet**

```python
def minmax_norm(pairs):
    # pairs: [(doc_idx, score)]
    if not pairs:
        return {}
    vals = np.array([s for _, s in pairs], dtype=float)
    mn, mx = float(vals.min()), float(vals.max())
    if mx - mn < 1e-9:
        return {i: 0.0 for i, _ in pairs}
```

### Cell 23: Module B testing
*Type:* `markdown`

**Context**

#Module B testing

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 24: # @title
*Type:* `code`

**Purpose**

- Detects whether a query is Bangla or English (usually via Unicode range checks).
- Translates queries between Bangla and English (OPUS-MT or Google Translate). This is the core CLIR bridge.

**Key snippet**

```python
# @title
import re
import time
from typing import Dict, List, Tuple, Optional
from transformers import MarianMTModel, MarianTokenizer
# -------------------------
# Language Detection
# -------------------------
```

**Gotchas / tips**

- Translation is a common failure source; log both original and translated queries for error analysis.
- Online translation can fail; the pipeline includes a fallback path so execution continues without crashing.

### Cell 25: Testing With B
*Type:* `markdown`

**Context**

#Testing With B

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 26: def run_models(bundle, topk=10, w_bm25=0.3, w_sem=0.5, w_fuzzy=0.2):
*Type:* `code`

**Purpose**

- Normalizes scores to [0,1] and combines multiple signals (BM25 + semantic + fuzzy) via weighted fusion.

**Key snippet**

```python
def run_models(bundle, topk=10, w_bm25=0.3, w_sem=0.5, w_fuzzy=0.2):
    q_en = bundle["q_en"]
    q_bn = bundle["q_bn"]
    results = []
    # --- English side
    bm25_en_hits = bm25_en.search(q_en, topk)
    tfidf_en_hits = tfidf_en.search(q_en, topk)
    fuzzy_en_hits = fuzzy_en.search(q_en, topk)
```

### Cell 27: query ="Tarek Zia"
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
query ="Tarek Zia"
processor = QueryProcessor(use_stopwords=False)
bundle = processor.process(query)
```

### Cell 28: df_res = run_models(bundle, topk=5, w_bm25=0.3, w_sem=0.5, w_fuzzy=0.2
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
df_res = run_models(bundle, topk=5, w_bm25=0.3, w_sem=0.5, w_fuzzy=0.2)
df_res
```

### Cell 29: def show_top_titles(df_res, topn=3):
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
def show_top_titles(df_res, topn=3):
    out = []
    for model in df_res["model"].unique():
        for lang in ["en","bn"]:
            sub = df_res[(df_res["model"]==model) & (df_res["lang"]==lang)].head(topn)
            titles = " | ".join(sub["title"].tolist())
            score = sub["score"].mean()
            out.append({"model": model, "lang": lang, "top_titles": titles,"score":score})
```

### Cell 30: All the result  tested by Google Translation in QueryProcessor
*Type:* `markdown`

**Context**

# All the result  tested by Google Translation in QueryProcessor

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 31: # -------------------------
*Type:* `code`

**Purpose**

- Translates queries between Bangla and English (OPUS-MT or Google Translate). This is the core CLIR bridge.

**Key snippet**

```python
# -------------------------
# Google Translation wrapper
# -------------------------
def google_translate(text: str, target: str) -> str:
    return GoogleTranslator(source="auto", target=target).translate(text)
```

**Gotchas / tips**

- Translation is a common failure source; log both original and translated queries for error analysis.
- Online translation can fail; the pipeline includes a fallback path so execution continues without crashing.

### Cell 32: q_norm="Bangladesh"
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
q_norm="Bangladesh"
google_translate(q_norm, target="bn")
```

### Cell 33: class QueryProcessorGoogle:
*Type:* `code`

**Purpose**

- Detects whether a query is Bangla or English (usually via Unicode range checks).

**Key snippet**

```python
class QueryProcessorGoogle:
    def __init__(self, use_stopwords: bool = False):
        self.use_stopwords = use_stopwords
    def process(self, raw_query: str) -> Dict:
        t0 = time.time()
        lang = detect_lang(raw_query)
        q_norm = normalize_query(raw_query, lang, self.use_stopwords)
        # expansions + NE mappings (based on normalized query)
```

### Cell 34: qp = QueryProcessorGoogle(use_stopwords=False)
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
qp = QueryProcessorGoogle(use_stopwords=False)
bundle = qp.process("Tarek Zia")
df_res = run_models(bundle, topk=5, w_bm25=0.3, w_sem=0.5, w_fuzzy=0.2)
df_res
```

### Cell 35: show_top_titles(df_res, topn=3)
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
show_top_titles(df_res, topn=3)
```

### Cell 36: Improved Version
*Type:* `markdown`

**Context**

# Improved Version

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 37: df_bn["title"][2996]
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
df_bn["title"][2996]
```

### Cell 38: df_en["title"][2996]
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
df_en["title"][2996]
```

### Cell 39: df = pd.concat([df_en, df_bn], ignore_index=True)
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
df = pd.concat([df_en, df_bn], ignore_index=True)
df
```

### Cell 40: # df["title"][2996]
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
# df["title"][2996]
df["title"][5996]
```

### Cell 41: from FlagEmbedding import BGEM3FlagModel
*Type:* `code`

**Purpose**

- Encodes queries/documents using a multilingual embedding model and retrieves by similarity.

**Key snippet**

```python
from FlagEmbedding import BGEM3FlagModel
import numpy as np
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
corpus_text = df['body'].tolist()
corpus_titles = df['title'].tolist()
embeddings = model.encode(
    corpus_text,
    batch_size=12,
```

**Gotchas / tips**

- Semantic embedding steps can be slow; cache document embeddings to disk and reuse across runs.
- FP16/float16 errors occur on CPU-only runtimes; fp16 is disabled and batch size is reduced.

### Cell 42: tokenized_corpus = [doc.split(" ") for doc in corpus_text]
*Type:* `code`

**Purpose**

- Implements Okapi BM25 over tokenized documents; a classic and competitive IR baseline.

**Key snippet**

```python
tokenized_corpus = [doc.split(" ") for doc in corpus_text]
bm25 = BM25Okapi(tokenized_corpus)
```

### Cell 43: def hybrid_search(query, top_k=10, alpha=0.7):
*Type:* `code`

**Purpose**

- Normalizes scores to [0,1] and combines multiple signals (BM25 + semantic + fuzzy) via weighted fusion.

**Key snippet**

```python
def hybrid_search(query, top_k=10, alpha=0.7):
    """
    alpha: Weight for Semantic Vector Search (0.0 to 1.0)
           1.0 = Pure Vector, 0.0 = Pure Keyword/Fuzzy
    """
    # --- A. SEMANTIC SEARCH (Dense) ---
    query_embedding = model.encode([query])['dense_vecs']
    # Calculate Cosine Similarity (Dot product for normalized vectors)
```

### Cell 44: def dual_translation_hybrid_search(query:str, alpha=0.7):
*Type:* `code`

**Purpose**

- Normalizes scores to [0,1] and combines multiple signals (BM25 + semantic + fuzzy) via weighted fusion.

**Key snippet**

```python
def dual_translation_hybrid_search(query:str, alpha=0.7):
    results = {}
    queries = return_translated(query)
    for q in queries:
        r = hybrid_search(q)
        results = results | r
    return results
```

### Cell 45: query = "Tarek Zia"
*Type:* `code`

**Purpose**

- Normalizes scores to [0,1] and combines multiple signals (BM25 + semantic + fuzzy) via weighted fusion.

**Key snippet**

```python
query = "Tarek Zia"
results = dual_translation_hybrid_search(query)
print(f"\nResults for '{query}':")
for res in results.values():
    print(f"[{res['type']}] {res['title']} ({res['score']:.4f})")
    # print(res["body"], "\n\n")
```

### Cell 46: query = "Bangladesh Election"
*Type:* `code`

**Purpose**

- Normalizes scores to [0,1] and combines multiple signals (BM25 + semantic + fuzzy) via weighted fusion.

**Key snippet**

```python
query = "Bangladesh Election"
results = dual_translation_hybrid_search(query)
data_list = []
for res in results.values():
    row = {
        'type': res['type'],
        'title': res['title'],
        'score': res['score'],
```

### Cell 47: k = 5
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
k = 5
top_k_df = df_sem.nlargest(k, 'score')
top_k_df
```

### Cell 48: Retrieval Evaluation
*Type:* `markdown`

**Context**

# Retrieval Evaluation
# Module D — Ranking, Scoring, & Evaluation (Core)

This cell provides narrative structure and records key links or section boundaries for reproducibility.This section adds **top‑K ranking with [0,1] matching confidence**, **query-time reporting**, and **IR evaluation metrics + labeling workflow**.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 49: import time, math
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
import time, math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
```

### Cell 50: D1) Ranking & Matching Score (0–1) + Low-confidence Warning
*Type:* `markdown`

**Context**

## D1) Ranking & Matching Score (0–1) + Low-confidence Warning

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 51: def _clip01(x: float) -> float:
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))
def rank_topk_for_query(
    raw_query: str,
    topk: int = 10,
    *,
    confidence_warn_threshold: float = 0.20,
    w_bm25: float = 0.3,
```

### Cell 52: D2) Query Execution Time Report
*Type:* `markdown`

**Context**

## D2) Query Execution Time Report

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 53: def pretty_time_report(r: Dict[str, Any]) -> None:
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
def pretty_time_report(r: Dict[str, Any]) -> None:
    q = r.get("query")
    tm = r.get("time_ms", {})
    print(f"Query: {q}")
    print("Time (ms):")
    for k in ["translation", "retrieval_total", "total"]:
        if tm.get(k) is not None:
            print(f"  - {k:16s}: {tm[k]}")
```

### Cell 54: D3) Evaluation Metrics (Precision@10, Recall@50, nDCG@10, MRR)
*Type:* `markdown`

**Context**

## D3) Evaluation Metrics (Precision@10, Recall@50, nDCG@10, MRR)

This cell provides narrative structure and records key links or section boundaries for reproducibility.### Labeling format (CSV)
Create a simple CSV with columns:
- `query`
- `doc_url`
- `language`
- `is_relevant` (yes/no)
- `annotator`

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 55: def make_labeling_sheet(
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
def make_labeling_sheet(
    queries: List[str],
    topn_per_query: int = 50,
    out_csv_path: str = "labels_template.csv",
) -> pd.DataFrame:
    rows = []
    for q in queries:
        r = rank_topk_for_query(q, topk=topn_per_query)
```

### Cell 56: def _bin_rel(x: Any) -> int:
*Type:* `code`

**Purpose**

- Computes Precision@K, Recall@K, nDCG@K, MRR (and related helpers) from relevance labels.

**Key snippet**

```python
def _bin_rel(x: Any) -> int:
    s = "" if pd.isna(x) else str(x).strip().lower()
    return 1 if s in {"1","true","yes","y"} else 0
def precision_at_k(rels: List[int], k: int) -> float:
    return float(sum(rels[:k])) / float(k) if k > 0 else 0.0
def recall_at_k(rels: List[int], total_relevant: int, k: int) -> float:
    return (float(sum(rels[:k])) / float(total_relevant)) if total_relevant > 0 else 0.0
def dcg_at_k(rels: List[int], k: int) -> float:
```

### Cell 57: Comparing with Google/Bing/DuckDuckGo (Required)
*Type:* `markdown`

**Context**

### Comparing with Google/Bing/DuckDuckGo (Required)

This cell provides narrative structure and records key links or section boundaries for reproducibility.Recommended simple workflow (manual):

1) For each of evaluation queries, search on Google/Bing/DDG.
2) Copy the **top 10 URLs** into a CSV (one file per engine) with columns: `query, doc_url, rank`.
3) Label those URLs using the same yes/no relevance labels.
4) Evaluate them with the helper below.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 58: def evaluate_external_engine(
*Type:* `code`

**Purpose**

- Computes Precision@K, Recall@K, nDCG@K, MRR (and related helpers) from relevance labels.

**Key snippet**

```python
def evaluate_external_engine(
    engine_ranked_csv: str,
    labels_csv_path: str,
    *,
    k: int = 10,
) -> pd.DataFrame:
    """Evaluate an external engine's ranking at k using your manual labels.
    engine_ranked_csv columns: query, doc_url, rank (1 = best)
```

### Cell 59: D4) Error Analysis Helpers (Case-study)
*Type:* `markdown`

**Context**

## D4) Error Analysis Helpers (Case-study)

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 60: def quick_case_study(raw_query: str, topk:int=10) -> None:
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
def quick_case_study(raw_query: str, topk:int=10) -> None:
    r = rank_topk_for_query(raw_query, topk=topk, return_all_models=True)
    print("Query:", r["query"])
    print("Detected:", r["bundle"].get("detected_lang"))
    print("Normalized:", r["bundle"].get("normalized"))
    print("q_en:", r["bundle"].get("q_en"))
    print("q_bn:", r["bundle"].get("q_bn"))
    pretty_time_report(r)
```

### Cell 61: EVAL_QUERIES = [
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
EVAL_QUERIES = [
    # Bangla (BN)
    "বাংলাদেশে মূল্যস্ফীতি কমেছে কি?",
    "ঢাকায় যানজট কমাতে নতুন পরিকল্পনা",
    "রোহিঙ্গা ক্যাম্পে স্বাস্থ্যসেবা পরিস্থিতি",
    "পদ্মা সেতু অর্থনৈতিক প্রভাব",
    "বাংলাদেশ বনাম ভারত সিরিজ সূচি",
    # English (EN)
```

### Cell 62: len(label_df)
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
len(label_df)
```

### Cell 63: label_df["query"][55]
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
label_df["query"][55]
```

### Cell 64: For short annotation purpose doing the less labeling
*Type:* `markdown`

**Context**

# For short annotation purpose doing the less labeling

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 65: label_df_2 = make_labeling_sheet(
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
label_df_2 = make_labeling_sheet(
    queries=EVAL_QUERIES,
    topn_per_query=3,
    out_csv_path=LABEL_SHEET_PATH
)
```

### Cell 66: label_df_2.head(10)
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
label_df_2.head(10)
```

### Cell 67: label_df_2["doc_url"]
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
label_df_2["doc_url"]
```

### Cell 68: FILLED_LABELS_PATH = "data/processed/labels_fiiled.csv"
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
FILLED_LABELS_PATH = "data/processed/labels_fiiled.csv"
metrics = evaluate_from_labels(FILLED_LABELS_PATH)
metrics
```

### Cell 69: per_query = evaluate_per_query(FILLED_LABELS_PATH)
*Type:* `code`

**Purpose**

- Computes Precision@K, Recall@K, nDCG@K, MRR (and related helpers) from relevance labels.

**Key snippet**

```python
per_query = evaluate_per_query(FILLED_LABELS_PATH)
per_query.sort_values("nDCG@10", ascending=False).head(20)
```

### Cell 70: # Pick failures (low score, wrong topic) + successes (semantic wins)
*Type:* `code`

**Purpose**

- Inspects retrieval failures (translation drift, NE mismatch, semantic vs lexical wins, cross-script ambiguity, code-switching).

**Key snippet**

```python
# Pick failures (low score, wrong topic) + successes (semantic wins)
quick_case_study("বাংলাদেশ বনাম ভারত সিরিজ সূচি")
quick_case_study("Bangla Desh vs বাংলাদেশ spelling in articles")  # optional stress test
quick_case_study("Dhaka air pollution AQI health impact")
```

