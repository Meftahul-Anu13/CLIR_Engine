# Module D — Code Walkthrough (210041227_Retrieval_evaluation_Module_D.ipynb)

This file explains the notebook **cell-by-cell** with **what each part does**, **why it exists** in the CLIR pipeline, and **how to use it** for your assignment.

## Where this fits in the assignment


**Module D (Ranking, Scoring, Evaluation, Error Analysis)**  
This notebook extends Module C by adding:
- A final top‑K ranker with `matching_score ∈ [0,1]` and low-confidence warnings  
- Time breakdown: translation vs retrieval vs total  
- Labeling sheet generator (CSV) for manual relevance labels  
- Metrics computation: Precision@10, Recall@50, nDCG@10, MRR  
- Error-analysis helpers (case study printer)


## Datasets used in this Colab

The following datasets are used across the notebooks (Module A produces them; Module C/D consume them):

**JSON/JSONL (processed news corpora)**
- `bangla_news.jsonl` (≈3000 rows) — Bangla corpus
- `english_news.jsonl` / `english_news-1 (1).json` (≈2999 rows) — English corpus

**CSV sources (site-level corpora / crawled outputs)**
- `dailystar_final (3).csv` — The Daily Star (≈501 rows)
- `dhakapost_bangla (3).csv` — Dhaka Post (Bangla) (≈397 rows)
- `banglatribune_bangla (2).csv` — Bangla Tribune (≈100 rows)
- `dhakapost_full_news (2).csv` — Dhaka Post full news (≈12 rows)

These files follow the assignment’s **minimum metadata** idea:
`title`, `body`, `url`, `date`, `language` (+ optional `tokens`, `source`, `date_method`).

## Notes / known issues to watch

- **Same TF‑IDF stray backtick bug** as Module C (remove it).
- `evaluate_per_query(...)` is called but **not defined**.  
  ✅ Fix: use `dfq, macro = evaluate_from_labels(...)` and treat `dfq` as your per-query table (or define a wrapper).
- Ensure your filled labels file path is correct: the notebook uses `data/processed/labels_fiiled.csv` (note the spelling).

## Cell-by-cell explanation


---

### Code cell 1

```python
01 | !pip -q install rank-bm25 scikit-learn sentence-transformers rapidfuzz
```

**Line-by-line explanation**

- **L1:** Colab shell command to install Python packages needed for this module.


---

### Code cell 2

```python
01 | !pip install -q spacy
02 | !python -m spacy download en_core_web_sm
03 | !pip install -q transformers torch
04 | !pip -q install sacremoses
```

**Line-by-line explanation**

- **L1:** Colab shell command to install Python packages needed for this module.
- **L2:** Colab shell command to run a Python module (here used to download spaCy models).
- **L3:** Colab shell command to install Python packages needed for this module.
- **L4:** Colab shell command to install Python packages needed for this module.


---

### Markdown cell 3

# Loading the DB[link text](https://drive.google.com/drive/folders/14ybm3SgzLffL7s5C1OYMSk5gm_gziMkg?usp=sharing)


---

### Code cell 4

```python
01 | !pip -q install gdown
```

**Line-by-line explanation**

- **L1:** Colab shell command to install Python packages needed for this module.


---

### Code cell 5

```python
01 | # !pip install deep-translator
```

**Line-by-line explanation**

- **L1:** Comment: !pip install deep-translator


---

### Code cell 6

```python
01 | !pip install FlagEmbedding
```

**Line-by-line explanation**

- **L1:** Colab shell command to install Python packages needed for this module.


---

### Code cell 7

```python
01 | import gdown
02 | import os
03 | 
04 | DATA_DIR = "data/processed"
05 | os.makedirs(DATA_DIR, exist_ok=True)
06 | 
07 | gdown.download_folder(
08 |     url="https://drive.google.com/drive/folders/14ybm3SgzLffL7s5C1OYMSk5gm_gziMkg",
09 |     output=DATA_DIR,
10 |     quiet=False,
11 |     use_cookies=False
12 | )
```

**Line-by-line explanation**

- **L1:** Imports `gdown`.
- **L2:** Imports `os` for filesystem utilities (paths, folders, environment).
- **L4:** Assigns a value to `DATA_DIR`.
- **L5:** Creates the target folder if it doesn't exist (`exist_ok=True` avoids crashing if it already exists).
- **L7:** Downloads the dataset folder from Google Drive into the local `data/processed` directory.
- **L8:** Assigns a value to `url`.
- **L9:** Assigns a value to `output`.
- **L10:** Assigns a value to `quiet`.
- **L11:** Assigns a value to `use_cookies`.
- **L12:** Executes this statement as part of the pipeline.


---

### Code cell 8

```python
01 | import json
02 | import pandas as pd
03 | 
04 | def load_jsonl(path):
05 |     rows = []
06 |     with open(path, "r", encoding="utf-8") as f:
07 |         for line in f:
08 |             rows.append(json.loads(line))
09 |     return pd.DataFrame(rows)
10 | 
11 | df_bn = load_jsonl("data/processed/bangla_news.jsonl")
12 | df_en = load_jsonl("data/processed/english_news.jsonl")
13 | 
14 | # build text field (Module A → Module C handoff)
15 | df_bn["text"] = (df_bn["title"].fillna("") + " " + df_bn["body"].fillna("")).str.strip()
16 | df_en["text"] = (df_en["title"].fillna("") + " " + df_en["body"].fillna("")).str.strip()
17 | 
18 | df_bn.head(2)
19 | df_en.head(2)
```

**Line-by-line explanation**

- **L1:** Imports `json` for reading/writing JSON & JSONL datasets.
- **L2:** Imports `pandas` for tabular data handling with DataFrames (filtering, saving).
- **L4:** Defines `load_jsonl()` (a reusable function for the pipeline).
- **L5:** Assigns a value to `rows`.
- **L6:** Assigns a value to `with open(path, "r", encoding`.
- **L7:** Starts a loop to process items one-by-one.
- **L8:** Executes this statement as part of the pipeline.
- **L9:** Returns the function output back to the caller.
- **L11:** Assigns a value to `df_bn`.
- **L12:** Assigns a value to `df_en`.
- **L14:** Comment: build text field (Module A → Module C handoff)
- **L15:** Assigns a value to `df_bn["text"]`.
- **L16:** Assigns a value to `df_en["text"]`.
- **L18:** Executes this statement as part of the pipeline.
- **L19:** Executes this statement as part of the pipeline.


---

### Code cell 9

```python
01 | def attach_docs(df_lang, pairs, lang, model_name):
02 |     out = []
03 |     for idx, score in pairs:
04 |         row = df_lang.iloc[int(idx)]
05 |         out.append({
06 |             "model": model_name,
07 |             "lang": lang,
08 |             "score": float(score),
09 |             "title": row.get("title",""),
10 |             "url": row.get("url",""),
11 |             "date": row.get("date","")
12 |         })
13 |     return out
```

**Line-by-line explanation**

- **L1:** Defines `attach_docs()` (a reusable function for the pipeline).
- **L2:** Assigns a value to `out`.
- **L3:** Starts a loop to process items one-by-one.
- **L4:** Assigns a value to `row`.
- **L5:** Executes this statement as part of the pipeline.
- **L6:** Executes this statement as part of the pipeline.
- **L7:** Executes this statement as part of the pipeline.
- **L8:** Executes this statement as part of the pipeline.
- **L9:** Executes this statement as part of the pipeline.
- **L10:** Executes this statement as part of the pipeline.
- **L11:** Executes this statement as part of the pipeline.
- **L12:** Executes this statement as part of the pipeline.
- **L13:** Returns the function output back to the caller.


---

### Code cell 10

```python
01 | from deep_translator import GoogleTranslator
02 | 
03 | def return_translated(keyword:str):
04 |     try:
05 |         if keyword[0].isascii():
06 |             query_en = keyword
07 |             query_bn = GoogleTranslator(source='auto', target='bn').translate(keyword)
08 |         else:
09 |             query_bn = keyword
10 |             query_en = GoogleTranslator(source='auto', target='en').translate(keyword)
11 |         return [query_en, query_bn]
12 |     except Exception as e:
13 |         # Fallback if internet fails
14 |         print(e)
15 |         query_en = keyword
16 |         return [query_en]
```

**Line-by-line explanation**

- **L1:** Imports `GoogleTranslator` from `deep_translator`.
- **L3:** Defines `return_translated()` (a reusable function for the pipeline).
- **L4:** Executes this statement as part of the pipeline.
- **L5:** Conditional branch: only runs the next indented block if the condition is true.
- **L6:** Assigns a value to `query_en`.
- **L7:** Assigns a value to `query_bn`.
- **L8:** Fallback branch if none of the previous conditions matched.
- **L9:** Assigns a value to `query_bn`.
- **L10:** Assigns a value to `query_en`.
- **L11:** Returns the function output back to the caller.
- **L12:** Executes this statement as part of the pipeline.
- **L13:** Comment: Fallback if internet fails
- **L14:** Executes this statement as part of the pipeline.
- **L15:** Assigns a value to `query_en`.
- **L16:** Returns the function output back to the caller.


---

### Code cell 11

```python
01 | qqq= return_translated("Bangladesh Election")
02 | qqq
```

**Line-by-line explanation**

- **L1:** Assigns a value to `qqq`.
- **L2:** Executes this statement as part of the pipeline.


---

### Markdown cell 12

# Model 1: Lexical Retrieval


---

### Code cell 13

```python
01 | import numpy as np
02 | from sklearn.feature_extraction.text import TfidfVectorizer
03 | 
04 | class TFIDFRetriever:
05 |     def __init__(self, texts):
06 |         self.vectorizer = TfidfVectorizer(
07 |             lowercase=False,
08 |             ngram_range=(1,2),
09 |             max_features=200000
10 |         )
11 |         self.X = self.vectorizer.fit_transform(texts)
12 | 
13 |     def search(self, query, topk=10):
14 |         qv = self.vectorizer.transform([query])
15 |         scores = (self.X @ qv.T).toarray().ravel()
16 |         idx = np.argsort(-scores)[:topk]`
17 |         return [(int(i), float(scores[i])) for i in idx]
18 | 
19 | tfidf_en = TFIDFRetriever(df_en["text"].tolist())
20 | tfidf_bn = TFIDFRetriever(df_bn["text"].tolist())
```

**Line-by-line explanation**

- **L1:** Imports `numpy` for fast numeric arrays (scoring, sorting top-k).
- **L2:** Imports `TfidfVectorizer` from `sklearn.feature_extraction.text` for TF-IDF vectorization and other utilities.
- **L4:** Defines class `TFIDFRetriever` — A TF‑IDF baseline retriever. Builds a vector space model and ranks by dot product similarity.
- **L5:** Defines `__init__()` (a reusable function for the pipeline).
- **L6:** Initializes TF‑IDF vectorizer (controls n‑grams/features for lexical matching).
- **L7:** Assigns a value to `lowercase`.
- **L8:** Assigns a value to `ngram_range`.
- **L9:** Assigns a value to `max_features`.
- **L10:** Executes this statement as part of the pipeline.
- **L11:** Assigns a value to `self.X`.
- **L13:** Defines `search()` (a reusable function for the pipeline).
- **L14:** Assigns a value to `qv`.
- **L15:** Assigns a value to `scores`.
- **L16:** Sorts document scores in descending order and keeps the top‑K indices (ranking step).
- **L17:** Returns the function output back to the caller.
- **L19:** Assigns a value to `tfidf_en`.
- **L20:** Assigns a value to `tfidf_bn`.


---

### Code cell 14

```python
01 | from rank_bm25 import BM25Okapi
02 | 
03 | def simple_tokenize(text: str):
04 |     return (text or "").split()
05 | 
06 | class BM25Retriever:
07 |     def __init__(self, texts):
08 |         self.corpus_tokens = [simple_tokenize(t) for t in texts]
09 |         self.bm25 = BM25Okapi(self.corpus_tokens)
10 | 
11 |     def search(self, query, topk=10):
12 |         q_tokens = simple_tokenize(query)
13 |         scores = self.bm25.get_scores(q_tokens)
14 |         idx = np.argsort(-scores)[:topk]
15 |         return [(int(i), float(scores[i])) for i in idx]
16 | 
17 | bm25_en = BM25Retriever(df_en["text"].tolist())
18 | bm25_bn = BM25Retriever(df_bn["text"].tolist())
```

**Line-by-line explanation**

- **L1:** Imports `BM25Okapi` from `rank_bm25` for BM25 baseline retriever (lexical scoring).
- **L3:** Defines `simple_tokenize()` (a reusable function for the pipeline).
- **L4:** Returns the function output back to the caller.
- **L6:** Defines class `BM25Retriever` — A BM25 baseline retriever. Works well for exact-term matches and is the standard lexical IR baseline.
- **L7:** Defines `__init__()` (a reusable function for the pipeline).
- **L8:** Assigns a value to `self.corpus_tokens`.
- **L9:** Builds a BM25 index over tokenized documents for lexical ranking.
- **L11:** Defines `search()` (a reusable function for the pipeline).
- **L12:** Assigns a value to `q_tokens`.
- **L13:** Assigns a value to `scores`.
- **L14:** Sorts document scores in descending order and keeps the top‑K indices (ranking step).
- **L15:** Returns the function output back to the caller.
- **L17:** Assigns a value to `bm25_en`.
- **L18:** Assigns a value to `bm25_bn`.


---

### Markdown cell 15

# Fuzzy / Transliteration Matching


---

### Code cell 16

```python
01 | from sklearn.feature_extraction.text import TfidfVectorizer
02 | 
03 | class CharNgramFuzzyRetriever:
04 |     def __init__(self, texts):
05 |         self.vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2)
06 |         self.X = self.vec.fit_transform(texts)
07 | 
08 |     def search(self, query, topk=10):
09 |         qv = self.vec.transform([query])
10 |         scores = (self.X @ qv.T).toarray().ravel()
11 |         idx = np.argsort(-scores)[:topk]
12 |         return [(int(i), float(scores[i])) for i in idx]
13 | 
14 | fuzzy_en = CharNgramFuzzyRetriever(df_en["text"].tolist())
15 | fuzzy_bn = CharNgramFuzzyRetriever(df_bn["text"].tolist())
```

**Line-by-line explanation**

- **L1:** Imports `TfidfVectorizer` from `sklearn.feature_extraction.text` for TF-IDF vectorization and other utilities.
- **L3:** Defines class `CharNgramFuzzyRetriever` — Character n‑gram TF‑IDF retriever. Helps with typos, transliteration variants, and partial matches.
- **L4:** Defines `__init__()` (a reusable function for the pipeline).
- **L5:** Initializes TF‑IDF vectorizer (controls n‑grams/features for lexical matching).
- **L6:** Assigns a value to `self.X`.
- **L8:** Defines `search()` (a reusable function for the pipeline).
- **L9:** Assigns a value to `qv`.
- **L10:** Assigns a value to `scores`.
- **L11:** Sorts document scores in descending order and keeps the top‑K indices (ranking step).
- **L12:** Returns the function output back to the caller.
- **L14:** Assigns a value to `fuzzy_en`.
- **L15:** Assigns a value to `fuzzy_bn`.


---

### Markdown cell 17

# Semantic Matching


---

### Code cell 18

```python
01 | # from sentence_transformers import SentenceTransformer
02 | 
03 | # class SemanticRetriever:
04 | #     def __init__(self, texts, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
05 | #         self.model = SentenceTransformer(model_name)
06 | #         self.emb = self.model.encode(
07 | #             texts,
08 | #             batch_size=64,
09 | #             show_progress_bar=True,
10 | #             normalize_embeddings=True
11 | #         )
12 | 
13 | #     def search(self, query, topk=10):
14 | #         q = self.model.encode([query], normalize_embeddings=True)[0]
15 | #         scores = self.emb @ q   # cosine because normalized
16 | #         idx = np.argsort(-scores)[:topk]
17 | #         return [(int(i), float(scores[i])) for i in idx]
18 | 
19 | # sem_en = SemanticRetriever(df_en["text"].tolist())
20 | # sem_bn = SemanticRetriever(df_bn["text"].tolist())
```

**Line-by-line explanation**

- **L1:** Comment: from sentence_transformers import SentenceTransformer
- **L3:** Comment: class SemanticRetriever:
- **L4:** Comment: def __init__(self, texts, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
- **L5:** Comment: self.model = SentenceTransformer(model_name)
- **L6:** Comment: self.emb = self.model.encode(
- **L7:** Comment: texts,
- **L8:** Comment: batch_size=64,
- **L9:** Comment: show_progress_bar=True,
- **L10:** Comment: normalize_embeddings=True
- **L11:** Comment: )
- **L13:** Comment: def search(self, query, topk=10):
- **L14:** Comment: q = self.model.encode([query], normalize_embeddings=True)[0]
- **L15:** Comment: scores = self.emb @ q   # cosine because normalized
- **L16:** Comment: idx = np.argsort(-scores)[:topk]
- **L17:** Comment: return [(int(i), float(scores[i])) for i in idx]
- **L19:** Comment: sem_en = SemanticRetriever(df_en["text"].tolist())
- **L20:** Comment: sem_bn = SemanticRetriever(df_bn["text"].tolist())


---

### Markdown cell 19

# BGEM3FlagModel


---

### Code cell 20

```python
01 | from FlagEmbedding import BGEM3FlagModel
02 | import numpy as np
03 | 
04 | class SemanticRetriever:
05 |     def __init__(self, texts, model_name='BAAI/bge-m3'):
06 | 
07 |         self.model = BGEM3FlagModel(model_name, use_fp16=True)
08 | 
09 |         embeddings = self.model.encode(
10 |             texts,
11 |             batch_size=12, 
12 |             max_length=512
13 |         )
14 |         self.emb = embeddings['dense_vecs']
15 | 
16 |     def search(self, query, topk=10):
17 | 
18 |         q_result = self.model.encode([query], max_length=512)
19 |         q = q_result['dense_vecs'][0]
20 | 
21 |         scores = self.emb @ q
22 |         idx = np.argsort(-scores)[:topk]
23 |         return [(int(i), float(scores[i])) for i in idx]
24 | 
25 | # Initialize with your dataframes
26 | sem_en = SemanticRetriever(df_en["text"].tolist())
27 | sem_bn = SemanticRetriever(df_bn["text"].tolist())
```

**Line-by-line explanation**

- **L1:** Imports `BGEM3FlagModel` from `FlagEmbedding` for BGE embedding models (semantic retrieval).
- **L2:** Imports `numpy` for fast numeric arrays (scoring, sorting top-k).
- **L4:** Defines class `SemanticRetriever` — Dense embedding retriever. Encodes documents and queries into vectors and ranks by similarity (best for paraphrases / semantic matches).
- **L5:** Defines `__init__()` (a reusable function for the pipeline).
- **L7:** Loads the BGE‑M3 embedding model (dense semantic retrieval).
- **L9:** Assigns a value to `embeddings`.
- **L10:** Executes this statement as part of the pipeline.
- **L11:** Assigns a value to `batch_size`.
- **L12:** Assigns a value to `max_length`.
- **L13:** Executes this statement as part of the pipeline.
- **L14:** Assigns a value to `self.emb`.
- **L16:** Defines `search()` (a reusable function for the pipeline).
- **L18:** Assigns a value to `q_result`.
- **L19:** Assigns a value to `q`.
- **L21:** Assigns a value to `scores`.
- **L22:** Sorts document scores in descending order and keeps the top‑K indices (ranking step).
- **L23:** Returns the function output back to the caller.
- **L25:** Comment: Initialize with your dataframes
- **L26:** Assigns a value to `sem_en`.
- **L27:** Assigns a value to `sem_bn`.


---

### Markdown cell 21

# Hybrid fusion (BM25 + Semantic + Fuzzy)


---

### Code cell 22

```python
01 | def minmax_norm(pairs):
02 |     # pairs: [(doc_idx, score)]
03 |     if not pairs:
04 |         return {}
05 |     vals = np.array([s for _, s in pairs], dtype=float)
06 |     mn, mx = float(vals.min()), float(vals.max())
07 |     if mx - mn < 1e-9:
08 |         return {i: 0.0 for i, _ in pairs}
09 |     return {i: (s - mn) / (mx - mn) for i, s in pairs}
10 | 
11 | def hybrid_fuse(bm25_pairs, sem_pairs, fuzzy_pairs, w_bm25=0.3, w_sem=0.5, w_fuzzy=0.2, topk=10):
12 |     bm = minmax_norm(bm25_pairs)
13 |     se = minmax_norm(sem_pairs)
14 |     fu = minmax_norm(fuzzy_pairs)
15 | 
16 |     all_ids = set(bm) | set(se) | set(fu)
17 |     fused = []
18 |     for i in all_ids:
19 |         score = w_bm25 * bm.get(i, 0.0) + w_sem * se.get(i, 0.0) + w_fuzzy * fu.get(i, 0.0)
20 |         fused.append((int(i), float(score)))
21 | 
22 |     fused.sort(key=lambda x: x[1], reverse=True)
23 |     return fused[:topk]
```

**Line-by-line explanation**

- **L1:** Defines `minmax_norm()` (a reusable function for the pipeline).
- **L2:** Comment: pairs: [(doc_idx, score)]
- **L3:** Conditional branch: only runs the next indented block if the condition is true.
- **L4:** Returns the function output back to the caller.
- **L5:** Assigns a value to `vals`.
- **L6:** Assigns a value to `mn, mx`.
- **L7:** Conditional branch: only runs the next indented block if the condition is true.
- **L8:** Returns the function output back to the caller.
- **L9:** Returns the function output back to the caller.
- **L11:** Defines `hybrid_fuse()` — Weighted fusion of normalized model scores (BM25 + semantic + fuzzy) to get a robust hybrid ranker.
- **L12:** Assigns a value to `bm`.
- **L13:** Assigns a value to `se`.
- **L14:** Assigns a value to `fu`.
- **L16:** Assigns a value to `all_ids`.
- **L17:** Assigns a value to `fused`.
- **L18:** Starts a loop to process items one-by-one.
- **L19:** Assigns a value to `score`.
- **L20:** Executes this statement as part of the pipeline.
- **L22:** Assigns a value to `fused.sort(key`.
- **L23:** Returns the function output back to the caller.


---

### Markdown cell 23

#Module B testing


---

### Code cell 24

```python
001 | # @title
002 | import re
003 | import time
004 | from typing import Dict, List, Tuple, Optional
005 | from transformers import MarianMTModel, MarianTokenizer
006 | 
007 | # -------------------------
008 | # Language Detection
009 | # -------------------------
010 | BN_RANGE = re.compile(r"[\u0980-\u09FF]")
011 | 
012 | def detect_lang(query: str) -> str:
013 |     """Detect if query is Bangla or English"""
014 |     return "bn" if BN_RANGE.search(query or "") else "en"
015 | 
016 | 
017 | # -------------------------
018 | # Stopwords (Optional but Recommended)
019 | # -------------------------
020 | STOPWORDS_EN = {
021 |     "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
022 |     "to", "for", "of", "and", "or", "but", "with", "from", "by"
023 | }
024 | 
025 | STOPWORDS_BN = {
026 |     "এবং", "বা", "কিন্তু", "যা", "যে", "এই", "সেই", "ও", "তে", "থেকে",
027 |     "দিয়ে", "হয়", "হয়েছে", "করা", "করে", "একটি", "কোনো", "আর"
028 | }
029 | 
030 | def remove_stopwords(query: str, lang: str) -> str:
031 |     """Remove stopwords (optional preprocessing)"""
032 |     stopwords = STOPWORDS_BN if lang == "bn" else STOPWORDS_EN
033 |     words = query.split()
034 |     filtered = [w for w in words if w.lower() not in stopwords]
035 |     return " ".join(filtered) if filtered else query  # fallback to original if all removed
036 | 
037 | 
038 | # -------------------------
039 | # Normalization
040 | # -------------------------
041 | def normalize_query(query: str, lang: str, remove_stops: bool = False) -> str:
042 |     """Normalize query: trim, lowercase (EN only), optional stopword removal"""
043 |     q = " ".join((query or "").strip().split())
044 |     if lang == "en":
045 |         q = q.lower()
046 |     if remove_stops:
047 |         q = remove_stopwords(q, lang)
048 |     return q
049 | 
050 | 
051 | # -------------------------
052 | # Translation (OPUS-MT)
053 | # -------------------------
054 | class TranslatorOPUS:
055 |     """
056 |     OPUS-MT translator with caching
057 |     - bn → en: Helsinki-NLP/opus-mt-bn-en
058 |     - en → bn: Helsinki-NLP/opus-mt-en-iir with >>ben<< token
059 |     """
060 |     def __init__(self, max_length: int = 64):
061 |         self.cache = {}
062 |         self.max_length = max_length
063 | 
064 |     def _load(self, model_name: str) -> Tuple[MarianTokenizer, MarianMTModel]:
065 |         if model_name not in self.cache:
066 |             tok = MarianTokenizer.from_pretrained(model_name)
067 |             mdl = MarianMTModel.from_pretrained(model_name)
068 |             self.cache[model_name] = (tok, mdl)
069 |         return self.cache[model_name]
070 | 
071 |     def _generate(self, model_name: str, text: str) -> str:
072 |         tok, mdl = self._load(model_name)
073 |         batch = tok([text], return_tensors="pt", truncation=True)
074 |         gen = mdl.generate(**batch, max_length=self.max_length)
075 |         return tok.decode(gen[0], skip_special_tokens=True).strip()
076 | 
077 |     def translate_bn_to_en(self, text: str) -> str:
078 |         return self._generate("Helsinki-NLP/opus-mt-bn-en", text)
079 | 
080 |     def translate_en_to_bn(self, text: str) -> str:
081 |         text_with_lang = f">>ben<< {text}"
082 |         out = self._generate("Helsinki-NLP/opus-mt-en-iir", text_with_lang)
083 |         return out.replace(">>ben<<", "").strip()
084 | 
085 |     def translate(self, text: str, src: str, tgt: str) -> str:
086 |         if src == "bn" and tgt == "en":
087 |             return self.translate_bn_to_en(text)
088 |         if src == "en" and tgt == "bn":
089 |             return self.translate_en_to_bn(text)
090 |         raise ValueError(f"Unsupported translation: {src}→{tgt}")
091 | 
092 | 
093 | # -------------------------
094 | # Query Expansion - IMPROVED
095 | # -------------------------
096 | EXPAND_EN = {
097 |     # Politics
098 |     "election": ["vote", "poll", "ballot", "voting"],
099 |     "government": ["administration", "authority", "regime"],
100 |     "minister": ["secretary", "official"],
101 |     "parliament": ["legislature", "assembly"],
102 | 
103 |     # Education
104 |     "education": ["school", "university", "learning", "teaching"],
105 |     "student": ["pupil", "learner"],
106 |     "teacher": ["instructor", "educator"],
107 | 
108 |     # Economy
109 |     "economy": ["finance", "inflation", "market", "trade"],
110 |     "business": ["commerce", "trade", "industry"],
111 |     "price": ["cost", "rate", "tariff"],
112 | 
113 |     # Sports
114 |     "cricket": ["test", "odi", "t20"],
115 |     "football": ["soccer", "match"],
116 |     "player": ["athlete", "sportsman"],
117 | 
118 |     # Common
119 |     "news": ["report", "article", "story"],
120 |     "death": ["died", "killed", "deceased"],
121 |     "attack": ["assault", "strike"],
122 | }
123 | 
124 | EXPAND_BN = {
125 |     "নির্বাচন": ["ভোট", "পোল", "নির্বাচনী"],
126 |     "সরকার": ["প্রশাসন", "কর্তৃপক্ষ"],
127 |     "শিক্ষা": ["স্কুল", "বিশ্ববিদ্যালয়", "পড়াশোনা"],
128 |     "অর্থনীতি": ["ফিন্যান্স", "বাজার"],
129 |     "ক্রিকেট": ["টেস্ট", "ওডিআই", "টি-টুয়েন্টি"],
130 | }
131 | 
132 | def expand_query(q: str, lang: str) -> List[str]:
133 |     """Expand query with synonyms and related terms"""
134 |     expand_dict = EXPAND_BN if lang == "bn" else EXPAND_EN
135 | 
136 |     terms = (q or "").lower().split() if lang == "en" else (q or "").split()
137 |     expanded = []
138 | 
139 |     for term in terms:
140 |         if term in expand_dict:
141 |             expanded.extend(expand_dict[term])
142 | 
143 |     # Remove duplicates while preserving order
144 |     return list(dict.fromkeys(expanded))
145 | 
146 | 
147 | # -------------------------
148 | # Named Entity Mapping - IMPROVED
149 | # -------------------------
150 | NE_EN2BN = {
151 |     # Places
152 |     "bangladesh": "বাংলাদেশ",
153 |     "dhaka": "ঢাকা",
154 |     "chittagong": "চট্টগ্রাম",
155 |     "sylhet": "সিলেট",
156 |     "rajshahi": "রাজশাহী",
157 |     "khulna": "খুলনা",
158 |     "cox's bazar": "কক্সবাজার",
159 | 
160 |     # Political Figures
161 |     "sheikh hasina": "শেখ হাসিনা",
162 |     "khaleda zia": "খালেদা জিয়া",
163 |     "muhammad yunus": "মুহাম্মদ ইউনূস",
164 | 
165 |     # Organizations
166 |     "awami league": "আওয়ামী লীগ",
167 |     "bnp": "বিএনপি",
168 |     "jamaat": "জামায়াত",
169 | 
170 |     # Institutions
171 |     "dhaka university": "ঢাকা বিশ্ববিদ্যালয়",
172 |     "buet": "বুয়েট",
173 | 
174 |     "tarique rahman": "তারেক রহমান",
175 |     "tarek rahman": "তারেক রহমান",
176 |     "tarek zia": "তারেক রহমান",
177 | }
178 | 
179 | NE_BN2EN = {v: k for k, v in NE_EN2BN.items()}
180 | 
181 | def map_named_entities(q: str, lang: str) -> List[Dict[str, str]]:
182 |     """Map named entities between languages"""
183 |     mapped = []
184 |     q_normalized = q.lower() if lang == "en" else q
185 | 
186 |     if lang == "en":
187 |         for en_term, bn_term in NE_EN2BN.items():
188 |             if en_term in q_normalized:
189 |                 mapped.append({"src": en_term, "tgt": bn_term})
190 |     else:
191 |         for bn_term, en_term in NE_BN2EN.items():
192 |             if bn_term in q_normalized:
193 |                 mapped.append({"src": bn_term, "tgt": en_term})
194 | 
195 |     return mapped
196 | 
197 | 
198 | # -------------------------
199 | # Translation Post-processing
200 | # -------------------------
201 | def postprocess_translation(q_original: str, q_translated: str, src_lang: str) -> str:
202 |     """Fix common translation errors"""
203 |     if src_lang == "bn":
204 |         # নির্বাচন should be "election" not "selection"
205 |         if "নির্বাচন" in q_original:
206 |             q_translated = q_translated.replace("selection", "election")
207 |         # ক্রিকেট should stay "cricket" not "criket"
208 |         if "ক্রিকেট" in q_original:
209 |             q_translated = q_translated.replace("criket", "cricket")
210 | 
211 |     return q_translated
212 | 
213 | 
214 | class QueryProcessor:
215 |     def __init__(self, use_stopwords: bool = False):
216 |         """
217 |         Initialize query processor
218 | 
219 |         Args:
220 |             use_stopwords: Whether to remove stopwords during normalization
221 |         """
222 |         self.translator = TranslatorOPUS()
223 |         self.use_stopwords = use_stopwords
224 | 
225 |     def process(self, raw_query: str) -> Dict:
226 |         """
227 |         Process query through complete pipeline
228 | 
229 |         Returns dict with:
230 |             - original, detected_lang, normalized
231 |             - q_en, q_bn (both languages always available)
232 |             - expansions, named_entity_mappings
233 |             - timing information
234 |         """
235 |         t0 = time.time()
236 | 
237 |         # Step 1: Language Detection
238 |         lang = detect_lang(raw_query)
239 | 
240 |         # Step 2: Normalization
241 |         q_norm = normalize_query(raw_query, lang, self.use_stopwords)
242 | 
243 |         # Step 3: Translation (with error handling)
244 |         try:
245 |             if lang == "bn":
246 |                 q_bn = q_norm
247 |                 q_en = self.translator.translate(q_norm, "bn", "en")
248 |                 q_en = postprocess_translation(q_bn, q_en, "bn")
249 |             else:
250 |                 q_en = q_norm
251 |                 q_bn = self.translator.translate(q_norm, "en", "bn")
252 |         except Exception as e:
253 |             # Fallback: no translation but don't crash
254 |             q_en = q_norm if lang == "en" else raw_query
255 |             q_bn = q_norm if lang == "bn" else raw_query
256 |             print(f"⚠️ Translation failed: {e}")
257 | 
258 |         # Step 4: Query Expansion
259 |         expansions = expand_query(q_norm, lang)
260 | 
261 |         # Step 5: Named Entity Mapping
262 |         ne_mappings = map_named_entities(q_norm, lang)
263 | 
264 |         total_ms = int((time.time() - t0) * 1000)
265 | 
266 |         return {
267 |             "original": raw_query,
268 |             "detected_lang": lang,
269 |             "normalized": q_norm,
270 |             "q_en": q_en,
271 |             "q_bn": q_bn,
272 |             "expansions": expansions,
273 |             "named_entity_mappings": ne_mappings,
274 |             "module_b_time_ms": total_ms,
275 |         }
276 | 
277 | 
278 | if __name__ == "__main__":
279 |     processor = QueryProcessor(use_stopwords=False)
280 | 
281 |     test_queries = [
282 |         "Bangladesh election results 2024",
283 |         "ঢাকায় ক্রিকেট খেলা",
284 |         "Sheikh Hasina news",
285 |         "শিক্ষা ব্যবস্থা সংস্কার",
286 |     ]
287 | 
288 |     print("="*60)
289 |     print("MODULE B - QUERY PROCESSING DEMO")
290 |     print("="*60)
291 | 
292 |     for query in test_queries:
293 |         print(f"\n📝 Query: {query}")
294 |         result = processor.process(query)
295 | 
296 |         print(f"   Language: {result['detected_lang']}")
297 |         print(f"   Normalized: {result['normalized']}")
298 |         print(f"   English: {result['q_en']}")
299 |         print(f"   Bangla: {result['q_bn']}")
300 | 
301 |         if result['expansions']:
302 |             print(f"   Expansions: {result['expansions']}")
303 | 
304 |         if result['named_entity_mappings']:
305 |             print(f"   NE Mappings: {result['named_entity_mappings']}")
306 | 
307 |         print(f"   Time: {result['module_b_time_ms']}ms")
```

**Line-by-line explanation**

- **L1:** Comment: @title
- **L2:** Imports `re` for regular expressions (pattern matching / cleaning text).
- **L3:** Imports `time` for timing and delays (crawl politeness / profiling).
- **L4:** Imports `Dict, List, Tuple, Optional` from `typing`.
- **L5:** Imports `MarianMTModel, MarianTokenizer` from `transformers` for translation + NER pipelines (OPUS-MT / XLM-R).
- **L7:** Comment: -------------------------
- **L8:** Comment: Language Detection
- **L9:** Comment: -------------------------
- **L10:** Assigns a value to `BN_RANGE`.
- **L12:** Defines `detect_lang()` — Detects Bangla vs English using Unicode ranges (fast and reliable for BN/EN). This is the first step in Module B so the system knows which branch to run.
- **L13:** Executes this statement as part of the pipeline.
- **L14:** Returns the function output back to the caller.
- **L17:** Comment: -------------------------
- **L18:** Comment: Stopwords (Optional but Recommended)
- **L19:** Comment: -------------------------
- **L20:** Assigns a value to `STOPWORDS_EN`.
- **L21:** Executes this statement as part of the pipeline.
- **L22:** Executes this statement as part of the pipeline.
- **L23:** Executes this statement as part of the pipeline.
- **L25:** Assigns a value to `STOPWORDS_BN`.
- **L26:** Executes this statement as part of the pipeline.
- **L27:** Executes this statement as part of the pipeline.
- **L28:** Executes this statement as part of the pipeline.
- **L30:** Defines `remove_stopwords()` (a reusable function for the pipeline).
- **L31:** Executes this statement as part of the pipeline.
- **L32:** Assigns a value to `stopwords`.
- **L33:** Assigns a value to `words`.
- **L34:** Assigns a value to `filtered`.
- **L35:** Returns the function output back to the caller.
- **L38:** Comment: -------------------------
- **L39:** Comment: Normalization
- **L40:** Comment: -------------------------
- **L41:** Defines `normalize_query()` — Normalizes user queries (trim whitespace; lowercase only for English). Reduces noise and improves lexical retrieval.
- **L42:** Executes this statement as part of the pipeline.
- **L43:** Assigns a value to `q`.
- **L44:** Conditional branch: only runs the next indented block if the condition is true.
- **L45:** Assigns a value to `q`.
- **L46:** Conditional branch: only runs the next indented block if the condition is true.
- **L47:** Assigns a value to `q`.
- **L48:** Returns the function output back to the caller.
- **L51:** Comment: -------------------------
- **L52:** Comment: Translation (OPUS-MT)
- **L53:** Comment: -------------------------
- **L54:** Defines class `TranslatorOPUS` (a reusable component).
- **L55:** Executes this statement as part of the pipeline.
- **L56:** Executes this statement as part of the pipeline.
- **L57:** Executes this statement as part of the pipeline.
- **L58:** Executes this statement as part of the pipeline.
- **L59:** Executes this statement as part of the pipeline.
- **L60:** Defines `__init__()` (a reusable function for the pipeline).
- **L61:** Assigns a value to `self.cache`.
- **L62:** Assigns a value to `self.max_length`.
- **L64:** Defines `_load()` (a reusable function for the pipeline).
- **L65:** Conditional branch: only runs the next indented block if the condition is true.
- **L66:** Assigns a value to `tok`.
- **L67:** Assigns a value to `mdl`.
- **L68:** Assigns a value to `self.cache[model_name]`.
- **L69:** Returns the function output back to the caller.
- **L71:** Defines `_generate()` (a reusable function for the pipeline).
- **L72:** Assigns a value to `tok, mdl`.
- **L73:** Assigns a value to `batch`.
- **L74:** Assigns a value to `gen`.
- **L75:** Returns the function output back to the caller.
- **L77:** Defines `translate_bn_to_en()` (a reusable function for the pipeline).
- **L78:** Returns the function output back to the caller.
- **L80:** Defines `translate_en_to_bn()` (a reusable function for the pipeline).
- **L81:** Assigns a value to `text_with_lang`.
- **L82:** Assigns a value to `out`.
- **L83:** Returns the function output back to the caller.
- **L85:** Defines `translate()` (a reusable function for the pipeline).
- **L86:** Conditional branch: only runs the next indented block if the condition is true.
- **L87:** Returns the function output back to the caller.
- **L88:** Conditional branch: only runs the next indented block if the condition is true.
- **L89:** Returns the function output back to the caller.
- **L90:** Executes this statement as part of the pipeline.
- **L93:** Comment: -------------------------
- **L94:** Comment: Query Expansion - IMPROVED
- **L95:** Comment: -------------------------
- **L96:** Assigns a value to `EXPAND_EN`.
- **L97:** Comment: Politics
- **L98:** Executes this statement as part of the pipeline.
- **L99:** Executes this statement as part of the pipeline.
- **L100:** Executes this statement as part of the pipeline.
- **L101:** Executes this statement as part of the pipeline.
- **L103:** Comment: Education
- **L104:** Executes this statement as part of the pipeline.
- **L105:** Executes this statement as part of the pipeline.
- **L106:** Executes this statement as part of the pipeline.
- **L108:** Comment: Economy
- **L109:** Executes this statement as part of the pipeline.
- **L110:** Executes this statement as part of the pipeline.
- **L111:** Executes this statement as part of the pipeline.
- **L113:** Comment: Sports
- **L114:** Executes this statement as part of the pipeline.
- **L115:** Executes this statement as part of the pipeline.
- **L116:** Executes this statement as part of the pipeline.
- **L118:** Comment: Common
- **L119:** Executes this statement as part of the pipeline.
- **L120:** Executes this statement as part of the pipeline.
- **L121:** Executes this statement as part of the pipeline.
- **L122:** Executes this statement as part of the pipeline.
- **L124:** Assigns a value to `EXPAND_BN`.
- **L125:** Executes this statement as part of the pipeline.
- **L126:** Executes this statement as part of the pipeline.
- **L127:** Executes this statement as part of the pipeline.
- **L128:** Executes this statement as part of the pipeline.
- **L129:** Executes this statement as part of the pipeline.
- **L130:** Executes this statement as part of the pipeline.
- **L132:** Defines `expand_query()` — Adds a small set of synonyms / related terms to reduce lexical mismatch (recommended Module B step).
- **L133:** Executes this statement as part of the pipeline.
- **L134:** Assigns a value to `expand_dict`.
- **L136:** Assigns a value to `terms`.
- **L137:** Assigns a value to `expanded`.
- **L139:** Starts a loop to process items one-by-one.
- **L140:** Conditional branch: only runs the next indented block if the condition is true.
- **L141:** Executes this statement as part of the pipeline.
- **L143:** Comment: Remove duplicates while preserving order
- **L144:** Returns the function output back to the caller.
- **L147:** Comment: -------------------------
- **L148:** Comment: Named Entity Mapping - IMPROVED
- **L149:** Comment: -------------------------
- **L150:** Assigns a value to `NE_EN2BN`.
- **L151:** Comment: Places
- **L152:** Executes this statement as part of the pipeline.
- **L153:** Executes this statement as part of the pipeline.
- **L154:** Executes this statement as part of the pipeline.
- **L155:** Executes this statement as part of the pipeline.
- **L156:** Executes this statement as part of the pipeline.
- **L157:** Executes this statement as part of the pipeline.
- **L158:** Executes this statement as part of the pipeline.
- **L160:** Comment: Political Figures
- **L161:** Executes this statement as part of the pipeline.
- **L162:** Executes this statement as part of the pipeline.
- **L163:** Executes this statement as part of the pipeline.
- **L165:** Comment: Organizations
- **L166:** Executes this statement as part of the pipeline.
- **L167:** Executes this statement as part of the pipeline.
- **L168:** Executes this statement as part of the pipeline.
- **L170:** Comment: Institutions
- **L171:** Executes this statement as part of the pipeline.
- **L172:** Executes this statement as part of the pipeline.
- **L174:** Executes this statement as part of the pipeline.
- **L175:** Executes this statement as part of the pipeline.
- **L176:** Executes this statement as part of the pipeline.
- **L177:** Executes this statement as part of the pipeline.
- **L179:** Assigns a value to `NE_BN2EN`.
- **L181:** Defines `map_named_entities()` — Maps known named entities across languages (e.g., Dhaka ↔ ঢাকা). This is crucial because NEs often fail under translation.
- **L182:** Executes this statement as part of the pipeline.
- **L183:** Assigns a value to `mapped`.
- **L184:** Assigns a value to `q_normalized`.
- **L186:** Conditional branch: only runs the next indented block if the condition is true.
- **L187:** Starts a loop to process items one-by-one.
- **L188:** Conditional branch: only runs the next indented block if the condition is true.
- **L189:** Executes this statement as part of the pipeline.
- **L190:** Fallback branch if none of the previous conditions matched.
- **L191:** Starts a loop to process items one-by-one.
- **L192:** Conditional branch: only runs the next indented block if the condition is true.
- **L193:** Executes this statement as part of the pipeline.
- **L195:** Returns the function output back to the caller.
- **L198:** Comment: -------------------------
- **L199:** Comment: Translation Post-processing
- **L200:** Comment: -------------------------
- **L201:** Defines `postprocess_translation()` (a reusable function for the pipeline).
- **L202:** Executes this statement as part of the pipeline.
- **L203:** Conditional branch: only runs the next indented block if the condition is true.
- **L204:** Comment: নির্বাচন should be "election" not "selection"
- **L205:** Conditional branch: only runs the next indented block if the condition is true.
- **L206:** Assigns a value to `q_translated`.
- **L207:** Comment: ক্রিকেট should stay "cricket" not "criket"
- **L208:** Conditional branch: only runs the next indented block if the condition is true.
- **L209:** Assigns a value to `q_translated`.
- **L211:** Returns the function output back to the caller.
- **L214:** Defines class `QueryProcessor` (a reusable component).
- **L215:** Defines `__init__()` (a reusable function for the pipeline).
- **L216:** Executes this statement as part of the pipeline.
- **L217:** Executes this statement as part of the pipeline.
- **L219:** Executes this statement as part of the pipeline.
- **L220:** Executes this statement as part of the pipeline.
- **L221:** Executes this statement as part of the pipeline.
- **L222:** Assigns a value to `self.translator`.
- **L223:** Assigns a value to `self.use_stopwords`.
- **L225:** Defines `process()` (a reusable function for the pipeline).
- **L226:** Executes this statement as part of the pipeline.
- **L227:** Executes this statement as part of the pipeline.
- **L229:** Executes this statement as part of the pipeline.
- **L230:** Executes this statement as part of the pipeline.
- **L231:** Executes this statement as part of the pipeline.
- **L232:** Executes this statement as part of the pipeline.
- **L233:** Executes this statement as part of the pipeline.
- **L234:** Executes this statement as part of the pipeline.
- **L235:** Assigns a value to `t0`.
- **L237:** Comment: Step 1: Language Detection
- **L238:** Assigns a value to `lang`.
- **L240:** Comment: Step 2: Normalization
- **L241:** Assigns a value to `q_norm`.
- **L243:** Comment: Step 3: Translation (with error handling)
- **L244:** Executes this statement as part of the pipeline.
- **L245:** Conditional branch: only runs the next indented block if the condition is true.
- **L246:** Assigns a value to `q_bn`.
- **L247:** Assigns a value to `q_en`.
- **L248:** Assigns a value to `q_en`.
- **L249:** Fallback branch if none of the previous conditions matched.
- **L250:** Assigns a value to `q_en`.
- **L251:** Assigns a value to `q_bn`.
- **L252:** Executes this statement as part of the pipeline.
- **L253:** Comment: Fallback: no translation but don't crash
- **L254:** Assigns a value to `q_en`.
- **L255:** Assigns a value to `q_bn`.
- **L256:** Executes this statement as part of the pipeline.
- **L258:** Comment: Step 4: Query Expansion
- **L259:** Assigns a value to `expansions`.
- **L261:** Comment: Step 5: Named Entity Mapping
- **L262:** Assigns a value to `ne_mappings`.
- **L264:** Assigns a value to `total_ms`.
- **L266:** Returns the function output back to the caller.
- **L267:** Executes this statement as part of the pipeline.
- **L268:** Executes this statement as part of the pipeline.
- **L269:** Executes this statement as part of the pipeline.
- **L270:** Executes this statement as part of the pipeline.
- **L271:** Executes this statement as part of the pipeline.
- **L272:** Executes this statement as part of the pipeline.
- **L273:** Executes this statement as part of the pipeline.
- **L274:** Executes this statement as part of the pipeline.
- **L275:** Executes this statement as part of the pipeline.
- **L278:** Conditional branch: only runs the next indented block if the condition is true.
- **L279:** Assigns a value to `processor`.
- **L281:** Assigns a value to `test_queries`.
- **L282:** Executes this statement as part of the pipeline.
- **L283:** Executes this statement as part of the pipeline.
- **L284:** Executes this statement as part of the pipeline.
- **L285:** Executes this statement as part of the pipeline.
- **L286:** Executes this statement as part of the pipeline.
- **L288:** Assigns a value to `print("`.
- **L289:** Executes this statement as part of the pipeline.
- **L290:** Assigns a value to `print("`.
- **L292:** Starts a loop to process items one-by-one.
- **L293:** Executes this statement as part of the pipeline.
- **L294:** Assigns a value to `result`.
- **L296:** Executes this statement as part of the pipeline.
- **L297:** Executes this statement as part of the pipeline.
- **L298:** Executes this statement as part of the pipeline.
- **L299:** Executes this statement as part of the pipeline.
- **L301:** Conditional branch: only runs the next indented block if the condition is true.
- **L302:** Executes this statement as part of the pipeline.
- **L304:** Conditional branch: only runs the next indented block if the condition is true.
- **L305:** Executes this statement as part of the pipeline.
- **L307:** Executes this statement as part of the pipeline.


---

### Markdown cell 25

#Testing With B


---

### Code cell 26

```python
01 | def run_models(bundle, topk=10, w_bm25=0.3, w_sem=0.5, w_fuzzy=0.2):
02 |     q_en = bundle["q_en"]
03 |     q_bn = bundle["q_bn"]
04 | 
05 |     results = []
06 | 
07 |     # --- English side
08 |     bm25_en_hits = bm25_en.search(q_en, topk)
09 |     tfidf_en_hits = tfidf_en.search(q_en, topk)
10 |     fuzzy_en_hits = fuzzy_en.search(q_en, topk)
11 |     sem_en_hits   = sem_en.search(q_en, topk)
12 | 
13 |     hybrid_en_hits = hybrid_fuse(
14 |         bm25_en_hits, sem_en_hits, fuzzy_en_hits,
15 |         w_bm25=w_bm25, w_sem=w_sem, w_fuzzy=w_fuzzy,
16 |         topk=topk
17 |     )
18 | 
19 |     results += attach_docs(df_en, bm25_en_hits, "en", "BM25")
20 |     results += attach_docs(df_en, tfidf_en_hits, "en", "TFIDF")
21 |     results += attach_docs(df_en, fuzzy_en_hits, "en", "FuzzyCharNgram")
22 |     results += attach_docs(df_en, sem_en_hits,   "en", "SemanticEmbed")
23 |     results += attach_docs(df_en, hybrid_en_hits,"en", "Hybrid")
24 | 
25 |     # --- Bangla side
26 |     bm25_bn_hits = bm25_bn.search(q_bn, topk)
27 |     tfidf_bn_hits = tfidf_bn.search(q_bn, topk)
28 |     fuzzy_bn_hits = fuzzy_bn.search(q_bn, topk)
29 |     sem_bn_hits   = sem_bn.search(q_bn, topk)
30 | 
31 |     hybrid_bn_hits = hybrid_fuse(
32 |         bm25_bn_hits, sem_bn_hits, fuzzy_bn_hits,
33 |         w_bm25=w_bm25, w_sem=w_sem, w_fuzzy=w_fuzzy,
34 |         topk=topk
35 |     )
36 | 
37 |     results += attach_docs(df_bn, bm25_bn_hits, "bn", "BM25")
38 |     results += attach_docs(df_bn, tfidf_bn_hits, "bn", "TFIDF")
39 |     results += attach_docs(df_bn, fuzzy_bn_hits, "bn", "FuzzyCharNgram")
40 |     results += attach_docs(df_bn, sem_bn_hits,   "bn", "SemanticEmbed")
41 |     results += attach_docs(df_bn, hybrid_bn_hits,"bn", "Hybrid")
42 | 
43 |     return pd.DataFrame(results).sort_values(["model","lang","score"], ascending=[True, True, False])
```

**Line-by-line explanation**

- **L1:** Defines `run_models()` (a reusable function for the pipeline).
- **L2:** Assigns a value to `q_en`.
- **L3:** Assigns a value to `q_bn`.
- **L5:** Assigns a value to `results`.
- **L7:** Comment: --- English side
- **L8:** Assigns a value to `bm25_en_hits`.
- **L9:** Assigns a value to `tfidf_en_hits`.
- **L10:** Assigns a value to `fuzzy_en_hits`.
- **L11:** Assigns a value to `sem_en_hits`.
- **L13:** Assigns a value to `hybrid_en_hits`.
- **L14:** Executes this statement as part of the pipeline.
- **L15:** Assigns a value to `w_bm25`.
- **L16:** Assigns a value to `topk`.
- **L17:** Executes this statement as part of the pipeline.
- **L19:** Assigns a value to `results +`.
- **L20:** Assigns a value to `results +`.
- **L21:** Assigns a value to `results +`.
- **L22:** Assigns a value to `results +`.
- **L23:** Assigns a value to `results +`.
- **L25:** Comment: --- Bangla side
- **L26:** Assigns a value to `bm25_bn_hits`.
- **L27:** Assigns a value to `tfidf_bn_hits`.
- **L28:** Assigns a value to `fuzzy_bn_hits`.
- **L29:** Assigns a value to `sem_bn_hits`.
- **L31:** Assigns a value to `hybrid_bn_hits`.
- **L32:** Executes this statement as part of the pipeline.
- **L33:** Assigns a value to `w_bm25`.
- **L34:** Assigns a value to `topk`.
- **L35:** Executes this statement as part of the pipeline.
- **L37:** Assigns a value to `results +`.
- **L38:** Assigns a value to `results +`.
- **L39:** Assigns a value to `results +`.
- **L40:** Assigns a value to `results +`.
- **L41:** Assigns a value to `results +`.
- **L43:** Returns the function output back to the caller.


---

### Code cell 27

```python
01 | query ="Tarek Zia"
02 | processor = QueryProcessor(use_stopwords=False)
03 | bundle = processor.process(query)
```

**Line-by-line explanation**

- **L1:** Assigns a value to `query`.
- **L2:** Assigns a value to `processor`.
- **L3:** Assigns a value to `bundle`.


---

### Code cell 28

```python
01 | df_res = run_models(bundle, topk=5, w_bm25=0.3, w_sem=0.5, w_fuzzy=0.2)
02 | df_res
```

**Line-by-line explanation**

- **L1:** Assigns a value to `df_res`.
- **L2:** Executes this statement as part of the pipeline.


---

### Code cell 29

```python
01 | def show_top_titles(df_res, topn=3):
02 |     out = []
03 |     for model in df_res["model"].unique():
04 |         for lang in ["en","bn"]:
05 |             sub = df_res[(df_res["model"]==model) & (df_res["lang"]==lang)].head(topn)
06 |             titles = " | ".join(sub["title"].tolist())
07 |             score = sub["score"].mean()
08 |             out.append({"model": model, "lang": lang, "top_titles": titles,"score":score})
09 |     return pd.DataFrame(out)
10 | 
11 | show_top_titles(df_res, topn=3)
```

**Line-by-line explanation**

- **L1:** Defines `show_top_titles()` (a reusable function for the pipeline).
- **L2:** Assigns a value to `out`.
- **L3:** Starts a loop to process items one-by-one.
- **L4:** Starts a loop to process items one-by-one.
- **L5:** Assigns a value to `sub`.
- **L6:** Assigns a value to `titles`.
- **L7:** Assigns a value to `score`.
- **L8:** Executes this statement as part of the pipeline.
- **L9:** Returns the function output back to the caller.
- **L11:** Assigns a value to `show_top_titles(df_res, topn`.


---

### Markdown cell 30

# All the result  tested by Google Translation in QueryProcessor 


---

### Code cell 31

```python
01 | # -------------------------
02 | # Google Translation wrapper
03 | # -------------------------
04 | def google_translate(text: str, target: str) -> str:
05 |     return GoogleTranslator(source="auto", target=target).translate(text)
```

**Line-by-line explanation**

- **L1:** Comment: -------------------------
- **L2:** Comment: Google Translation wrapper
- **L3:** Comment: -------------------------
- **L4:** Defines `google_translate()` (a reusable function for the pipeline).
- **L5:** Returns the function output back to the caller.


---

### Code cell 32

```python
01 | q_norm="Bangladesh"
02 | google_translate(q_norm, target="bn")
```

**Line-by-line explanation**

- **L1:** Assigns a value to `q_norm`.
- **L2:** Assigns a value to `google_translate(q_norm, target`.


---

### Code cell 33

```python
01 | class QueryProcessorGoogle:
02 |     def __init__(self, use_stopwords: bool = False):
03 |         self.use_stopwords = use_stopwords
04 | 
05 |     def process(self, raw_query: str) -> Dict:
06 |         t0 = time.time()
07 | 
08 |         lang = detect_lang(raw_query)
09 |         q_norm = normalize_query(raw_query, lang, self.use_stopwords)
10 | 
11 |         # expansions + NE mappings (based on normalized query)
12 |         expansions = expand_query(q_norm, lang)
13 |         ne_mappings = map_named_entities(q_norm, lang)
14 | 
15 |         # translation with timing
16 |         t_trans0 = time.time()
17 |         try:
18 |             if lang == "bn":
19 |                 q_bn = q_norm
20 |                 q_en = google_translate(q_norm, target="en")
21 |                 q_en = postprocess_bn_to_en(q_bn, q_en)
22 |             else:
23 |                 q_en = q_norm
24 |                 q_bn = google_translate(q_norm, target="bn")
25 |         except Exception as e:
26 |             # fallback (no crash)
27 |             print(f"⚠️ Google translation failed: {e}")
28 |             q_en = q_norm if lang == "en" else q_norm
29 |             q_bn = q_norm if lang == "bn" else q_norm
30 | 
31 |         trans_ms = int((time.time() - t_trans0) * 1000)
32 | 
33 |         # IMPORTANT: inject NE targets into translated forms to help retrieval
34 |         # (this is what makes names like "Tarek Zia" work better)
35 |         for m in ne_mappings:
36 |             if lang == "en":
37 |                 # English query -> ensure Bangla contains mapped NE
38 |                 q_bn = f"{q_bn} {m['tgt']}".strip()
39 |             else:
40 |                 # Bangla query -> ensure English contains mapped NE
41 |                 q_en = f"{q_en} {m['tgt']}".strip()
42 | 
43 |         total_ms = int((time.time() - t0) * 1000)
44 | 
45 |         return {
46 |             "original": raw_query,
47 |             "detected_lang": lang,
48 |             "normalized": q_norm,
49 |             "q_en": q_en,
50 |             "q_bn": q_bn,
51 |             "expansions": expansions,
52 |             "named_entity_mappings": ne_mappings,
53 |             "time_ms": {
54 |                 "total": total_ms,
55 |                 "translation": trans_ms,
56 |             },
57 |         }
```

**Line-by-line explanation**

- **L1:** Defines class `QueryProcessorGoogle` (a reusable component).
- **L2:** Defines `__init__()` (a reusable function for the pipeline).
- **L3:** Assigns a value to `self.use_stopwords`.
- **L5:** Defines `process()` (a reusable function for the pipeline).
- **L6:** Assigns a value to `t0`.
- **L8:** Assigns a value to `lang`.
- **L9:** Assigns a value to `q_norm`.
- **L11:** Comment: expansions + NE mappings (based on normalized query)
- **L12:** Assigns a value to `expansions`.
- **L13:** Assigns a value to `ne_mappings`.
- **L15:** Comment: translation with timing
- **L16:** Assigns a value to `t_trans0`.
- **L17:** Executes this statement as part of the pipeline.
- **L18:** Conditional branch: only runs the next indented block if the condition is true.
- **L19:** Assigns a value to `q_bn`.
- **L20:** Assigns a value to `q_en`.
- **L21:** Assigns a value to `q_en`.
- **L22:** Fallback branch if none of the previous conditions matched.
- **L23:** Assigns a value to `q_en`.
- **L24:** Assigns a value to `q_bn`.
- **L25:** Executes this statement as part of the pipeline.
- **L26:** Comment: fallback (no crash)
- **L27:** Executes this statement as part of the pipeline.
- **L28:** Assigns a value to `q_en`.
- **L29:** Assigns a value to `q_bn`.
- **L31:** Assigns a value to `trans_ms`.
- **L33:** Comment: IMPORTANT: inject NE targets into translated forms to help retrieval
- **L34:** Comment: (this is what makes names like "Tarek Zia" work better)
- **L35:** Starts a loop to process items one-by-one.
- **L36:** Conditional branch: only runs the next indented block if the condition is true.
- **L37:** Comment: English query -> ensure Bangla contains mapped NE
- **L38:** Assigns a value to `q_bn`.
- **L39:** Fallback branch if none of the previous conditions matched.
- **L40:** Comment: Bangla query -> ensure English contains mapped NE
- **L41:** Assigns a value to `q_en`.
- **L43:** Assigns a value to `total_ms`.
- **L45:** Returns the function output back to the caller.
- **L46:** Executes this statement as part of the pipeline.
- **L47:** Executes this statement as part of the pipeline.
- **L48:** Executes this statement as part of the pipeline.
- **L49:** Executes this statement as part of the pipeline.
- **L50:** Executes this statement as part of the pipeline.
- **L51:** Executes this statement as part of the pipeline.
- **L52:** Executes this statement as part of the pipeline.
- **L53:** Executes this statement as part of the pipeline.
- **L54:** Executes this statement as part of the pipeline.
- **L55:** Executes this statement as part of the pipeline.
- **L56:** Executes this statement as part of the pipeline.
- **L57:** Executes this statement as part of the pipeline.


---

### Code cell 34

```python
01 | qp = QueryProcessorGoogle(use_stopwords=False)
02 | 
03 | bundle = qp.process("Tarek Zia")
04 | df_res = run_models(bundle, topk=5, w_bm25=0.3, w_sem=0.5, w_fuzzy=0.2)
05 | df_res
```

**Line-by-line explanation**

- **L1:** Assigns a value to `qp`.
- **L3:** Assigns a value to `bundle`.
- **L4:** Assigns a value to `df_res`.
- **L5:** Executes this statement as part of the pipeline.


---

### Code cell 35

```python
01 | show_top_titles(df_res, topn=3)
```

**Line-by-line explanation**

- **L1:** Assigns a value to `show_top_titles(df_res, topn`.


---

### Markdown cell 36

# Improved Version


---

### Code cell 37

```python
01 | df_bn["title"][2996]
```

**Line-by-line explanation**

- **L1:** Executes this statement as part of the pipeline.


---

### Code cell 38

```python
01 | df_en["title"][2996]
```

**Line-by-line explanation**

- **L1:** Executes this statement as part of the pipeline.


---

### Code cell 39

```python
01 | df = pd.concat([df_en, df_bn], ignore_index=True)
02 | df
```

**Line-by-line explanation**

- **L1:** Assigns a value to `df`.
- **L2:** Executes this statement as part of the pipeline.


---

### Code cell 40

```python
01 | # df["title"][2996]
02 | df["title"][5996]
```

**Line-by-line explanation**

- **L1:** Comment: df["title"][2996]
- **L2:** Executes this statement as part of the pipeline.


---

### Code cell 41

```python
01 | from FlagEmbedding import BGEM3FlagModel
02 | import numpy as np
03 | 
04 | 
05 | model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
06 | 
07 | 
08 | corpus_text = df['body'].tolist()
09 | corpus_titles = df['title'].tolist()
10 | 
11 | 
12 | embeddings = model.encode(
13 |     corpus_text,
14 |     batch_size=12,
15 |     max_length=512
16 | )["dense_vecs"]
17 | 
18 | 
19 | # embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
```

**Line-by-line explanation**

- **L1:** Imports `BGEM3FlagModel` from `FlagEmbedding` for BGE embedding models (semantic retrieval).
- **L2:** Imports `numpy` for fast numeric arrays (scoring, sorting top-k).
- **L5:** Loads the BGE‑M3 embedding model (dense semantic retrieval).
- **L8:** Assigns a value to `corpus_text`.
- **L9:** Assigns a value to `corpus_titles`.
- **L12:** Assigns a value to `embeddings`.
- **L13:** Executes this statement as part of the pipeline.
- **L14:** Assigns a value to `batch_size`.
- **L15:** Assigns a value to `max_length`.
- **L16:** Executes this statement as part of the pipeline.
- **L19:** Comment: embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)


---

### Code cell 42

```python
01 | tokenized_corpus = [doc.split(" ") for doc in corpus_text]
02 | bm25 = BM25Okapi(tokenized_corpus)
```

**Line-by-line explanation**

- **L1:** Assigns a value to `tokenized_corpus`.
- **L2:** Builds a BM25 index over tokenized documents for lexical ranking.


---

### Code cell 43

```python
01 | def hybrid_search(query, top_k=10, alpha=0.7):
02 |     """
03 |     alpha: Weight for Semantic Vector Search (0.0 to 1.0)
04 |            1.0 = Pure Vector, 0.0 = Pure Keyword/Fuzzy
05 |     """
06 |     # --- A. SEMANTIC SEARCH (Dense) ---
07 |     query_embedding = model.encode([query])['dense_vecs']
08 |     # Calculate Cosine Similarity (Dot product for normalized vectors)
09 |     semantic_scores = embeddings @ query_embedding.T
10 |     semantic_scores = semantic_scores.flatten()
11 | 
12 |     # --- B. LEXICAL/FUZZY SEARCH (Sparse) ---
13 |     # 1. First, handle typos in query using Fuzzy matching against corpus words (Optional optimization)
14 |     # (Here we stick to BM25 for robustness as full fuzzy on 5k docs is slow,
15 |     # but BM25 handles partial matches well).
16 |     tokenized_query = query.split(" ")
17 |     lexical_scores = bm25.get_scores(tokenized_query)
18 | 
19 |     # Normalize scores to 0-1 range to combine them
20 |     def normalize(scores):
21 |         return (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-9)
22 | 
23 |     norm_semantic = normalize(semantic_scores)
24 |     norm_lexical = normalize(lexical_scores)
25 | 
26 |     # --- C. HYBRID FUSION ---
27 |     final_scores = (alpha * norm_semantic) + ((1 - alpha) * norm_lexical)
28 | 
29 |     # Get Top-K
30 |     top_indices = np.argsort(final_scores)[::-1][:top_k]
31 | 
32 |     results = {}
33 |     for idx in top_indices:
34 |         results[corpus_titles[idx]] = {
35 |             "title": corpus_titles[idx],
36 |             "body": corpus_text[idx],
37 |             "score": final_scores[idx],
38 |             "type": "Semantic" if norm_semantic[idx] > norm_lexical[idx] else "Lexical/Fuzzy"
39 |         }
40 |     return results
```

**Line-by-line explanation**

- **L1:** Defines `hybrid_search()` (a reusable function for the pipeline).
- **L2:** Executes this statement as part of the pipeline.
- **L3:** Executes this statement as part of the pipeline.
- **L4:** Assigns a value to `1.0`.
- **L5:** Executes this statement as part of the pipeline.
- **L6:** Comment: --- A. SEMANTIC SEARCH (Dense) ---
- **L7:** Assigns a value to `query_embedding`.
- **L8:** Comment: Calculate Cosine Similarity (Dot product for normalized vectors)
- **L9:** Assigns a value to `semantic_scores`.
- **L10:** Assigns a value to `semantic_scores`.
- **L12:** Comment: --- B. LEXICAL/FUZZY SEARCH (Sparse) ---
- **L13:** Comment: 1. First, handle typos in query using Fuzzy matching against corpus words (Optional optimization)
- **L14:** Comment: (Here we stick to BM25 for robustness as full fuzzy on 5k docs is slow,
- **L15:** Comment: but BM25 handles partial matches well).
- **L16:** Assigns a value to `tokenized_query`.
- **L17:** Assigns a value to `lexical_scores`.
- **L19:** Comment: Normalize scores to 0-1 range to combine them
- **L20:** Defines `normalize()` (a reusable function for the pipeline).
- **L21:** Returns the function output back to the caller.
- **L23:** Assigns a value to `norm_semantic`.
- **L24:** Assigns a value to `norm_lexical`.
- **L26:** Comment: --- C. HYBRID FUSION ---
- **L27:** Assigns a value to `final_scores`.
- **L29:** Comment: Get Top-K
- **L30:** Sorts document scores in descending order and keeps the top‑K indices (ranking step).
- **L32:** Assigns a value to `results`.
- **L33:** Starts a loop to process items one-by-one.
- **L34:** Assigns a value to `results[corpus_titles[idx]]`.
- **L35:** Executes this statement as part of the pipeline.
- **L36:** Executes this statement as part of the pipeline.
- **L37:** Executes this statement as part of the pipeline.
- **L38:** Executes this statement as part of the pipeline.
- **L39:** Executes this statement as part of the pipeline.
- **L40:** Returns the function output back to the caller.


---

### Code cell 44

```python
01 | def dual_translation_hybrid_search(query:str, alpha=0.7):
02 |     results = {}
03 |     queries = return_translated(query)
04 |     for q in queries:
05 |         r = hybrid_search(q)
06 |         results = results | r
07 | 
08 |     return results
```

**Line-by-line explanation**

- **L1:** Defines `dual_translation_hybrid_search()` (a reusable function for the pipeline).
- **L2:** Assigns a value to `results`.
- **L3:** Assigns a value to `queries`.
- **L4:** Starts a loop to process items one-by-one.
- **L5:** Assigns a value to `r`.
- **L6:** Assigns a value to `results`.
- **L8:** Returns the function output back to the caller.


---

### Code cell 45

```python
01 | query = "Tarek Zia"
02 | results = dual_translation_hybrid_search(query)
03 | 
04 | print(f"\nResults for '{query}':")
05 | for res in results.values():
06 |     print(f"[{res['type']}] {res['title']} ({res['score']:.4f})")
07 |     # print(res["body"], "\n\n")
```

**Line-by-line explanation**

- **L1:** Assigns a value to `query`.
- **L2:** Assigns a value to `results`.
- **L4:** Executes this statement as part of the pipeline.
- **L5:** Starts a loop to process items one-by-one.
- **L6:** Executes this statement as part of the pipeline.
- **L7:** Comment: print(res["body"], "\n\n")


---

### Code cell 46

```python
01 | query = "Bangladesh Election"
02 | results = dual_translation_hybrid_search(query) 
03 | 
04 | data_list = []
05 | 
06 | for res in results.values():
07 | 
08 |     row = {
09 |         'type': res['type'],
10 |         'title': res['title'],
11 |         'score': res['score'],
12 |         'body': res.get('body', "")  
13 |     }
14 |     
15 |     data_list.append(row)
16 | 
17 | 
18 | df_sem = pd.DataFrame(data_list)
19 | 
20 | 
21 | df_sem
```

**Line-by-line explanation**

- **L1:** Assigns a value to `query`.
- **L2:** Assigns a value to `results`.
- **L4:** Assigns a value to `data_list`.
- **L6:** Starts a loop to process items one-by-one.
- **L8:** Assigns a value to `row`.
- **L9:** Executes this statement as part of the pipeline.
- **L10:** Executes this statement as part of the pipeline.
- **L11:** Executes this statement as part of the pipeline.
- **L12:** Executes this statement as part of the pipeline.
- **L13:** Executes this statement as part of the pipeline.
- **L15:** Executes this statement as part of the pipeline.
- **L18:** Assigns a value to `df_sem`.
- **L21:** Executes this statement as part of the pipeline.


---

### Code cell 47

```python
01 | k = 5
02 | top_k_df = df_sem.nlargest(k, 'score')
03 | 
04 | top_k_df
```

**Line-by-line explanation**

- **L1:** Assigns a value to `k`.
- **L2:** Assigns a value to `top_k_df`.
- **L4:** Executes this statement as part of the pipeline.


---

### Markdown cell 48

# Retrieval Evaluation
# Module D — Ranking, Scoring, & Evaluation (Core)

This section adds **top‑K ranking with [0,1] matching confidence**, **query-time reporting**, and **IR evaluation metrics + labeling workflow**.


---

### Code cell 49

```python
01 | import time, math
02 | import numpy as np
03 | import pandas as pd
04 | from typing import Dict, List, Any, Tuple
```

**Line-by-line explanation**

- **L1:** Imports `time` for timing and delays (crawl politeness / profiling), `math`.
- **L2:** Imports `numpy` for fast numeric arrays (scoring, sorting top-k).
- **L3:** Imports `pandas` for tabular data handling with DataFrames (filtering, saving).
- **L4:** Imports `Dict, List, Any, Tuple` from `typing`.


---

### Markdown cell 50

## D1) Ranking & Matching Score (0–1) + Low-confidence Warning


---

### Code cell 51

```python
01 | def _clip01(x: float) -> float:
02 |     return float(max(0.0, min(1.0, x)))
03 | 
04 | def rank_topk_for_query(
05 |     raw_query: str,
06 |     topk: int = 10,
07 |     *,
08 |     confidence_warn_threshold: float = 0.20,
09 |     w_bm25: float = 0.3,
10 |     w_sem: float = 0.5,
11 |     w_fuzzy: float = 0.2,
12 |     return_all_models: bool = False,
13 | ) -> Dict[str, Any]:
14 |     """Runs your pipeline and returns a sorted top‑K list with matching_score ∈ [0,1]."""
15 |     qp = QueryProcessorGoogle()
16 |     bundle = qp.process(raw_query)
17 | 
18 |     t0 = time.perf_counter()
19 | 
20 |     # Retrieval (timed)
21 |     t_retr0 = time.perf_counter()
22 |     df_res = run_models(bundle, topk=max(topk, 50), w_bm25=w_bm25, w_sem=w_sem, w_fuzzy=w_fuzzy)
23 |     retr_ms = int((time.perf_counter() - t_retr0) * 1000)
24 | 
25 |     # Final ranking = Hybrid results across both languages
26 |     df_hybrid = df_res[df_res["model"] == "Hybrid"].copy()
27 |     df_hybrid = df_hybrid.sort_values("score", ascending=False)
28 | 
29 |     # Ensure matching_score is in [0,1]
30 |     df_hybrid["matching_score"] = df_hybrid["score"].map(_clip01)
31 | 
32 |     top_df = df_hybrid.head(topk).reset_index(drop=True)
33 | 
34 |     total_ms = int((time.perf_counter() - t0) * 1000)
35 |     time_ms = {
36 |         "total": total_ms,
37 |         "translation": bundle.get("time_ms", {}).get("translation"),
38 |         "retrieval_total": retr_ms,
39 |     }
40 | 
41 |     warning = None
42 |     if len(top_df) > 0:
43 |         top_score = float(top_df.loc[0, "matching_score"])
44 |         if top_score < confidence_warn_threshold:
45 |             warning = (
46 |                 f"⚠ Warning: Retrieved results may not be relevant. Matching confidence is low (score: {top_score:.2f}).\n"
47 |                 "Consider rephrasing your query or checking translation quality."
48 |             )
49 | 
50 |     out = {
51 |         "query": raw_query,
52 |         "bundle": bundle,
53 |         "topk": top_df[["lang","title","url","date","matching_score"]].to_dict(orient="records"),
54 |         "warning": warning,
55 |         "time_ms": time_ms,
56 |     }
57 |     if return_all_models:
58 |         out["all_models"] = df_res
59 |     return out
60 | 
61 | # Example:
62 | r = rank_topk_for_query("ঢাকা বিশ্ববিদ্যালয় ভর্তি", topk=5)
63 | r["warning"], r["time_ms"], r["topk"][0]
```

**Line-by-line explanation**

- **L1:** Defines `_clip01()` (a reusable function for the pipeline).
- **L2:** Returns the function output back to the caller.
- **L4:** Defines `rank_topk_for_query()` — Module D entry point: runs query processing + retrieval, produces a final top‑K list with confidence score ∈ [0,1], and emits a low-confidence warning when needed.
- **L5:** Executes this statement as part of the pipeline.
- **L6:** Assigns a value to `topk: int`.
- **L7:** Executes this statement as part of the pipeline.
- **L8:** Assigns a value to `confidence_warn_threshold: float`.
- **L9:** Assigns a value to `w_bm25: float`.
- **L10:** Assigns a value to `w_sem: float`.
- **L11:** Assigns a value to `w_fuzzy: float`.
- **L12:** Assigns a value to `return_all_models: bool`.
- **L13:** Executes this statement as part of the pipeline.
- **L14:** Executes this statement as part of the pipeline.
- **L15:** Assigns a value to `qp`.
- **L16:** Assigns a value to `bundle`.
- **L18:** Assigns a value to `t0`.
- **L20:** Comment: Retrieval (timed)
- **L21:** Assigns a value to `t_retr0`.
- **L22:** Assigns a value to `df_res`.
- **L23:** Assigns a value to `retr_ms`.
- **L25:** Comment: Final ranking = Hybrid results across both languages
- **L26:** Assigns a value to `df_hybrid`.
- **L27:** Assigns a value to `df_hybrid`.
- **L29:** Comment: Ensure matching_score is in [0,1]
- **L30:** Assigns a value to `df_hybrid["matching_score"]`.
- **L32:** Assigns a value to `top_df`.
- **L34:** Assigns a value to `total_ms`.
- **L35:** Assigns a value to `time_ms`.
- **L36:** Executes this statement as part of the pipeline.
- **L37:** Executes this statement as part of the pipeline.
- **L38:** Executes this statement as part of the pipeline.
- **L39:** Executes this statement as part of the pipeline.
- **L41:** Assigns a value to `warning`.
- **L42:** Conditional branch: only runs the next indented block if the condition is true.
- **L43:** Assigns a value to `top_score`.
- **L44:** Conditional branch: only runs the next indented block if the condition is true.
- **L45:** Assigns a value to `warning`.
- **L46:** Executes this statement as part of the pipeline.
- **L47:** Executes this statement as part of the pipeline.
- **L48:** Executes this statement as part of the pipeline.
- **L50:** Assigns a value to `out`.
- **L51:** Executes this statement as part of the pipeline.
- **L52:** Executes this statement as part of the pipeline.
- **L53:** Assigns a value to `"topk": top_df[["lang","title","url","date","matching_score"]].to_dict(orient`.
- **L54:** Executes this statement as part of the pipeline.
- **L55:** Executes this statement as part of the pipeline.
- **L56:** Executes this statement as part of the pipeline.
- **L57:** Conditional branch: only runs the next indented block if the condition is true.
- **L58:** Assigns a value to `out["all_models"]`.
- **L59:** Returns the function output back to the caller.
- **L61:** Comment: Example:
- **L62:** Assigns a value to `r`.
- **L63:** Executes this statement as part of the pipeline.


---

### Markdown cell 52

## D2) Query Execution Time Report


---

### Code cell 53

```python
01 | def pretty_time_report(r: Dict[str, Any]) -> None:
02 |     q = r.get("query")
03 |     tm = r.get("time_ms", {})
04 |     print(f"Query: {q}")
05 |     print("Time (ms):")
06 |     for k in ["translation", "retrieval_total", "total"]:
07 |         if tm.get(k) is not None:
08 |             print(f"  - {k:16s}: {tm[k]}")
09 |     if r.get("warning"):
10 |         print("\n" + r["warning"])
11 | 
12 | def run_batch_queries(queries: List[str], topk:int=10) -> pd.DataFrame:
13 |     rows = []
14 |     for q in queries:
15 |         r = rank_topk_for_query(q, topk=topk)
16 |         tm = r["time_ms"]
17 |         rows.append({
18 |             "query": q,
19 |             "detected_lang": r["bundle"].get("detected_lang"),
20 |             "translation_ms": tm.get("translation"),
21 |             "retrieval_ms": tm.get("retrieval_total"),
22 |             "total_ms": tm.get("total"),
23 |             "top1_score": r["topk"][0]["matching_score"] if r["topk"] else None,
24 |             "warn": bool(r.get("warning")),
25 |         })
26 |     return pd.DataFrame(rows)
27 | 
28 | # Example:
29 | queries = ["ঢাকা", "Bangladesh election", "শিক্ষা মন্ত্রণালয়"]
30 | df_perf = run_batch_queries(queries, topk=10)
31 | df_perf
```

**Line-by-line explanation**

- **L1:** Defines `pretty_time_report()` (a reusable function for the pipeline).
- **L2:** Assigns a value to `q`.
- **L3:** Assigns a value to `tm`.
- **L4:** Executes this statement as part of the pipeline.
- **L5:** Executes this statement as part of the pipeline.
- **L6:** Starts a loop to process items one-by-one.
- **L7:** Conditional branch: only runs the next indented block if the condition is true.
- **L8:** Executes this statement as part of the pipeline.
- **L9:** Conditional branch: only runs the next indented block if the condition is true.
- **L10:** Executes this statement as part of the pipeline.
- **L12:** Defines `run_batch_queries()` (a reusable function for the pipeline).
- **L13:** Assigns a value to `rows`.
- **L14:** Starts a loop to process items one-by-one.
- **L15:** Assigns a value to `r`.
- **L16:** Assigns a value to `tm`.
- **L17:** Executes this statement as part of the pipeline.
- **L18:** Executes this statement as part of the pipeline.
- **L19:** Executes this statement as part of the pipeline.
- **L20:** Executes this statement as part of the pipeline.
- **L21:** Executes this statement as part of the pipeline.
- **L22:** Executes this statement as part of the pipeline.
- **L23:** Executes this statement as part of the pipeline.
- **L24:** Executes this statement as part of the pipeline.
- **L25:** Executes this statement as part of the pipeline.
- **L26:** Returns the function output back to the caller.
- **L28:** Comment: Example:
- **L29:** Assigns a value to `queries`.
- **L30:** Assigns a value to `df_perf`.
- **L31:** Executes this statement as part of the pipeline.


---

### Markdown cell 54

## D3) Evaluation Metrics (Precision@10, Recall@50, nDCG@10, MRR)

### Labeling format (CSV)
Create a simple CSV with columns:
- `query`
- `doc_url`
- `language`
- `is_relevant` (yes/no)
- `annotator`


---

### Code cell 55

```python
01 | def make_labeling_sheet(
02 |     queries: List[str],
03 |     topn_per_query: int = 50,
04 |     out_csv_path: str = "labels_template.csv",
05 | ) -> pd.DataFrame:
06 |     rows = []
07 |     for q in queries:
08 |         r = rank_topk_for_query(q, topk=topn_per_query)
09 |         for item in r["topk"]:
10 |             rows.append({
11 |                 "query": q,
12 |                 "doc_url": item["url"],
13 |                 "language": item["lang"],
14 |                 "is_relevant": "",   # fill with yes/no manually
15 |                 "annotator": "",
16 |             })
17 |     df = pd.DataFrame(rows).drop_duplicates(["query","doc_url"])
18 |     df.to_csv(out_csv_path, index=False, encoding="utf-8")
19 |     return df
```

**Line-by-line explanation**

- **L1:** Defines `make_labeling_sheet()` (a reusable function for the pipeline).
- **L2:** Executes this statement as part of the pipeline.
- **L3:** Assigns a value to `topn_per_query: int`.
- **L4:** Assigns a value to `out_csv_path: str`.
- **L5:** Executes this statement as part of the pipeline.
- **L6:** Assigns a value to `rows`.
- **L7:** Starts a loop to process items one-by-one.
- **L8:** Assigns a value to `r`.
- **L9:** Starts a loop to process items one-by-one.
- **L10:** Executes this statement as part of the pipeline.
- **L11:** Executes this statement as part of the pipeline.
- **L12:** Executes this statement as part of the pipeline.
- **L13:** Executes this statement as part of the pipeline.
- **L14:** Executes this statement as part of the pipeline.
- **L15:** Executes this statement as part of the pipeline.
- **L16:** Executes this statement as part of the pipeline.
- **L17:** Assigns a value to `df`.
- **L18:** Writes a DataFrame to a CSV file (used for labels/exports).
- **L19:** Returns the function output back to the caller.


---

### Code cell 56

```python
01 | def _bin_rel(x: Any) -> int:
02 |     s = "" if pd.isna(x) else str(x).strip().lower()
03 |     return 1 if s in {"1","true","yes","y"} else 0
04 | 
05 | def precision_at_k(rels: List[int], k: int) -> float:
06 |     return float(sum(rels[:k])) / float(k) if k > 0 else 0.0
07 | 
08 | def recall_at_k(rels: List[int], total_relevant: int, k: int) -> float:
09 |     return (float(sum(rels[:k])) / float(total_relevant)) if total_relevant > 0 else 0.0
10 | 
11 | def dcg_at_k(rels: List[int], k: int) -> float:
12 |     score = 0.0
13 |     for i, rel in enumerate(rels[:k], start=1):
14 |         if rel:
15 |             score += 1.0 / math.log2(i + 1)
16 |     return score
17 | 
18 | def ndcg_at_k(rels: List[int], k: int) -> float:
19 |     dcg = dcg_at_k(rels, k)
20 |     ideal = sorted(rels, reverse=True)
21 |     idcg = dcg_at_k(ideal, k)
22 |     return (dcg / idcg) if idcg > 0 else 0.0
23 | 
24 | def mrr(rels: List[int]) -> float:
25 |     for i, rel in enumerate(rels, start=1):
26 |         if rel:
27 |             return 1.0 / float(i)
28 |     return 0.0
29 | 
30 | def evaluate_from_labels(
31 |     labels_csv_path: str,
32 |     *,
33 |     topk_eval: int = 50,
34 | ) -> Tuple[pd.DataFrame, Dict[str, float]]:
35 |     labels = pd.read_csv(labels_csv_path)
36 |     labels["rel"] = labels["is_relevant"].map(_bin_rel)
37 | 
38 |     per_query = []
39 |     for q, g in labels.groupby("query"):
40 |         total_rel = int(g["rel"].sum())
41 |         r = rank_topk_for_query(q, topk=topk_eval)
42 |         ranked_urls = [x["url"] for x in r["topk"]]
43 |         url2rel = {row["doc_url"]: int(row["rel"]) for _, row in g.iterrows()}
44 |         rels = [url2rel.get(u, 0) for u in ranked_urls]
45 | 
46 |         per_query.append({
47 |             "query": q,
48 |             "P@10": precision_at_k(rels, 10),
49 |             "R@50": recall_at_k(rels, total_rel, 50),
50 |             "nDCG@10": ndcg_at_k(rels, 10),
51 |             "MRR": mrr(rels),
52 |             "total_relevant_labeled": total_rel,
53 |         })
54 | 
55 |     dfq = pd.DataFrame(per_query)
56 |     macro = {
57 |         "Precision@10": float(dfq["P@10"].mean()) if len(dfq) else 0.0,
58 |         "Recall@50": float(dfq["R@50"].mean()) if len(dfq) else 0.0,
59 |         "nDCG@10": float(dfq["nDCG@10"].mean()) if len(dfq) else 0.0,
60 |         "MRR": float(dfq["MRR"].mean()) if len(dfq) else 0.0,
61 |         "num_queries": int(len(dfq)),
62 |     }
63 |     return dfq, macro
```

**Line-by-line explanation**

- **L1:** Defines `_bin_rel()` (a reusable function for the pipeline).
- **L2:** Assigns a value to `s`.
- **L3:** Returns the function output back to the caller.
- **L5:** Defines `precision_at_k()` (a reusable function for the pipeline).
- **L6:** Returns the function output back to the caller.
- **L8:** Defines `recall_at_k()` (a reusable function for the pipeline).
- **L9:** Returns the function output back to the caller.
- **L11:** Defines `dcg_at_k()` (a reusable function for the pipeline).
- **L12:** Assigns a value to `score`.
- **L13:** Starts a loop to process items one-by-one.
- **L14:** Conditional branch: only runs the next indented block if the condition is true.
- **L15:** Assigns a value to `score +`.
- **L16:** Returns the function output back to the caller.
- **L18:** Defines `ndcg_at_k()` (a reusable function for the pipeline).
- **L19:** Assigns a value to `dcg`.
- **L20:** Assigns a value to `ideal`.
- **L21:** Assigns a value to `idcg`.
- **L22:** Returns the function output back to the caller.
- **L24:** Defines `mrr()` (a reusable function for the pipeline).
- **L25:** Starts a loop to process items one-by-one.
- **L26:** Conditional branch: only runs the next indented block if the condition is true.
- **L27:** Returns the function output back to the caller.
- **L28:** Returns the function output back to the caller.
- **L30:** Defines `evaluate_from_labels()` — Computes IR metrics (P@10, R@50, nDCG@10, MRR) using your manual relevance labels CSV. This is the core evaluation required by the assignment.
- **L31:** Executes this statement as part of the pipeline.
- **L32:** Executes this statement as part of the pipeline.
- **L33:** Assigns a value to `topk_eval: int`.
- **L34:** Executes this statement as part of the pipeline.
- **L35:** Assigns a value to `labels`.
- **L36:** Assigns a value to `labels["rel"]`.
- **L38:** Assigns a value to `per_query`.
- **L39:** Starts a loop to process items one-by-one.
- **L40:** Assigns a value to `total_rel`.
- **L41:** Assigns a value to `r`.
- **L42:** Assigns a value to `ranked_urls`.
- **L43:** Assigns a value to `url2rel`.
- **L44:** Assigns a value to `rels`.
- **L46:** Executes this statement as part of the pipeline.
- **L47:** Executes this statement as part of the pipeline.
- **L48:** Executes this statement as part of the pipeline.
- **L49:** Executes this statement as part of the pipeline.
- **L50:** Executes this statement as part of the pipeline.
- **L51:** Executes this statement as part of the pipeline.
- **L52:** Executes this statement as part of the pipeline.
- **L53:** Executes this statement as part of the pipeline.
- **L55:** Assigns a value to `dfq`.
- **L56:** Assigns a value to `macro`.
- **L57:** Executes this statement as part of the pipeline.
- **L58:** Executes this statement as part of the pipeline.
- **L59:** Executes this statement as part of the pipeline.
- **L60:** Executes this statement as part of the pipeline.
- **L61:** Executes this statement as part of the pipeline.
- **L62:** Executes this statement as part of the pipeline.
- **L63:** Returns the function output back to the caller.


---

### Markdown cell 57

### Comparing with Google/Bing/DuckDuckGo (Required)

Recommended simple workflow (manual):

1) For each of evaluation queries, search on Google/Bing/DDG.
2) Copy the **top 10 URLs** into a CSV (one file per engine) with columns: `query, doc_url, rank`.
3) Label those URLs using the same yes/no relevance labels.
4) Evaluate them with the helper below.


---

### Code cell 58

```python
01 | def evaluate_external_engine(
02 |     engine_ranked_csv: str,
03 |     labels_csv_path: str,
04 |     *,
05 |     k: int = 10,
06 | ) -> pd.DataFrame:
07 |     """Evaluate an external engine's ranking at k using your manual labels.
08 |     engine_ranked_csv columns: query, doc_url, rank (1 = best)
09 |     labels_csv columns: query, doc_url, is_relevant
10 |     """
11 |     eng = pd.read_csv(engine_ranked_csv)
12 |     lbl = pd.read_csv(labels_csv_path)
13 |     lbl["rel"] = lbl["is_relevant"].map(_bin_rel)
14 | 
15 |     label_dict = {(r["query"], r["doc_url"]): int(r["rel"]) for _, r in lbl.iterrows()}
16 | 
17 |     rows = []
18 |     for q, g in eng.groupby("query"):
19 |         g = g.sort_values("rank").head(k)
20 |         rels = [label_dict.get((q, u), 0) for u in g["doc_url"].tolist()]
21 |         rows.append({
22 |             "query": q,
23 |             f"P@{k}": precision_at_k(rels, k),
24 |             f"nDCG@{k}": ndcg_at_k(rels, k),
25 |             "MRR": mrr(rels),
26 |         })
27 |     return pd.DataFrame(rows)
```

**Line-by-line explanation**

- **L1:** Defines `evaluate_external_engine()` (a reusable function for the pipeline).
- **L2:** Executes this statement as part of the pipeline.
- **L3:** Executes this statement as part of the pipeline.
- **L4:** Executes this statement as part of the pipeline.
- **L5:** Assigns a value to `k: int`.
- **L6:** Executes this statement as part of the pipeline.
- **L7:** Executes this statement as part of the pipeline.
- **L8:** Assigns a value to `engine_ranked_csv columns: query, doc_url, rank (1`.
- **L9:** Executes this statement as part of the pipeline.
- **L10:** Executes this statement as part of the pipeline.
- **L11:** Assigns a value to `eng`.
- **L12:** Assigns a value to `lbl`.
- **L13:** Assigns a value to `lbl["rel"]`.
- **L15:** Assigns a value to `label_dict`.
- **L17:** Assigns a value to `rows`.
- **L18:** Starts a loop to process items one-by-one.
- **L19:** Assigns a value to `g`.
- **L20:** Assigns a value to `rels`.
- **L21:** Executes this statement as part of the pipeline.
- **L22:** Executes this statement as part of the pipeline.
- **L23:** Executes this statement as part of the pipeline.
- **L24:** Executes this statement as part of the pipeline.
- **L25:** Executes this statement as part of the pipeline.
- **L26:** Executes this statement as part of the pipeline.
- **L27:** Returns the function output back to the caller.


---

### Markdown cell 59

## D4) Error Analysis Helpers (Case-study)


---

### Code cell 60

```python
01 | def quick_case_study(raw_query: str, topk:int=10) -> None:
02 |     r = rank_topk_for_query(raw_query, topk=topk, return_all_models=True)
03 |     print("Query:", r["query"])
04 |     print("Detected:", r["bundle"].get("detected_lang"))
05 |     print("Normalized:", r["bundle"].get("normalized"))
06 |     print("q_en:", r["bundle"].get("q_en"))
07 |     print("q_bn:", r["bundle"].get("q_bn"))
08 |     pretty_time_report(r)
09 | 
10 |     print("\nTop results:")
11 |     for i, item in enumerate(r["topk"], start=1):
12 |         print(f"  {i:02d}. [{item['lang']}] {item['matching_score']:.3f} | {item['title']}")
13 | 
14 |     df_all = r.get("all_models")
15 |     if df_all is not None and len(df_all):
16 |         print("\nModel comparison (top 3 per model/lang):")
17 |         print(show_top_titles(df_all, topn=3))
```

**Line-by-line explanation**

- **L1:** Defines `quick_case_study()` — Prints a compact ‘error analysis’ view for a query: translation outputs, timings, and top results + model comparison table.
- **L2:** Assigns a value to `r`.
- **L3:** Executes this statement as part of the pipeline.
- **L4:** Executes this statement as part of the pipeline.
- **L5:** Executes this statement as part of the pipeline.
- **L6:** Executes this statement as part of the pipeline.
- **L7:** Executes this statement as part of the pipeline.
- **L8:** Executes this statement as part of the pipeline.
- **L10:** Executes this statement as part of the pipeline.
- **L11:** Starts a loop to process items one-by-one.
- **L12:** Executes this statement as part of the pipeline.
- **L14:** Assigns a value to `df_all`.
- **L15:** Conditional branch: only runs the next indented block if the condition is true.
- **L16:** Executes this statement as part of the pipeline.
- **L17:** Assigns a value to `print(show_top_titles(df_all, topn`.


---

### Code cell 61

```python
01 | EVAL_QUERIES = [
02 |     # Bangla (BN)
03 |     "বাংলাদেশে মূল্যস্ফীতি কমেছে কি?",
04 |     "ঢাকায় যানজট কমাতে নতুন পরিকল্পনা",
05 |     "রোহিঙ্গা ক্যাম্পে স্বাস্থ্যসেবা পরিস্থিতি",
06 |     "পদ্মা সেতু অর্থনৈতিক প্রভাব",
07 |     "বাংলাদেশ বনাম ভারত সিরিজ সূচি",
08 | 
09 |     # English (EN)
10 |     "Bangladesh GDP growth forecast 2024",
11 |     "Dhaka air pollution AQI health impact",
12 |     "Rohingya repatriation latest updates",
13 |     "Padma Bridge impact on trade and logistics",
14 |     "Bangladesh election reforms debate",
15 | ]
16 | 
17 | LABEL_SHEET_PATH = "labels_to_fill.csv"
18 | 
19 | 
20 | label_df = make_labeling_sheet(
21 |     queries=EVAL_QUERIES,
22 |     topn_per_query=50,           
23 |     out_csv_path=LABEL_SHEET_PATH
24 | )
25 | 
26 | print("Saved labeling sheet to:", LABEL_SHEET_PATH)
27 | label_df.head(10)
```

**Line-by-line explanation**

- **L1:** Assigns a value to `EVAL_QUERIES`.
- **L2:** Comment: Bangla (BN)
- **L3:** Executes this statement as part of the pipeline.
- **L4:** Executes this statement as part of the pipeline.
- **L5:** Executes this statement as part of the pipeline.
- **L6:** Executes this statement as part of the pipeline.
- **L7:** Executes this statement as part of the pipeline.
- **L9:** Comment: English (EN)
- **L10:** Executes this statement as part of the pipeline.
- **L11:** Executes this statement as part of the pipeline.
- **L12:** Executes this statement as part of the pipeline.
- **L13:** Executes this statement as part of the pipeline.
- **L14:** Executes this statement as part of the pipeline.
- **L15:** Executes this statement as part of the pipeline.
- **L17:** Assigns a value to `LABEL_SHEET_PATH`.
- **L20:** Assigns a value to `label_df`.
- **L21:** Assigns a value to `queries`.
- **L22:** Assigns a value to `topn_per_query`.
- **L23:** Assigns a value to `out_csv_path`.
- **L24:** Executes this statement as part of the pipeline.
- **L26:** Executes this statement as part of the pipeline.
- **L27:** Executes this statement as part of the pipeline.


---

### Code cell 62

```python
01 | len(label_df)
```

**Line-by-line explanation**

- **L1:** Executes this statement as part of the pipeline.


---

### Code cell 63

```python
01 | label_df["query"][55]
```

**Line-by-line explanation**

- **L1:** Executes this statement as part of the pipeline.


---

### Markdown cell 64

# For short annotation purpose doing the less labeling 


---

### Code cell 65

```python
01 | label_df_2 = make_labeling_sheet(
02 |     queries=EVAL_QUERIES,
03 |     topn_per_query=3,           
04 |     out_csv_path=LABEL_SHEET_PATH
05 | )
```

**Line-by-line explanation**

- **L1:** Assigns a value to `label_df_2`.
- **L2:** Assigns a value to `queries`.
- **L3:** Assigns a value to `topn_per_query`.
- **L4:** Assigns a value to `out_csv_path`.
- **L5:** Executes this statement as part of the pipeline.


---

### Code cell 66

```python
01 | label_df_2.head(10)
```

**Line-by-line explanation**

- **L1:** Executes this statement as part of the pipeline.


---

### Code cell 67

```python
01 | label_df_2["doc_url"]
```

**Line-by-line explanation**

- **L1:** Executes this statement as part of the pipeline.


---

### Code cell 68

```python
01 | FILLED_LABELS_PATH = "data/processed/labels_fiiled.csv"  
02 | 
03 | metrics = evaluate_from_labels(FILLED_LABELS_PATH)
04 | metrics
```

**Line-by-line explanation**

- **L1:** Assigns a value to `FILLED_LABELS_PATH`.
- **L3:** Assigns a value to `metrics`.
- **L4:** Executes this statement as part of the pipeline.


---

### Code cell 69

```python
01 | per_query = evaluate_per_query(FILLED_LABELS_PATH)
02 | per_query.sort_values("nDCG@10", ascending=False).head(20)
```

**Line-by-line explanation**

- **L1:** Assigns a value to `per_query`.
- **L2:** Assigns a value to `per_query.sort_values("nDCG@10", ascending`.


---

### Code cell 70

```python
01 | # Pick failures (low score, wrong topic) + successes (semantic wins)
02 | quick_case_study("বাংলাদেশ বনাম ভারত সিরিজ সূচি")
03 | quick_case_study("Bangla Desh vs বাংলাদেশ spelling in articles")  # optional stress test
04 | quick_case_study("Dhaka air pollution AQI health impact")
```

**Line-by-line explanation**

- **L1:** Comment: Pick failures (low score, wrong topic) + successes (semantic wins)
- **L2:** Executes this statement as part of the pipeline.
- **L3:** Executes this statement as part of the pipeline.
- **L4:** Executes this statement as part of the pipeline.


## Practical usage tips

- Run cells **top-to-bottom** (Colab state matters).
- If a cell installs packages (`!pip ...`), run it once and then restart runtime only if you get version conflicts.
- Keep your datasets under a consistent folder (e.g., `data/processed/`) so Module C/D can load them without edits.


## Quick fixes (copy/paste)

- **Same TF‑IDF stray backtick bug** as Module C (remove it).
- `evaluate_per_query(...)` is called but **not defined**.  
  ✅ Fix: use `dfq, macro = evaluate_from_labels(...)` and treat `dfq` as your per-query table (or define a wrapper).
- Ensure your filled labels file path is correct: the notebook uses `data/processed/labels_fiiled.csv` (note the spelling).
