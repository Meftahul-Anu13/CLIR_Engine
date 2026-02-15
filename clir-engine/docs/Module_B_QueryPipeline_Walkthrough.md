# Module B — Code Walkthrough (Query_pipeline_Module_B.ipynb)

This file explains the notebook **cell-by-cell** with **what each part does**, **why it exists** in the CLIR pipeline, and **how to use it** for your assignment.

## Where this fits in the assignment


**Module B (Query Processing & Cross-Lingual Handling)**  
This notebook implements:
- Language detection (BN vs EN)  
- Normalization (whitespace + lowercasing for EN)  
- Query translation using OPUS-MT (transformers / Marian)  
- Query expansion (simple synonym dictionary)  
- Named-entity extraction (spaCy EN + XLM-R NER) + cross-lingual NE mapping  
- Full pipeline wrapper returning both `q_en` and `q_bn`


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

- This notebook contains an **“improved pipeline” copy** later in the file (it redefines functions like `detect_lang`). That is OK, but remember: later definitions override earlier ones.
- Translation models can be slow on CPU. For quick demos, keep queries short.

## Cell-by-cell explanation


---

### Code cell 1

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

### Code cell 2

```python
01 | !pip -q install sacremoses
```

**Line-by-line explanation**

- **L1:** Colab shell command to install Python packages needed for this module.


---

### Markdown cell 3

# Language detection (Bangla vs English)


---

### Code cell 4

```python
01 | import re
02 | 
03 | BN_RANGE = re.compile(r'[\u0980-\u09FF]')
04 | 
05 | def detect_lang(query: str) -> str:
06 |     q = query.strip()
07 |     if BN_RANGE.search(q):
08 |         return "bn"
09 |     # fallback: assume english if mostly latin
10 |     return "en"
```

**Line-by-line explanation**

- **L1:** Imports `re` for regular expressions (pattern matching / cleaning text).
- **L3:** Assigns a value to `BN_RANGE`.
- **L5:** Defines `detect_lang()` — Detects Bangla vs English using Unicode ranges (fast and reliable for BN/EN). This is the first step in Module B so the system knows which branch to run.
- **L6:** Assigns a value to `q`.
- **L7:** Conditional branch: only runs the next indented block if the condition is true.
- **L8:** Returns the function output back to the caller.
- **L9:** Comment: fallback: assume english if mostly latin
- **L10:** Returns the function output back to the caller.


---

### Markdown cell 5

# Normalization (required)


---

### Code cell 6

```python
01 | def normalize_query(query: str, lang: str) -> str:
02 |     q = " ".join(query.strip().split())  # collapse whitespace
03 |     if lang == "en":
04 |         q = q.lower()
05 |     # For Bangla, lowercasing doesn't matter; keep as-is
06 |     return q
```

**Line-by-line explanation**

- **L1:** Defines `normalize_query()` — Normalizes user queries (trim whitespace; lowercase only for English). Reduces noise and improves lexical retrieval.
- **L2:** Assigns a value to `q`.
- **L3:** Conditional branch: only runs the next indented block if the condition is true.
- **L4:** Assigns a value to `q`.
- **L5:** Comment: For Bangla, lowercasing doesn't matter; keep as-is
- **L6:** Returns the function output back to the caller.


---

### Markdown cell 7

# Query translation / conversion


---

### Code cell 8

```python
01 | from transformers import MarianMTModel, MarianTokenizer
02 | 
03 | class OpusTranslator:
04 |     """
05 |     bn -> en : Helsinki-NLP/opus-mt-bn-en
06 |     en -> bn : Helsinki-NLP/opus-mt-en-iir with required >>ben<< token
07 |     """
08 |     def __init__(self):
09 |         self.cache = {}
10 | 
11 |     def _load(self, model_name: str):
12 |         if model_name not in self.cache:
13 |             tok = MarianTokenizer.from_pretrained(model_name)
14 |             mdl = MarianMTModel.from_pretrained(model_name)
15 |             self.cache[model_name] = (tok, mdl)
16 |         return self.cache[model_name]
17 | 
18 |     def translate_bn_to_en(self, text: str) -> str:
19 |         model_name = "Helsinki-NLP/opus-mt-bn-en"
20 |         tok, mdl = self._load(model_name)
21 |         batch = tok([text], return_tensors="pt", truncation=True)
22 |         gen = mdl.generate(**batch, max_length=64)
23 |         return tok.decode(gen[0], skip_special_tokens=True)
24 | 
25 |     def translate_en_to_bn(self, text: str) -> str:
26 |         # multilingual target group; needs language token >>ben<<
27 |         model_name = "Helsinki-NLP/opus-mt-en-iir"
28 |         tok, mdl = self._load(model_name)
29 | 
30 |         text_with_lang = f">>ben<< {text}"  # REQUIRED by model card :contentReference[oaicite:2]{index=2}
31 |         batch = tok([text_with_lang], return_tensors="pt", truncation=True)
32 |         gen = mdl.generate(**batch, max_length=64)
33 |         out = tok.decode(gen[0], skip_special_tokens=True)
34 | 
35 |         # sometimes the output may keep the token; strip defensively
36 |         return out.replace(">>ben<<", "").strip()
37 | 
38 | translator = OpusTranslator()
39 | 
40 | def translate_query(q_norm: str, lang: str):
41 |     if lang == "bn":
42 |         return q_norm, translator.translate_bn_to_en(q_norm)   # (bn, en)
43 |     else:
44 |         return q_norm, translator.translate_en_to_bn(q_norm)   # (en, bn)
```

**Line-by-line explanation**

- **L1:** Imports `MarianMTModel, MarianTokenizer` from `transformers` for translation + NER pipelines (OPUS-MT / XLM-R).
- **L3:** Defines class `OpusTranslator` (a reusable component).
- **L4:** Executes this statement as part of the pipeline.
- **L5:** Executes this statement as part of the pipeline.
- **L6:** Executes this statement as part of the pipeline.
- **L7:** Executes this statement as part of the pipeline.
- **L8:** Defines `__init__()` (a reusable function for the pipeline).
- **L9:** Assigns a value to `self.cache`.
- **L11:** Defines `_load()` (a reusable function for the pipeline).
- **L12:** Conditional branch: only runs the next indented block if the condition is true.
- **L13:** Assigns a value to `tok`.
- **L14:** Assigns a value to `mdl`.
- **L15:** Assigns a value to `self.cache[model_name]`.
- **L16:** Returns the function output back to the caller.
- **L18:** Defines `translate_bn_to_en()` (a reusable function for the pipeline).
- **L19:** Assigns a value to `model_name`.
- **L20:** Assigns a value to `tok, mdl`.
- **L21:** Assigns a value to `batch`.
- **L22:** Assigns a value to `gen`.
- **L23:** Returns the function output back to the caller.
- **L25:** Defines `translate_en_to_bn()` (a reusable function for the pipeline).
- **L26:** Comment: multilingual target group; needs language token >>ben<<
- **L27:** Assigns a value to `model_name`.
- **L28:** Assigns a value to `tok, mdl`.
- **L30:** Assigns a value to `text_with_lang`.
- **L31:** Assigns a value to `batch`.
- **L32:** Assigns a value to `gen`.
- **L33:** Assigns a value to `out`.
- **L35:** Comment: sometimes the output may keep the token; strip defensively
- **L36:** Returns the function output back to the caller.
- **L38:** Assigns a value to `translator`.
- **L40:** Defines `translate_query()` — Translates the normalized query to the other language so the same query can retrieve documents from both corpora (required CLIR step).
- **L41:** Conditional branch: only runs the next indented block if the condition is true.
- **L42:** Returns the function output back to the caller.
- **L43:** Fallback branch if none of the previous conditions matched.
- **L44:** Returns the function output back to the caller.


---

### Markdown cell 9

# Query expansion


---

### Code cell 10

```python
01 | EXPAND_DICT_EN = {
02 |     "election": ["vote", "poll", "ballot"],
03 |     "education": ["school", "university", "student"],
04 | }
05 | 
06 | def expand_query(q: str, lang: str):
07 |     terms = q.split()
08 |     extra = []
09 |     if lang == "en":
10 |         for t in terms:
11 |             extra += EXPAND_DICT_EN.get(t, [])
12 |     # For bn you can keep empty initially
13 |     extra = list(dict.fromkeys(extra))  # unique preserve order
14 |     return extra
```

**Line-by-line explanation**

- **L1:** Assigns a value to `EXPAND_DICT_EN`.
- **L2:** Executes this statement as part of the pipeline.
- **L3:** Executes this statement as part of the pipeline.
- **L4:** Executes this statement as part of the pipeline.
- **L6:** Defines `expand_query()` — Adds a small set of synonyms / related terms to reduce lexical mismatch (recommended Module B step).
- **L7:** Assigns a value to `terms`.
- **L8:** Assigns a value to `extra`.
- **L9:** Conditional branch: only runs the next indented block if the condition is true.
- **L10:** Starts a loop to process items one-by-one.
- **L11:** Assigns a value to `extra +`.
- **L12:** Comment: For bn you can keep empty initially
- **L13:** Assigns a value to `extra`.
- **L14:** Returns the function output back to the caller.


---

### Markdown cell 11

# Named Entity extraction + mapping


---

### Markdown cell 12

`Layer 1 `


---

### Code cell 13

```python
01 | NE_MAP_EN2BN = {
02 |     "bangladesh": "বাংলাদেশ",
03 |     "dhaka": "ঢাকা",
04 |     "sheikh hasina": "শেখ হাসিনা",
05 | }
06 | NE_MAP_BN2EN = {v: k for k, v in NE_MAP_EN2BN.items()}
07 | 
08 | def map_named_entities(q: str, lang: str):
09 |     q_low = q.lower()
10 |     mapped = []
11 |     if lang == "en":
12 |         for k, v in NE_MAP_EN2BN.items():
13 |             if k in q_low:
14 |                 mapped.append((k, v))
15 |     else:
16 |         for k, v in NE_MAP_BN2EN.items():
17 |             if k in q:
18 |                 mapped.append((k, v))
19 |     return mapped
```

**Line-by-line explanation**

- **L1:** Assigns a value to `NE_MAP_EN2BN`.
- **L2:** Executes this statement as part of the pipeline.
- **L3:** Executes this statement as part of the pipeline.
- **L4:** Executes this statement as part of the pipeline.
- **L5:** Executes this statement as part of the pipeline.
- **L6:** Assigns a value to `NE_MAP_BN2EN`.
- **L8:** Defines `map_named_entities()` — Maps known named entities across languages (e.g., Dhaka ↔ ঢাকা). This is crucial because NEs often fail under translation.
- **L9:** Assigns a value to `q_low`.
- **L10:** Assigns a value to `mapped`.
- **L11:** Conditional branch: only runs the next indented block if the condition is true.
- **L12:** Starts a loop to process items one-by-one.
- **L13:** Conditional branch: only runs the next indented block if the condition is true.
- **L14:** Executes this statement as part of the pipeline.
- **L15:** Fallback branch if none of the previous conditions matched.
- **L16:** Starts a loop to process items one-by-one.
- **L17:** Conditional branch: only runs the next indented block if the condition is true.
- **L18:** Executes this statement as part of the pipeline.
- **L19:** Returns the function output back to the caller.


---

### Markdown cell 14

`Layer 2 `


---

### Code cell 15

```python
01 | import spacy
02 | 
03 | nlp_en = spacy.load("en_core_web_sm")
04 | 
05 | def extract_ner_en(text: str):
06 |     doc = nlp_en(text)
07 |     entities = []
08 |     for ent in doc.ents:
09 |         entities.append({
10 |             "text": ent.text,
11 |             "label": ent.label_  # PERSON, GPE, ORG, LOC, etc.
12 |         })
13 |     return entities
```

**Line-by-line explanation**

- **L1:** Imports `spacy` for English NLP pipeline (tokenization/NER).
- **L3:** Assigns a value to `nlp_en`.
- **L5:** Defines `extract_ner_en()` (a reusable function for the pipeline).
- **L6:** Assigns a value to `doc`.
- **L7:** Assigns a value to `entities`.
- **L8:** Starts a loop to process items one-by-one.
- **L9:** Executes this statement as part of the pipeline.
- **L10:** Executes this statement as part of the pipeline.
- **L11:** Executes this statement as part of the pipeline.
- **L12:** Executes this statement as part of the pipeline.
- **L13:** Returns the function output back to the caller.


---

### Code cell 16

```python
01 | from transformers import pipeline
02 | 
03 | ner_bn = pipeline(
04 |     "ner",
05 |     model="Davlan/xlm-roberta-base-ner-hrl",
06 |     aggregation_strategy="simple"
07 | )
```

**Line-by-line explanation**

- **L1:** Imports `pipeline` from `transformers` for translation + NER pipelines (OPUS-MT / XLM-R).
- **L3:** Assigns a value to `ner_bn`.
- **L4:** Executes this statement as part of the pipeline.
- **L5:** Assigns a value to `model`.
- **L6:** Assigns a value to `aggregation_strategy`.
- **L7:** Executes this statement as part of the pipeline.


---

### Code cell 17

```python
01 | def extract_ner_bn(text: str):
02 |     results = ner_bn(text)
03 |     entities = []
04 |     for r in results:
05 |         entities.append({
06 |             "text": r["word"],
07 |             "label": r["entity_group"]  # PER, LOC, ORG
08 |         })
09 |     return entities
```

**Line-by-line explanation**

- **L1:** Defines `extract_ner_bn()` (a reusable function for the pipeline).
- **L2:** Assigns a value to `results`.
- **L3:** Assigns a value to `entities`.
- **L4:** Starts a loop to process items one-by-one.
- **L5:** Executes this statement as part of the pipeline.
- **L6:** Executes this statement as part of the pipeline.
- **L7:** Executes this statement as part of the pipeline.
- **L8:** Executes this statement as part of the pipeline.
- **L9:** Returns the function output back to the caller.


---

### Code cell 18

```python
01 | NE_MAP_EN2BN = {
02 |     "bangladesh": "বাংলাদেশ",
03 |     "dhaka": "ঢাকা",
04 |     "sheikh hasina": "শেখ হাসিনা",
05 |     "awami league": "আওয়ামী লীগ",
06 | }
07 | 
08 | NE_MAP_BN2EN = {v: k for k, v in NE_MAP_EN2BN.items()}
09 | def map_entities(entities, lang):
10 |     mapped = []
11 |     for e in entities:
12 |         key = e["text"].lower()
13 |         if lang == "en" and key in NE_MAP_EN2BN:
14 |             mapped.append({
15 |                 "src": e["text"],
16 |                 "tgt": NE_MAP_EN2BN[key],
17 |                 "type": e["label"]
18 |             })
19 |         elif lang == "bn" and e["text"] in NE_MAP_BN2EN:
20 |             mapped.append({
21 |                 "src": e["text"],
22 |                 "tgt": NE_MAP_BN2EN[e["text"]],
23 |                 "type": e["label"]
24 |             })
25 |     return mapped
26 | def extract_and_map_entities(query: str, lang: str):
27 |     if lang == "en":
28 |         ents = extract_ner_en(query)
29 |     else:
30 |         ents = extract_ner_bn(query)
31 | 
32 |     mapped = map_entities(ents, lang)
33 | 
34 |     return {
35 |         "entities": ents,
36 |         "mapped_entities": mapped
37 |     }
```

**Line-by-line explanation**

- **L1:** Assigns a value to `NE_MAP_EN2BN`.
- **L2:** Executes this statement as part of the pipeline.
- **L3:** Executes this statement as part of the pipeline.
- **L4:** Executes this statement as part of the pipeline.
- **L5:** Executes this statement as part of the pipeline.
- **L6:** Executes this statement as part of the pipeline.
- **L8:** Assigns a value to `NE_MAP_BN2EN`.
- **L9:** Defines `map_entities()` (a reusable function for the pipeline).
- **L10:** Assigns a value to `mapped`.
- **L11:** Starts a loop to process items one-by-one.
- **L12:** Assigns a value to `key`.
- **L13:** Conditional branch: only runs the next indented block if the condition is true.
- **L14:** Executes this statement as part of the pipeline.
- **L15:** Executes this statement as part of the pipeline.
- **L16:** Executes this statement as part of the pipeline.
- **L17:** Executes this statement as part of the pipeline.
- **L18:** Executes this statement as part of the pipeline.
- **L19:** Alternative conditional branch if previous conditions were false.
- **L20:** Executes this statement as part of the pipeline.
- **L21:** Executes this statement as part of the pipeline.
- **L22:** Executes this statement as part of the pipeline.
- **L23:** Executes this statement as part of the pipeline.
- **L24:** Executes this statement as part of the pipeline.
- **L25:** Returns the function output back to the caller.
- **L26:** Defines `extract_and_map_entities()` (a reusable function for the pipeline).
- **L27:** Conditional branch: only runs the next indented block if the condition is true.
- **L28:** Assigns a value to `ents`.
- **L29:** Fallback branch if none of the previous conditions matched.
- **L30:** Assigns a value to `ents`.
- **L32:** Assigns a value to `mapped`.
- **L34:** Returns the function output back to the caller.
- **L35:** Executes this statement as part of the pipeline.
- **L36:** Executes this statement as part of the pipeline.
- **L37:** Executes this statement as part of the pipeline.


---

### Markdown cell 19

# Pipeline


---

### Code cell 20

```python
01 | import time
02 | def process_query(query: str):
03 |     t0 = time.time()
04 | 
05 |     lang = detect_lang(query)
06 |     q_norm = normalize_query(query, lang)
07 | 
08 |     t_trans0 = time.time()
09 |     q_src, q_other = translate_query(q_norm, lang)
10 |     trans_ms = int((time.time() - t_trans0) * 1000)
11 | 
12 |     if lang == "bn":
13 |         q_bn, q_en = q_src, q_other
14 |     else:
15 |         q_en, q_bn = q_src, q_other
16 | 
17 |     t_ner0 = time.time()
18 |     ner_info = extract_and_map_entities(q_norm, lang)
19 |     ner_ms = int((time.time() - t_ner0) * 1000)
20 | 
21 |     expansions = expand_query(q_norm, lang)
22 | 
23 |     total_ms = int((time.time() - t0) * 1000)
24 | 
25 |     return {
26 |         "query": query,
27 |         "detected_lang": lang,
28 |         "normalized": q_norm,
29 |         "q_en": q_en,
30 |         "q_bn": q_bn,
31 |         "expansions": expansions,
32 |         "entities": ner_info.get("entities", []),
33 |         "mapped_entities": ner_info.get("mapped_entities", []),
34 |         "time_ms": {
35 |             "total": total_ms,
36 |             "translation": trans_ms,
37 |             "ner": ner_ms
38 |         }
39 |     }
```

**Line-by-line explanation**

- **L1:** Imports `time` for timing and delays (crawl politeness / profiling).
- **L2:** Defines `process_query()` (a reusable function for the pipeline).
- **L3:** Assigns a value to `t0`.
- **L5:** Assigns a value to `lang`.
- **L6:** Assigns a value to `q_norm`.
- **L8:** Assigns a value to `t_trans0`.
- **L9:** Assigns a value to `q_src, q_other`.
- **L10:** Assigns a value to `trans_ms`.
- **L12:** Conditional branch: only runs the next indented block if the condition is true.
- **L13:** Assigns a value to `q_bn, q_en`.
- **L14:** Fallback branch if none of the previous conditions matched.
- **L15:** Assigns a value to `q_en, q_bn`.
- **L17:** Assigns a value to `t_ner0`.
- **L18:** Assigns a value to `ner_info`.
- **L19:** Assigns a value to `ner_ms`.
- **L21:** Assigns a value to `expansions`.
- **L23:** Assigns a value to `total_ms`.
- **L25:** Returns the function output back to the caller.
- **L26:** Executes this statement as part of the pipeline.
- **L27:** Executes this statement as part of the pipeline.
- **L28:** Executes this statement as part of the pipeline.
- **L29:** Executes this statement as part of the pipeline.
- **L30:** Executes this statement as part of the pipeline.
- **L31:** Executes this statement as part of the pipeline.
- **L32:** Executes this statement as part of the pipeline.
- **L33:** Executes this statement as part of the pipeline.
- **L34:** Executes this statement as part of the pipeline.
- **L35:** Executes this statement as part of the pipeline.
- **L36:** Executes this statement as part of the pipeline.
- **L37:** Executes this statement as part of the pipeline.
- **L38:** Executes this statement as part of the pipeline.
- **L39:** Executes this statement as part of the pipeline.


---

### Code cell 21

```python
01 | process_query("বাংলাদেশ নির্বাচন")
```

**Line-by-line explanation**

- **L1:** Executes this statement as part of the pipeline.


---

### Code cell 22

```python
01 | process_query("Bangladesh election")
```

**Line-by-line explanation**

- **L1:** Executes this statement as part of the pipeline.


---

### Code cell 23

```python
01 | queries = [
02 |     "বাংলাদেশ নির্বাচন",
03 |     "Bangladesh election",
04 |     "ঢাকা traffic problem",
05 |     "বাংলাদেশ election results",
06 |     "চেয়ার"
07 | ]
08 | 
09 | for q in queries:
10 |     print(process_query(q))
11 |     print("-" * 80)
```

**Line-by-line explanation**

- **L1:** Assigns a value to `queries`.
- **L2:** Executes this statement as part of the pipeline.
- **L3:** Executes this statement as part of the pipeline.
- **L4:** Executes this statement as part of the pipeline.
- **L5:** Executes this statement as part of the pipeline.
- **L6:** Executes this statement as part of the pipeline.
- **L7:** Executes this statement as part of the pipeline.
- **L9:** Starts a loop to process items one-by-one.
- **L10:** Executes this statement as part of the pipeline.
- **L11:** Executes this statement as part of the pipeline.


---

### Code cell 24

```python
01 | LATIN_RANGE = re.compile(r'[A-Za-z]')
02 | BN_RANGE = re.compile(r'[\u0980-\u09FF]')
03 | 
04 | def is_mixed_script(text: str) -> bool:
05 |     return bool(BN_RANGE.search(text)) and bool(LATIN_RANGE.search(text))
06 | 
07 | def split_mixed(text: str):
08 |     bn_part = " ".join(re.findall(r'[\u0980-\u09FF]+', text))
09 |     en_part = " ".join(re.findall(r'[A-Za-z]+', text))
10 |     return bn_part.strip(), en_part.strip()
```

**Line-by-line explanation**

- **L1:** Assigns a value to `LATIN_RANGE`.
- **L2:** Assigns a value to `BN_RANGE`.
- **L4:** Defines `is_mixed_script()` (a reusable function for the pipeline).
- **L5:** Returns the function output back to the caller.
- **L7:** Defines `split_mixed()` (a reusable function for the pipeline).
- **L8:** Assigns a value to `bn_part`.
- **L9:** Assigns a value to `en_part`.
- **L10:** Returns the function output back to the caller.


---

### Code cell 25

```python
01 | def postprocess_bn_to_en(q_bn: str, q_en: str) -> str:
02 |     if "নির্বাচন" in q_bn:
03 |         q_en = q_en.replace("selection", "election")
04 |     return q_en
05 | 
06 | def postprocess_en_to_bn(q_en: str, q_bn: str) -> str:
07 |     # If election appears, ensure Bangla contains নির্বাচন (not নিষেধাজ্ঞা নির্বাচন)
08 |     if "election" in q_en and "নির্বাচন" not in q_bn:
09 |         q_bn = q_bn.replace("নিষেধাজ্ঞা", "").strip()
10 |         if "নির্বাচন" not in q_bn:
11 |             q_bn = f"{q_bn} নির্বাচন".strip()
12 |     # If results appears, add ফলাফল if missing
13 |     if "results" in q_en and "ফলাফল" not in q_bn:
14 |         q_bn = f"{q_bn} ফলাফল".strip()
15 |     return q_bn
```

**Line-by-line explanation**

- **L1:** Defines `postprocess_bn_to_en()` (a reusable function for the pipeline).
- **L2:** Conditional branch: only runs the next indented block if the condition is true.
- **L3:** Assigns a value to `q_en`.
- **L4:** Returns the function output back to the caller.
- **L6:** Defines `postprocess_en_to_bn()` (a reusable function for the pipeline).
- **L7:** Comment: If election appears, ensure Bangla contains নির্বাচন (not নিষেধাজ্ঞা নির্বাচন)
- **L8:** Conditional branch: only runs the next indented block if the condition is true.
- **L9:** Assigns a value to `q_bn`.
- **L10:** Conditional branch: only runs the next indented block if the condition is true.
- **L11:** Assigns a value to `q_bn`.
- **L12:** Comment: If results appears, add ফলাফল if missing
- **L13:** Conditional branch: only runs the next indented block if the condition is true.
- **L14:** Assigns a value to `q_bn`.
- **L15:** Returns the function output back to the caller.


---

### Markdown cell 26

#Testing


---

### Code cell 27

```python
001 | import re
002 | import time
003 | from typing import Dict, List, Tuple, Optional
004 | from transformers import MarianMTModel, MarianTokenizer
005 | 
006 | # -------------------------
007 | # Language Detection
008 | # -------------------------
009 | BN_RANGE = re.compile(r"[\u0980-\u09FF]")
010 | 
011 | def detect_lang(query: str) -> str:
012 |     """Detect if query is Bangla or English"""
013 |     return "bn" if BN_RANGE.search(query or "") else "en"
014 | 
015 | 
016 | # -------------------------
017 | # Stopwords (Optional but Recommended)
018 | # -------------------------
019 | STOPWORDS_EN = {
020 |     "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
021 |     "to", "for", "of", "and", "or", "but", "with", "from", "by"
022 | }
023 | 
024 | STOPWORDS_BN = {
025 |     "এবং", "বা", "কিন্তু", "যা", "যে", "এই", "সেই", "ও", "তে", "থেকে",
026 |     "দিয়ে", "হয়", "হয়েছে", "করা", "করে", "একটি", "কোনো", "আর"
027 | }
028 | 
029 | def remove_stopwords(query: str, lang: str) -> str:
030 |     """Remove stopwords (optional preprocessing)"""
031 |     stopwords = STOPWORDS_BN if lang == "bn" else STOPWORDS_EN
032 |     words = query.split()
033 |     filtered = [w for w in words if w.lower() not in stopwords]
034 |     return " ".join(filtered) if filtered else query  # fallback to original if all removed
035 | 
036 | 
037 | # -------------------------
038 | # Normalization
039 | # -------------------------
040 | def normalize_query(query: str, lang: str, remove_stops: bool = False) -> str:
041 |     """Normalize query: trim, lowercase (EN only), optional stopword removal"""
042 |     q = " ".join((query or "").strip().split())
043 |     if lang == "en":
044 |         q = q.lower()
045 |     if remove_stops:
046 |         q = remove_stopwords(q, lang)
047 |     return q
048 | 
049 | 
050 | # -------------------------
051 | # Translation (OPUS-MT)
052 | # -------------------------
053 | class TranslatorOPUS:
054 |     """
055 |     OPUS-MT translator with caching
056 |     - bn → en: Helsinki-NLP/opus-mt-bn-en
057 |     - en → bn: Helsinki-NLP/opus-mt-en-iir with >>ben<< token
058 |     """
059 |     def __init__(self, max_length: int = 64):
060 |         self.cache = {}
061 |         self.max_length = max_length
062 | 
063 |     def _load(self, model_name: str) -> Tuple[MarianTokenizer, MarianMTModel]:
064 |         if model_name not in self.cache:
065 |             tok = MarianTokenizer.from_pretrained(model_name)
066 |             mdl = MarianMTModel.from_pretrained(model_name)
067 |             self.cache[model_name] = (tok, mdl)
068 |         return self.cache[model_name]
069 | 
070 |     def _generate(self, model_name: str, text: str) -> str:
071 |         tok, mdl = self._load(model_name)
072 |         batch = tok([text], return_tensors="pt", truncation=True)
073 |         gen = mdl.generate(**batch, max_length=self.max_length)
074 |         return tok.decode(gen[0], skip_special_tokens=True).strip()
075 | 
076 |     def translate_bn_to_en(self, text: str) -> str:
077 |         return self._generate("Helsinki-NLP/opus-mt-bn-en", text)
078 | 
079 |     def translate_en_to_bn(self, text: str) -> str:
080 |         text_with_lang = f">>ben<< {text}"
081 |         out = self._generate("Helsinki-NLP/opus-mt-en-iir", text_with_lang)
082 |         return out.replace(">>ben<<", "").strip()
083 | 
084 |     def translate(self, text: str, src: str, tgt: str) -> str:
085 |         if src == "bn" and tgt == "en":
086 |             return self.translate_bn_to_en(text)
087 |         if src == "en" and tgt == "bn":
088 |             return self.translate_en_to_bn(text)
089 |         raise ValueError(f"Unsupported translation: {src}→{tgt}")
090 | 
091 | 
092 | # -------------------------
093 | # Query Expansion - IMPROVED
094 | # -------------------------
095 | EXPAND_EN = {
096 |     # Politics
097 |     "election": ["vote", "poll", "ballot", "voting"],
098 |     "government": ["administration", "authority", "regime"],
099 |     "minister": ["secretary", "official"],
100 |     "parliament": ["legislature", "assembly"],
101 | 
102 |     # Education
103 |     "education": ["school", "university", "learning", "teaching"],
104 |     "student": ["pupil", "learner"],
105 |     "teacher": ["instructor", "educator"],
106 | 
107 |     # Economy
108 |     "economy": ["finance", "inflation", "market", "trade"],
109 |     "business": ["commerce", "trade", "industry"],
110 |     "price": ["cost", "rate", "tariff"],
111 | 
112 |     # Sports
113 |     "cricket": ["test", "odi", "t20"],
114 |     "football": ["soccer", "match"],
115 |     "player": ["athlete", "sportsman"],
116 | 
117 |     # Common
118 |     "news": ["report", "article", "story"],
119 |     "death": ["died", "killed", "deceased"],
120 |     "attack": ["assault", "strike"],
121 | }
122 | 
123 | EXPAND_BN = {
124 |     "নির্বাচন": ["ভোট", "পোল", "নির্বাচনী"],
125 |     "সরকার": ["প্রশাসন", "কর্তৃপক্ষ"],
126 |     "শিক্ষা": ["স্কুল", "বিশ্ববিদ্যালয়", "পড়াশোনা"],
127 |     "অর্থনীতি": ["ফিন্যান্স", "বাজার"],
128 |     "ক্রিকেট": ["টেস্ট", "ওডিআই", "টি-টুয়েন্টি"],
129 | }
130 | 
131 | def expand_query(q: str, lang: str) -> List[str]:
132 |     """Expand query with synonyms and related terms"""
133 |     expand_dict = EXPAND_BN if lang == "bn" else EXPAND_EN
134 | 
135 |     terms = (q or "").lower().split() if lang == "en" else (q or "").split()
136 |     expanded = []
137 | 
138 |     for term in terms:
139 |         if term in expand_dict:
140 |             expanded.extend(expand_dict[term])
141 | 
142 |     # Remove duplicates while preserving order
143 |     return list(dict.fromkeys(expanded))
144 | 
145 | 
146 | # -------------------------
147 | # Named Entity Mapping - IMPROVED
148 | # -------------------------
149 | NE_EN2BN = {
150 |     # Places
151 |     "bangladesh": "বাংলাদেশ",
152 |     "dhaka": "ঢাকা",
153 |     "chittagong": "চট্টগ্রাম",
154 |     "sylhet": "সিলেট",
155 |     "rajshahi": "রাজশাহী",
156 |     "khulna": "খুলনা",
157 |     "cox's bazar": "কক্সবাজার",
158 | 
159 |     # Political Figures
160 |     "sheikh hasina": "শেখ হাসিনা",
161 |     "khaleda zia": "খালেদা জিয়া",
162 |     "muhammad yunus": "মুহাম্মদ ইউনূস",
163 | 
164 |     # Organizations
165 |     "awami league": "আওয়ামী লীগ",
166 |     "bnp": "বিএনপি",
167 |     "jamaat": "জামায়াত",
168 | 
169 |     # Institutions
170 |     "dhaka university": "ঢাকা বিশ্ববিদ্যালয়",
171 |     "buet": "বুয়েট",
172 | }
173 | 
174 | NE_BN2EN = {v: k for k, v in NE_EN2BN.items()}
175 | 
176 | def map_named_entities(q: str, lang: str) -> List[Dict[str, str]]:
177 |     """Map named entities between languages"""
178 |     mapped = []
179 |     q_normalized = q.lower() if lang == "en" else q
180 | 
181 |     if lang == "en":
182 |         for en_term, bn_term in NE_EN2BN.items():
183 |             if en_term in q_normalized:
184 |                 mapped.append({"src": en_term, "tgt": bn_term})
185 |     else:
186 |         for bn_term, en_term in NE_BN2EN.items():
187 |             if bn_term in q_normalized:
188 |                 mapped.append({"src": bn_term, "tgt": en_term})
189 | 
190 |     return mapped
191 | 
192 | 
193 | # -------------------------
194 | # Translation Post-processing
195 | # -------------------------
196 | def postprocess_translation(q_original: str, q_translated: str, src_lang: str) -> str:
197 |     """Fix common translation errors"""
198 |     if src_lang == "bn":
199 |         # নির্বাচন should be "election" not "selection"
200 |         if "নির্বাচন" in q_original:
201 |             q_translated = q_translated.replace("selection", "election")
202 |         # ক্রিকেট should stay "cricket" not "criket"
203 |         if "ক্রিকেট" in q_original:
204 |             q_translated = q_translated.replace("criket", "cricket")
205 | 
206 |     return q_translated
207 | 
208 | 
209 | class QueryProcessor:
210 |     def __init__(self, use_stopwords: bool = False):
211 |         """
212 |         Initialize query processor
213 | 
214 |         Args:
215 |             use_stopwords: Whether to remove stopwords during normalization
216 |         """
217 |         self.translator = TranslatorOPUS()
218 |         self.use_stopwords = use_stopwords
219 | 
220 |     def process(self, raw_query: str) -> Dict:
221 |         """
222 |         Process query through complete pipeline
223 | 
224 |         Returns dict with:
225 |             - original, detected_lang, normalized
226 |             - q_en, q_bn (both languages always available)
227 |             - expansions, named_entity_mappings
228 |             - timing information
229 |         """
230 |         t0 = time.time()
231 | 
232 |         # Step 1: Language Detection
233 |         lang = detect_lang(raw_query)
234 | 
235 |         # Step 2: Normalization
236 |         q_norm = normalize_query(raw_query, lang, self.use_stopwords)
237 | 
238 |         # Step 3: Translation (with error handling)
239 |         try:
240 |             if lang == "bn":
241 |                 q_bn = q_norm
242 |                 q_en = self.translator.translate(q_norm, "bn", "en")
243 |                 q_en = postprocess_translation(q_bn, q_en, "bn")
244 |             else:
245 |                 q_en = q_norm
246 |                 q_bn = self.translator.translate(q_norm, "en", "bn")
247 |         except Exception as e:
248 |             # Fallback: no translation but don't crash
249 |             q_en = q_norm if lang == "en" else raw_query
250 |             q_bn = q_norm if lang == "bn" else raw_query
251 |             print(f"⚠️ Translation failed: {e}")
252 | 
253 |         # Step 4: Query Expansion
254 |         expansions = expand_query(q_norm, lang)
255 | 
256 |         # Step 5: Named Entity Mapping
257 |         ne_mappings = map_named_entities(q_norm, lang)
258 | 
259 |         total_ms = int((time.time() - t0) * 1000)
260 | 
261 |         return {
262 |             "original": raw_query,
263 |             "detected_lang": lang,
264 |             "normalized": q_norm,
265 |             "q_en": q_en,
266 |             "q_bn": q_bn,
267 |             "expansions": expansions,
268 |             "named_entity_mappings": ne_mappings,
269 |             "module_b_time_ms": total_ms,
270 |         }
271 | 
272 | 
273 | if __name__ == "__main__":
274 |     processor = QueryProcessor(use_stopwords=False)
275 | 
276 |     test_queries = [
277 |         "Bangladesh election results 2024",
278 |         "ঢাকায় ক্রিকেট খেলা",
279 |         "Sheikh Hasina news",
280 |         "শিক্ষা ব্যবস্থা সংস্কার",
281 |     ]
282 | 
283 |     print("="*60)
284 |     print("MODULE B - QUERY PROCESSING DEMO")
285 |     print("="*60)
286 | 
287 |     for query in test_queries:
288 |         print(f"\n📝 Query: {query}")
289 |         result = processor.process(query)
290 | 
291 |         print(f"   Language: {result['detected_lang']}")
292 |         print(f"   Normalized: {result['normalized']}")
293 |         print(f"   English: {result['q_en']}")
294 |         print(f"   Bangla: {result['q_bn']}")
295 | 
296 |         if result['expansions']:
297 |             print(f"   Expansions: {result['expansions']}")
298 | 
299 |         if result['named_entity_mappings']:
300 |             print(f"   NE Mappings: {result['named_entity_mappings']}")
301 | 
302 |         print(f"   Time: {result['module_b_time_ms']}ms")
```

**Line-by-line explanation**

- **L1:** Imports `re` for regular expressions (pattern matching / cleaning text).
- **L2:** Imports `time` for timing and delays (crawl politeness / profiling).
- **L3:** Imports `Dict, List, Tuple, Optional` from `typing`.
- **L4:** Imports `MarianMTModel, MarianTokenizer` from `transformers` for translation + NER pipelines (OPUS-MT / XLM-R).
- **L6:** Comment: -------------------------
- **L7:** Comment: Language Detection
- **L8:** Comment: -------------------------
- **L9:** Assigns a value to `BN_RANGE`.
- **L11:** Defines `detect_lang()` — Detects Bangla vs English using Unicode ranges (fast and reliable for BN/EN). This is the first step in Module B so the system knows which branch to run.
- **L12:** Executes this statement as part of the pipeline.
- **L13:** Returns the function output back to the caller.
- **L16:** Comment: -------------------------
- **L17:** Comment: Stopwords (Optional but Recommended)
- **L18:** Comment: -------------------------
- **L19:** Assigns a value to `STOPWORDS_EN`.
- **L20:** Executes this statement as part of the pipeline.
- **L21:** Executes this statement as part of the pipeline.
- **L22:** Executes this statement as part of the pipeline.
- **L24:** Assigns a value to `STOPWORDS_BN`.
- **L25:** Executes this statement as part of the pipeline.
- **L26:** Executes this statement as part of the pipeline.
- **L27:** Executes this statement as part of the pipeline.
- **L29:** Defines `remove_stopwords()` (a reusable function for the pipeline).
- **L30:** Executes this statement as part of the pipeline.
- **L31:** Assigns a value to `stopwords`.
- **L32:** Assigns a value to `words`.
- **L33:** Assigns a value to `filtered`.
- **L34:** Returns the function output back to the caller.
- **L37:** Comment: -------------------------
- **L38:** Comment: Normalization
- **L39:** Comment: -------------------------
- **L40:** Defines `normalize_query()` — Normalizes user queries (trim whitespace; lowercase only for English). Reduces noise and improves lexical retrieval.
- **L41:** Executes this statement as part of the pipeline.
- **L42:** Assigns a value to `q`.
- **L43:** Conditional branch: only runs the next indented block if the condition is true.
- **L44:** Assigns a value to `q`.
- **L45:** Conditional branch: only runs the next indented block if the condition is true.
- **L46:** Assigns a value to `q`.
- **L47:** Returns the function output back to the caller.
- **L50:** Comment: -------------------------
- **L51:** Comment: Translation (OPUS-MT)
- **L52:** Comment: -------------------------
- **L53:** Defines class `TranslatorOPUS` (a reusable component).
- **L54:** Executes this statement as part of the pipeline.
- **L55:** Executes this statement as part of the pipeline.
- **L56:** Executes this statement as part of the pipeline.
- **L57:** Executes this statement as part of the pipeline.
- **L58:** Executes this statement as part of the pipeline.
- **L59:** Defines `__init__()` (a reusable function for the pipeline).
- **L60:** Assigns a value to `self.cache`.
- **L61:** Assigns a value to `self.max_length`.
- **L63:** Defines `_load()` (a reusable function for the pipeline).
- **L64:** Conditional branch: only runs the next indented block if the condition is true.
- **L65:** Assigns a value to `tok`.
- **L66:** Assigns a value to `mdl`.
- **L67:** Assigns a value to `self.cache[model_name]`.
- **L68:** Returns the function output back to the caller.
- **L70:** Defines `_generate()` (a reusable function for the pipeline).
- **L71:** Assigns a value to `tok, mdl`.
- **L72:** Assigns a value to `batch`.
- **L73:** Assigns a value to `gen`.
- **L74:** Returns the function output back to the caller.
- **L76:** Defines `translate_bn_to_en()` (a reusable function for the pipeline).
- **L77:** Returns the function output back to the caller.
- **L79:** Defines `translate_en_to_bn()` (a reusable function for the pipeline).
- **L80:** Assigns a value to `text_with_lang`.
- **L81:** Assigns a value to `out`.
- **L82:** Returns the function output back to the caller.
- **L84:** Defines `translate()` (a reusable function for the pipeline).
- **L85:** Conditional branch: only runs the next indented block if the condition is true.
- **L86:** Returns the function output back to the caller.
- **L87:** Conditional branch: only runs the next indented block if the condition is true.
- **L88:** Returns the function output back to the caller.
- **L89:** Executes this statement as part of the pipeline.
- **L92:** Comment: -------------------------
- **L93:** Comment: Query Expansion - IMPROVED
- **L94:** Comment: -------------------------
- **L95:** Assigns a value to `EXPAND_EN`.
- **L96:** Comment: Politics
- **L97:** Executes this statement as part of the pipeline.
- **L98:** Executes this statement as part of the pipeline.
- **L99:** Executes this statement as part of the pipeline.
- **L100:** Executes this statement as part of the pipeline.
- **L102:** Comment: Education
- **L103:** Executes this statement as part of the pipeline.
- **L104:** Executes this statement as part of the pipeline.
- **L105:** Executes this statement as part of the pipeline.
- **L107:** Comment: Economy
- **L108:** Executes this statement as part of the pipeline.
- **L109:** Executes this statement as part of the pipeline.
- **L110:** Executes this statement as part of the pipeline.
- **L112:** Comment: Sports
- **L113:** Executes this statement as part of the pipeline.
- **L114:** Executes this statement as part of the pipeline.
- **L115:** Executes this statement as part of the pipeline.
- **L117:** Comment: Common
- **L118:** Executes this statement as part of the pipeline.
- **L119:** Executes this statement as part of the pipeline.
- **L120:** Executes this statement as part of the pipeline.
- **L121:** Executes this statement as part of the pipeline.
- **L123:** Assigns a value to `EXPAND_BN`.
- **L124:** Executes this statement as part of the pipeline.
- **L125:** Executes this statement as part of the pipeline.
- **L126:** Executes this statement as part of the pipeline.
- **L127:** Executes this statement as part of the pipeline.
- **L128:** Executes this statement as part of the pipeline.
- **L129:** Executes this statement as part of the pipeline.
- **L131:** Defines `expand_query()` — Adds a small set of synonyms / related terms to reduce lexical mismatch (recommended Module B step).
- **L132:** Executes this statement as part of the pipeline.
- **L133:** Assigns a value to `expand_dict`.
- **L135:** Assigns a value to `terms`.
- **L136:** Assigns a value to `expanded`.
- **L138:** Starts a loop to process items one-by-one.
- **L139:** Conditional branch: only runs the next indented block if the condition is true.
- **L140:** Executes this statement as part of the pipeline.
- **L142:** Comment: Remove duplicates while preserving order
- **L143:** Returns the function output back to the caller.
- **L146:** Comment: -------------------------
- **L147:** Comment: Named Entity Mapping - IMPROVED
- **L148:** Comment: -------------------------
- **L149:** Assigns a value to `NE_EN2BN`.
- **L150:** Comment: Places
- **L151:** Executes this statement as part of the pipeline.
- **L152:** Executes this statement as part of the pipeline.
- **L153:** Executes this statement as part of the pipeline.
- **L154:** Executes this statement as part of the pipeline.
- **L155:** Executes this statement as part of the pipeline.
- **L156:** Executes this statement as part of the pipeline.
- **L157:** Executes this statement as part of the pipeline.
- **L159:** Comment: Political Figures
- **L160:** Executes this statement as part of the pipeline.
- **L161:** Executes this statement as part of the pipeline.
- **L162:** Executes this statement as part of the pipeline.
- **L164:** Comment: Organizations
- **L165:** Executes this statement as part of the pipeline.
- **L166:** Executes this statement as part of the pipeline.
- **L167:** Executes this statement as part of the pipeline.
- **L169:** Comment: Institutions
- **L170:** Executes this statement as part of the pipeline.
- **L171:** Executes this statement as part of the pipeline.
- **L172:** Executes this statement as part of the pipeline.
- **L174:** Assigns a value to `NE_BN2EN`.
- **L176:** Defines `map_named_entities()` — Maps known named entities across languages (e.g., Dhaka ↔ ঢাকা). This is crucial because NEs often fail under translation.
- **L177:** Executes this statement as part of the pipeline.
- **L178:** Assigns a value to `mapped`.
- **L179:** Assigns a value to `q_normalized`.
- **L181:** Conditional branch: only runs the next indented block if the condition is true.
- **L182:** Starts a loop to process items one-by-one.
- **L183:** Conditional branch: only runs the next indented block if the condition is true.
- **L184:** Executes this statement as part of the pipeline.
- **L185:** Fallback branch if none of the previous conditions matched.
- **L186:** Starts a loop to process items one-by-one.
- **L187:** Conditional branch: only runs the next indented block if the condition is true.
- **L188:** Executes this statement as part of the pipeline.
- **L190:** Returns the function output back to the caller.
- **L193:** Comment: -------------------------
- **L194:** Comment: Translation Post-processing
- **L195:** Comment: -------------------------
- **L196:** Defines `postprocess_translation()` (a reusable function for the pipeline).
- **L197:** Executes this statement as part of the pipeline.
- **L198:** Conditional branch: only runs the next indented block if the condition is true.
- **L199:** Comment: নির্বাচন should be "election" not "selection"
- **L200:** Conditional branch: only runs the next indented block if the condition is true.
- **L201:** Assigns a value to `q_translated`.
- **L202:** Comment: ক্রিকেট should stay "cricket" not "criket"
- **L203:** Conditional branch: only runs the next indented block if the condition is true.
- **L204:** Assigns a value to `q_translated`.
- **L206:** Returns the function output back to the caller.
- **L209:** Defines class `QueryProcessor` (a reusable component).
- **L210:** Defines `__init__()` (a reusable function for the pipeline).
- **L211:** Executes this statement as part of the pipeline.
- **L212:** Executes this statement as part of the pipeline.
- **L214:** Executes this statement as part of the pipeline.
- **L215:** Executes this statement as part of the pipeline.
- **L216:** Executes this statement as part of the pipeline.
- **L217:** Assigns a value to `self.translator`.
- **L218:** Assigns a value to `self.use_stopwords`.
- **L220:** Defines `process()` (a reusable function for the pipeline).
- **L221:** Executes this statement as part of the pipeline.
- **L222:** Executes this statement as part of the pipeline.
- **L224:** Executes this statement as part of the pipeline.
- **L225:** Executes this statement as part of the pipeline.
- **L226:** Executes this statement as part of the pipeline.
- **L227:** Executes this statement as part of the pipeline.
- **L228:** Executes this statement as part of the pipeline.
- **L229:** Executes this statement as part of the pipeline.
- **L230:** Assigns a value to `t0`.
- **L232:** Comment: Step 1: Language Detection
- **L233:** Assigns a value to `lang`.
- **L235:** Comment: Step 2: Normalization
- **L236:** Assigns a value to `q_norm`.
- **L238:** Comment: Step 3: Translation (with error handling)
- **L239:** Executes this statement as part of the pipeline.
- **L240:** Conditional branch: only runs the next indented block if the condition is true.
- **L241:** Assigns a value to `q_bn`.
- **L242:** Assigns a value to `q_en`.
- **L243:** Assigns a value to `q_en`.
- **L244:** Fallback branch if none of the previous conditions matched.
- **L245:** Assigns a value to `q_en`.
- **L246:** Assigns a value to `q_bn`.
- **L247:** Executes this statement as part of the pipeline.
- **L248:** Comment: Fallback: no translation but don't crash
- **L249:** Assigns a value to `q_en`.
- **L250:** Assigns a value to `q_bn`.
- **L251:** Executes this statement as part of the pipeline.
- **L253:** Comment: Step 4: Query Expansion
- **L254:** Assigns a value to `expansions`.
- **L256:** Comment: Step 5: Named Entity Mapping
- **L257:** Assigns a value to `ne_mappings`.
- **L259:** Assigns a value to `total_ms`.
- **L261:** Returns the function output back to the caller.
- **L262:** Executes this statement as part of the pipeline.
- **L263:** Executes this statement as part of the pipeline.
- **L264:** Executes this statement as part of the pipeline.
- **L265:** Executes this statement as part of the pipeline.
- **L266:** Executes this statement as part of the pipeline.
- **L267:** Executes this statement as part of the pipeline.
- **L268:** Executes this statement as part of the pipeline.
- **L269:** Executes this statement as part of the pipeline.
- **L270:** Executes this statement as part of the pipeline.
- **L273:** Conditional branch: only runs the next indented block if the condition is true.
- **L274:** Assigns a value to `processor`.
- **L276:** Assigns a value to `test_queries`.
- **L277:** Executes this statement as part of the pipeline.
- **L278:** Executes this statement as part of the pipeline.
- **L279:** Executes this statement as part of the pipeline.
- **L280:** Executes this statement as part of the pipeline.
- **L281:** Executes this statement as part of the pipeline.
- **L283:** Assigns a value to `print("`.
- **L284:** Executes this statement as part of the pipeline.
- **L285:** Assigns a value to `print("`.
- **L287:** Starts a loop to process items one-by-one.
- **L288:** Executes this statement as part of the pipeline.
- **L289:** Assigns a value to `result`.
- **L291:** Executes this statement as part of the pipeline.
- **L292:** Executes this statement as part of the pipeline.
- **L293:** Executes this statement as part of the pipeline.
- **L294:** Executes this statement as part of the pipeline.
- **L296:** Conditional branch: only runs the next indented block if the condition is true.
- **L297:** Executes this statement as part of the pipeline.
- **L299:** Conditional branch: only runs the next indented block if the condition is true.
- **L300:** Executes this statement as part of the pipeline.
- **L302:** Executes this statement as part of the pipeline.


## Practical usage tips

- Run cells **top-to-bottom** (Colab state matters).
- If a cell installs packages (`!pip ...`), run it once and then restart runtime only if you get version conflicts.
- Keep your datasets under a consistent folder (e.g., `data/processed/`) so Module C/D can load them without edits.


## Quick fixes (copy/paste)

- This notebook contains an **“improved pipeline” copy** later in the file (it redefines functions like `detect_lang`). That is OK, but remember: later definitions override earlier ones.
- Translation models can be slow on CPU. For quick demos, keep queries short.
