# Module B — Query Processing & Cross‑Lingual Handling


## How to run
Run cells from top to bottom. Downstream cells depend on artifacts produced earlier, so the notebook order is part of the pipeline.

## Datasets used
This project uses a multilingual Bangla–English news dataset with a shared schema:

- `title`, `body`, `url`, `date`, `language` (required by the assignment)
- `text = title + body`

In our Colab runs, we used the following datasets (row counts shown for your reporting):

- 3000 documents (JSON/JSONL)
- 2999 documents (JSON/JSONL)
- 501 documents (CSV)
- 397 documents (CSV)
- 100 documents (CSV)
- 12 documents (CSV)

> Tip: Keep your processed Bangla and English corpora in *JSONL* for streaming-friendly loading and easier debugging.

## Outputs you should expect
- A `QueryProcessor` (or equivalent) that outputs both `q_en` and `q_bn`.
- Optional expansions and NE mappings for debugging and error analysis.
- Timing info for translation and preprocessing (useful for Module D reporting).

## Cell-by-cell walkthrough

### Cell 1: pip install -q spacy
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

### Cell 2: pip -q install sacremoses
*Type:* `code`

**Purpose**

- Installs the Python libraries required for preprocessing, translation, retrieval, and evaluation in this notebook.

**Key snippet**

```python
!pip -q install sacremoses
```

### Cell 3: Language detection (Bangla vs English)
*Type:* `markdown`

**Context**

# Language detection (Bangla vs English)

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 4: import re
*Type:* `code`

**Purpose**

- Detects whether a query is Bangla or English (usually via Unicode range checks).

**Key snippet**

```python
import re
BN_RANGE = re.compile(r'[\u0980-\u09FF]')
def detect_lang(query: str) -> str:
    q = query.strip()
    if BN_RANGE.search(q):
        return "bn"
    # fallback: assume english if mostly latin
    return "en"
```

### Cell 5: Normalization (required)
*Type:* `markdown`

**Context**

# Normalization (required)

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 6: def normalize_query(query: str, lang: str) -> str:
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
def normalize_query(query: str, lang: str) -> str:
    q = " ".join(query.strip().split())  # collapse whitespace
    if lang == "en":
        q = q.lower()
    # For Bangla, lowercasing doesn't matter; keep as-is
    return q
```

### Cell 7: Query translation / conversion
*Type:* `markdown`

**Context**

# Query translation / conversion

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 8: from transformers import MarianMTModel, MarianTokenizer
*Type:* `code`

**Purpose**

- Translates queries between Bangla and English (OPUS-MT or Google Translate). This is the core CLIR bridge.

**Key snippet**

```python
from transformers import MarianMTModel, MarianTokenizer
class OpusTranslator:
    """
    bn -> en : Helsinki-NLP/opus-mt-bn-en
    en -> bn : Helsinki-NLP/opus-mt-en-iir with required >>ben<< token
    """
    def __init__(self):
        self.cache = {}
```

**Gotchas / tips**

- Translation is a common failure source; log both original and translated queries for error analysis.
- Online translation can fail; the pipeline includes a fallback path so execution continues without crashing.

### Cell 9: Query expansion
*Type:* `markdown`

**Context**

# Query expansion

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 10: EXPAND_DICT_EN = {
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
EXPAND_DICT_EN = {
    "election": ["vote", "poll", "ballot"],
    "education": ["school", "university", "student"],
}
def expand_query(q: str, lang: str):
    terms = q.split()
    extra = []
    if lang == "en":
```

### Cell 11: Named Entity extraction + mapping
*Type:* `markdown`

**Context**

# Named Entity extraction + mapping

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 12: `Layer 1 `
*Type:* `markdown`

**Context**

`Layer 1 `

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 13: NE_MAP_EN2BN = {
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
NE_MAP_EN2BN = {
    "bangladesh": "বাংলাদেশ",
    "dhaka": "ঢাকা",
    "sheikh hasina": "শেখ হাসিনা",
}
NE_MAP_BN2EN = {v: k for k, v in NE_MAP_EN2BN.items()}
def map_named_entities(q: str, lang: str):
    q_low = q.lower()
```

### Cell 14: `Layer 2 `
*Type:* `markdown`

**Context**

`Layer 2 `

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 15: import spacy
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
import spacy
nlp_en = spacy.load("en_core_web_sm")
def extract_ner_en(text: str):
    doc = nlp_en(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
```

### Cell 16: from transformers import pipeline
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
from transformers import pipeline
ner_bn = pipeline(
    "ner",
    model="Davlan/xlm-roberta-base-ner-hrl",
    aggregation_strategy="simple"
)
```

### Cell 17: def extract_ner_bn(text: str):
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
def extract_ner_bn(text: str):
    results = ner_bn(text)
    entities = []
    for r in results:
        entities.append({
            "text": r["word"],
            "label": r["entity_group"]  # PER, LOC, ORG
        })
```

### Cell 18: NE_MAP_EN2BN = {
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
NE_MAP_EN2BN = {
    "bangladesh": "বাংলাদেশ",
    "dhaka": "ঢাকা",
    "sheikh hasina": "শেখ হাসিনা",
    "awami league": "আওয়ামী লীগ",
}
NE_MAP_BN2EN = {v: k for k, v in NE_MAP_EN2BN.items()}
def map_entities(entities, lang):
```

### Cell 19: Pipeline
*Type:* `markdown`

**Context**

# Pipeline

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 20: import time
*Type:* `code`

**Purpose**

- Detects whether a query is Bangla or English (usually via Unicode range checks).

**Key snippet**

```python
import time
def process_query(query: str):
    t0 = time.time()
    lang = detect_lang(query)
    q_norm = normalize_query(query, lang)
    t_trans0 = time.time()
    q_src, q_other = translate_query(q_norm, lang)
    trans_ms = int((time.time() - t_trans0) * 1000)
```

### Cell 21: process_query("বাংলাদেশ নির্বাচন")
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
process_query("বাংলাদেশ নির্বাচন")
```

### Cell 22: process_query("Bangladesh election")
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
process_query("Bangladesh election")
```

### Cell 23: queries = [
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
queries = [
    "বাংলাদেশ নির্বাচন",
    "Bangladesh election",
    "ঢাকা traffic problem",
    "বাংলাদেশ election results",
    "চেয়ার"
]
for q in queries:
```

### Cell 24: LATIN_RANGE = re.compile(r'[A-Za-z]')
*Type:* `code`

**Purpose**

- Detects whether a query is Bangla or English (usually via Unicode range checks).

**Key snippet**

```python
LATIN_RANGE = re.compile(r'[A-Za-z]')
BN_RANGE = re.compile(r'[\u0980-\u09FF]')
def is_mixed_script(text: str) -> bool:
    return bool(BN_RANGE.search(text)) and bool(LATIN_RANGE.search(text))
def split_mixed(text: str):
    bn_part = " ".join(re.findall(r'[\u0980-\u09FF]+', text))
    en_part = " ".join(re.findall(r'[A-Za-z]+', text))
    return bn_part.strip(), en_part.strip()
```

### Cell 25: def postprocess_bn_to_en(q_bn: str, q_en: str) -> str:
*Type:* `code`

**Purpose**

- Runs a part of the pipeline for this module.

**Key snippet**

```python
def postprocess_bn_to_en(q_bn: str, q_en: str) -> str:
    if "নির্বাচন" in q_bn:
        q_en = q_en.replace("selection", "election")
    return q_en
def postprocess_en_to_bn(q_en: str, q_bn: str) -> str:
    # If election appears, ensure Bangla contains নির্বাচন (not নিষেধাজ্ঞা নির্বাচন)
    if "election" in q_en and "নির্বাচন" not in q_bn:
        q_bn = q_bn.replace("নিষেধাজ্ঞা", "").strip()
```

### Cell 26: Testing
*Type:* `markdown`

**Context**

#Testing

This cell provides narrative structure and records key links or section boundaries for reproducibility.

This cell provides narrative structure and records key links or section boundaries for reproducibility.

### Cell 27: import re
*Type:* `code`

**Purpose**

- Detects whether a query is Bangla or English (usually via Unicode range checks).
- Translates queries between Bangla and English (OPUS-MT or Google Translate). This is the core CLIR bridge.

**Key snippet**

```python
import re
import time
from typing import Dict, List, Tuple, Optional
from transformers import MarianMTModel, MarianTokenizer
# -------------------------
# Language Detection
# -------------------------
BN_RANGE = re.compile(r"[\u0980-\u09FF]")
```


