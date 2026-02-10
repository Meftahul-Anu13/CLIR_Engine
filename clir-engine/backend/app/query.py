from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Dict, List, Tuple

import torch
from transformers import MarianMTModel, MarianTokenizer

BN_RANGE = re.compile(r"[\u0980-\u09FF]")

EXPANSION_EN: Dict[str, List[str]] = {
    "election": ["vote", "polls", "ballot"],
    "education": ["school", "university", "teachers"],
    "economy": ["growth", "inflation", "gdp"],
    "sports": ["cricket", "football", "match"],
    "climate": ["weather", "cyclone", "disaster"],
}

STOPWORDS_EN = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "and",
    "to",
    "is",
    "are",
}

NE_EN_TO_BN = {
    "sheikh hasina": "শেখ হাসিনা",
    "khaleda zia": "খালেদা জিয়া",
    "dhaka": "ঢাকা",
    "chittagong": "চট্টগ্রাম",
    "padma bridge": "পদ্মা সেতু",
}
NE_BN_TO_EN = {bn: en for en, bn in NE_EN_TO_BN.items()}

MODEL_MAP = {
    ("bn", "en"): "Helsinki-NLP/opus-mt-bn-en",
    ("en", "bn"): "Helsinki-NLP/opus-mt-en-bn",
}


def detect_language(text: str) -> str:
    if BN_RANGE.search(text or ""):
        return "bn"
    return "en"


def normalize_query(query: str, lang: str, remove_stopwords: bool = False) -> str:
    q = " ".join((query or "").strip().split())
    if lang == "en":
        q = q.lower()
        if remove_stopwords:
            q = " ".join([tok for tok in q.split() if tok not in STOPWORDS_EN])
    return q


def expand_query(query: str, lang: str) -> List[str]:
    expansions: List[str] = []
    if lang == "en":
        for token in query.split():
            expansions.extend(EXPANSION_EN.get(token, []))
    return sorted(set(expansions))


def map_named_entities(query: str, lang: str) -> List[Dict[str, str]]:
    mapped: List[Dict[str, str]] = []
    q = query.lower()
    if lang == "en":
        for en, bn in NE_EN_TO_BN.items():
            if en in q:
                mapped.append({"src": en, "tgt": bn})
    else:
        for bn, en in NE_BN_TO_EN.items():
            if bn in q:
                mapped.append({"src": bn, "tgt": en})
    return mapped


class MarianTranslator:
    """Lazy loader for Marian MT models with a lightweight cache."""

    def __init__(self):
        self.models: Dict[Tuple[str, str], Tuple[MarianTokenizer, MarianMTModel]] = {}

    def _load(self, src: str, tgt: str) -> Tuple[MarianTokenizer, MarianMTModel]:
        key = (src, tgt)
        if key not in MODEL_MAP:
            raise ValueError(f"Unsupported translation pair: {src}->{tgt}")
        if key not in self.models:
            model_name = MODEL_MAP[key]
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            self.models[key] = (tokenizer, model)
        return self.models[key]

    def translate(self, text: str, src: str, tgt: str) -> str:
        tokenizer, model = self._load(src, tgt)
        encoded = tokenizer([text], return_tensors="pt", truncation=True)
        with torch.no_grad():
            generated = model.generate(**encoded, max_length=96)
        return tokenizer.decode(generated[0], skip_special_tokens=True).strip()


TRANSLATOR = MarianTranslator()


@lru_cache(maxsize=512)
def cached_translate(text: str, src: str, tgt: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    return TRANSLATOR.translate(cleaned, src, tgt)


class QueryProcessor:
    """Full Module B pipeline."""

    def __init__(self, remove_stopwords: bool = False):
        self.remove_stopwords = remove_stopwords

    def process(self, raw_query: str) -> Dict:
        start = time.time()
        detected_lang = detect_language(raw_query)
        normalized = normalize_query(raw_query, detected_lang, self.remove_stopwords)

        translation_start = time.time()
        q_en = normalized
        q_bn = normalized
        try:
            if detected_lang == "en":
                q_bn = cached_translate(normalized, "en", "bn")
            else:
                q_en = cached_translate(normalized, "bn", "en")
        except Exception:
            # keep normalized text on both channels if translation fails
            q_en = normalized
            q_bn = normalized
        translation_ms = int((time.time() - translation_start) * 1000)

        expansions = expand_query(normalized, detected_lang)
        named_entities = map_named_entities(normalized, detected_lang)

        # inject named entities into translated side for better recall
        for mapping in named_entities:
            if detected_lang == "en":
                q_bn = f"{q_bn} {mapping['tgt']}"
            else:
                q_en = f"{q_en} {mapping['tgt']}"

        timing_total = int((time.time() - start) * 1000)

        return {
            "original": raw_query,
            "detected_lang": detected_lang,
            "normalized": normalized,
            "query_variants": {
                "en": q_en.strip(),
                "bn": q_bn.strip(),
            },
            "expansions": expansions,
            "named_entities": named_entities,
            "timing": {
                "total_ms": timing_total,
                "translation_ms": translation_ms,
            },
        }
