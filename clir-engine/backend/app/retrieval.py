from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from rapidfuzz import fuzz
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from .dataset import DatasetStore, Document
from .index_store import DEFAULT_MODEL, IndexStore
from .ranking import fuse_scores, normalize_channel


def simple_tokenize(text: str) -> List[str]:
    return [tok for tok in (text or "").split() if tok]


class BM25Retriever:
    def __init__(self, documents: Sequence[Document]):
        self.languages = [doc.language for doc in documents]
        self.corpus_tokens = [simple_tokenize(doc.text) for doc in documents]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(
        self,
        query: str,
        topk: int,
        allowed_languages: Optional[Set[str]] = None,
    ) -> Dict[int, float]:
        if not query.strip():
            return {}
        scores = self.bm25.get_scores(simple_tokenize(query))
        return _select_top(scores, topk, self.languages, allowed_languages)


class TFIDFRetriever:
    def __init__(self, documents: Sequence[Document]):
        self.languages = [doc.language for doc in documents]
        texts = [doc.text for doc in documents]
        self.vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1, 2), max_features=250_000)
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(
        self,
        query: str,
        topk: int,
        allowed_languages: Optional[Set[str]] = None,
    ) -> Dict[int, float]:
        if not query.strip():
            return {}
        q_vec = self.vectorizer.transform([query])
        scores = (self.matrix @ q_vec.T).toarray().ravel()
        return _select_top(scores, topk, self.languages, allowed_languages)


class SemanticRetriever:
    def __init__(self, dataset: DatasetStore, index_store: IndexStore, model_name: str = DEFAULT_MODEL):
        self.dataset = dataset
        self.index_store = index_store
        self.languages = dataset.languages_list()
        self.model = SentenceTransformer(model_name)
        self.index_store.ensure_ready()

    def search(
        self,
        query: str,
        topk: int,
        allowed_languages: Optional[Set[str]] = None,
    ) -> Dict[int, float]:
        if not query.strip():
            return {}
        q_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        distances, indices = self.index_store.index.search(q_vec, topk)
        scores = distances[0]
        idxs = indices[0]
        results: Dict[int, float] = {}
        for doc_id, score in zip(idxs, scores):
            if doc_id < 0:
                continue
            lang = self.languages[doc_id]
            if allowed_languages and lang not in allowed_languages:
                continue
            results[int(doc_id)] = float((score + 1.0) / 2.0)  # map cosine [-1,1] -> [0,1]
        return results


class FuzzyScorer:
    def __init__(self, documents: Sequence[Document]):
        self.documents = documents

    def score_candidates(self, queries: Sequence[str], candidate_ids: Iterable[int]) -> Dict[int, float]:
        normalized_queries = [q for q in queries if q]
        if not normalized_queries:
            return {}
        scores: Dict[int, float] = {}
        for doc_id in candidate_ids:
            doc = self.documents[doc_id]
            haystack = f"{doc.title} {doc.body[:400]}"
            best = 0.0
            for q in normalized_queries:
                best = max(best, fuzz.token_set_ratio(q, haystack) / 100.0)
            scores[doc_id] = best
        return scores


def _select_top(
    scores: Sequence[float],
    topk: int,
    languages: Sequence[str],
    allowed_languages: Optional[Set[str]],
) -> Dict[int, float]:
    if not len(scores):
        return {}
    order = np.argsort(-np.array(scores))
    results: Dict[int, float] = {}
    for idx in order:
        lang = languages[int(idx)]
        if allowed_languages and lang not in allowed_languages:
            continue
        value = float(scores[int(idx)])
        results[int(idx)] = value
        if len(results) >= topk:
            break
    return results


class RetrievalManager:
    def __init__(
        self,
        dataset: DatasetStore,
        index_store: IndexStore,
        lexical_topn: int = 40,
        semantic_topn: int = 60,
    ):
        self.dataset = dataset
        self.bm25 = BM25Retriever(dataset.documents)
        self.tfidf = TFIDFRetriever(dataset.documents)
        self.semantic = SemanticRetriever(dataset, index_store)
        self.fuzzy = FuzzyScorer(dataset.documents)
        self.lexical_topn = lexical_topn
        self.semantic_topn = semantic_topn
        self.weights = {
            "bm25": 0.3,
            "tfidf": 0.2,
            "semantic": 0.4,
            "fuzzy": 0.1,
        }

    def search(
        self,
        query_bundle: Dict,
        language_filter: str,
        k: int,
        include_internal_ids: bool = False,
    ) -> Dict:
        allowed = None if language_filter == "all" else {language_filter}
        queries = list({v for v in query_bundle["query_variants"].values() if v})

        lexical_start = time.time()
        bm25_scores = self._run_model(self.bm25, queries, self.lexical_topn, allowed)
        tfidf_scores = self._run_model(self.tfidf, queries, self.lexical_topn, allowed)
        lexical_ms = int((time.time() - lexical_start) * 1000)

        semantic_start = time.time()
        semantic_scores = self._run_model(self.semantic, queries, self.semantic_topn, allowed)
        semantic_ms = int((time.time() - semantic_start) * 1000)

        candidate_ids = set(bm25_scores) | set(tfidf_scores) | set(semantic_scores)

        fuzzy_start = time.time()
        fuzzy_scores = self.fuzzy.score_candidates(queries, candidate_ids)
        fuzzy_ms = int((time.time() - fuzzy_start) * 1000)

        ranking_start = time.time()
        normalized = {
            "bm25": normalize_channel(bm25_scores),
            "tfidf": normalize_channel(tfidf_scores),
            "semantic": normalize_channel(semantic_scores),
            "fuzzy": normalize_channel(fuzzy_scores),
        }
        fused = fuse_scores(normalized, self.weights)
        ranking_ms = int((time.time() - ranking_start) * 1000)

        ordered_ids = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        results = []
        rank = 1
        for doc_id, score in ordered_ids:
            doc = self.dataset.get_document(doc_id)
            if allowed and doc.language not in allowed:
                continue
            snippet = build_snippet(doc.body or doc.title or "")
            record = {
                "rank": rank,
                "score": round(float(score), 4),
                "title": doc.title,
                "url": doc.url,
                "date": doc.date,
                "language": doc.language,
                "source": doc.source or "",
                "snippet": snippet,
                "debug": {
                    "bm25": round(normalized["bm25"].get(doc_id, 0.0), 4),
                    "tfidf": round(normalized["tfidf"].get(doc_id, 0.0), 4),
                    "semantic": round(normalized["semantic"].get(doc_id, 0.0), 4),
                    "fuzzy": round(normalized["fuzzy"].get(doc_id, 0.0), 4),
                },
            }
            if include_internal_ids:
                record["_doc_id"] = doc_id
            results.append(record)
            rank += 1
            if len(results) >= k:
                break

        return {
            "results": results,
            "timing_ms": {
                "lexical": lexical_ms,
                "semantic": semantic_ms,
                "fuzzy": fuzzy_ms,
                "ranking": ranking_ms,
            },
        }

    def _run_model(
        self,
        model,
        queries: List[str],
        topn: int,
        allowed_languages: Optional[Set[str]],
    ) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        for query in queries:
            hits = model.search(query, topk=topn, allowed_languages=allowed_languages)
            for doc_id, score in hits.items():
                if doc_id not in scores or score > scores[doc_id]:
                    scores[doc_id] = score
        return scores


def build_snippet(text: str, max_words: int = 60) -> str:
    tokens = text.split()
    if len(tokens) <= max_words:
        return text.strip()
    return " ".join(tokens[:max_words]).strip() + "…"
