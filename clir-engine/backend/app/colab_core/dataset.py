from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class Document:
    doc_id: int
    title: str
    body: str
    url: str
    date: Optional[str]
    language: str
    source: Optional[str]

    @property
    def text(self) -> str:
        title = self.title or ""
        body = self.body or ""
        merged = f"{title.strip()} {body.strip()}".strip()
        return merged

    def as_result(self) -> Dict[str, Optional[str]]:
        return {
            "title": self.title or "",
            "url": self.url or "",
            "date": self.date,
            "language": self.language,
            "source": self.source or "",
            "body": self.body or "",
        }


class DatasetStore:
    """Loads processed news datasets and provides helpers for retrieval."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.documents: List[Document] = []
        self._language_counts: Dict[str, int] = {}
        self._load()

    def _read_jsonl(self, path: Path) -> List[Dict]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file missing: {path}")

        rows: List[Dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def _load(self) -> None:
        datasets = [
            (self.data_dir / "english_news.jsonl", "en"),
            (self.data_dir / "bangla_news.jsonl", "bn"),
        ]

        next_doc_id = 0
        for path, default_lang in datasets:
            rows = self._read_jsonl(path)
            for rec in rows:
                language = (rec.get("language") or rec.get("lang") or default_lang or "").strip().lower()
                if language not in {"en", "bn"}:
                    language = default_lang
                doc = Document(
                    doc_id=next_doc_id,
                    title=rec.get("title", "") or "",
                    body=rec.get("body", "") or "",
                    url=rec.get("url", "") or "",
                    date=rec.get("date"),
                    language=language,
                    source=rec.get("source"),
                )
                self.documents.append(doc)
                lang = doc.language or "unknown"
                self._language_counts[lang] = self._language_counts.get(lang, 0) + 1
                next_doc_id += 1

        if not self.documents:
            raise RuntimeError("No documents loaded. Ensure processed JSONL files exist.")

    @property
    def size(self) -> int:
        return len(self.documents)

    def language_counts(self) -> Dict[str, int]:
        return dict(self._language_counts)

    def get_document(self, doc_id: int) -> Document:
        return self.documents[doc_id]

    def iter_documents(self) -> Iterable[Document]:
        return iter(self.documents)

    def texts(self) -> List[str]:
        return [doc.text for doc in self.documents]

    def languages_list(self) -> List[str]:
        return [doc.language for doc in self.documents]
