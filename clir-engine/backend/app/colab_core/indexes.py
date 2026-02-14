from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

from .dataset import DatasetStore
from .config_loader import get_config


class IndexStore:
    def __init__(self, dataset: DatasetStore, storage_dir: Path):
        self.dataset = dataset
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.emb_path = self.storage_dir / "doc_emb.npy"
        self.index_path = self.storage_dir / "faiss.index"
        self.fingerprint_path = self.storage_dir / "doc_fingerprint.json"
        cfg = get_config()
        self.model_name = cfg["semantic_model"]
        self._embeddings: Optional[np.ndarray] = None
        self._index: Optional[faiss.IndexFlatIP] = None

    def ready(self) -> bool:
        return (
            self.emb_path.exists()
            and self.index_path.exists()
            and self.fingerprint_path.exists()
        )

    def load(self) -> Tuple[np.ndarray, faiss.IndexFlatIP]:
        if not self.ready():
            raise FileNotFoundError("Embeddings/index missing; run build_indexes.py")
        if self._embeddings is None:
            self._embeddings = np.load(self.emb_path)
        if self._index is None:
            self._index = faiss.read_index(str(self.index_path))  # type: ignore[arg-type]
        return self._embeddings, self._index

    def ensure_ready(self) -> Tuple[np.ndarray, faiss.IndexFlatIP]:
        needs_build = False
        if not self.ready():
            needs_build = True
        elif not self._fingerprint_matches():
            needs_build = True
        if needs_build:
            self.build_and_persist()
        return self.load()

    def build_and_persist(self, batch_size: int = 128) -> None:
        texts = self.dataset.texts()
        if not texts:
            raise RuntimeError("No texts available for embedding.")

        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = embeddings.astype("float32")
        dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        np.save(self.emb_path, embeddings)
        faiss.write_index(index, str(self.index_path))
        self._write_fingerprint(self._dataset_fingerprint())

        self._embeddings = embeddings
        self._index = index

    def _dataset_fingerprint(self) -> Dict[str, str]:
        docs = self.dataset.documents
        cfg = get_config()
        return {
            "count": len(docs),
            "first_url": docs[0].url if docs else "",
            "last_url": docs[-1].url if docs else "",
            "model": self.model_name,
            "config_version": cfg.get("config_version", 0),
        }

    def _write_fingerprint(self, data: Dict[str, str]) -> None:
        with self.fingerprint_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle)

    def _fingerprint_matches(self) -> bool:
        try:
            existing = json.loads(self.fingerprint_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        current = self._dataset_fingerprint()
        return (
            existing.get("count") == current["count"]
            and existing.get("first_url") == current["first_url"]
            and existing.get("last_url") == current["last_url"]
            and existing.get("model") == current["model"]
            and existing.get("config_version") == current["config_version"]
        )

    def current_fingerprint(self) -> Dict[str, str]:
        if self.fingerprint_path.exists():
            try:
                return json.loads(self.fingerprint_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return self._dataset_fingerprint()

    @property
    def embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            self.load()
        assert self._embeddings is not None
        return self._embeddings

    @property
    def index(self) -> faiss.IndexFlatIP:
        if self._index is None:
            self.load()
        assert self._index is not None
        return self._index
