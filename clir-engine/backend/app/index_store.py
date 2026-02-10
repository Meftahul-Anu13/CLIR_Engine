from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

from .dataset import DatasetStore

EMBED_STORAGE = "doc_emb.npy"
FAISS_STORAGE = "faiss.index"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class IndexStore:
    def __init__(self, dataset: DatasetStore, storage_dir: Path):
        self.dataset = dataset
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.emb_path = self.storage_dir / EMBED_STORAGE
        self.index_path = self.storage_dir / FAISS_STORAGE
        self.model_name = DEFAULT_MODEL
        self._embeddings: Optional[np.ndarray] = None
        self._index: Optional[faiss.IndexFlatIP] = None

    # ---------------------------------------------------------------- utilities
    def ready(self) -> bool:
        return self.emb_path.exists() and self.index_path.exists()

    def load(self) -> Tuple[np.ndarray, faiss.IndexFlatIP]:
        if not self.ready():
            raise FileNotFoundError(
                f"Missing embeddings/index in {self.storage_dir}. Run scripts/build_indexes.py first."
            )
        if self._embeddings is None:
            self._embeddings = np.load(self.emb_path)
        if self._index is None:
            self._index = faiss.read_index(str(self.index_path))  # type: ignore[arg-type]
        return self._embeddings, self._index

    def ensure_ready(self) -> Tuple[np.ndarray, faiss.IndexFlatIP]:
        try:
            return self.load()
        except FileNotFoundError:
            self.build_and_persist()
            return self.load()

    # ------------------------------------------------------------------ builders
    def build_and_persist(
        self,
        model_name: Optional[str] = None,
        batch_size: int = 128,
    ) -> None:
        texts = self.dataset.texts()
        if not texts:
            raise RuntimeError("No texts available for embedding.")

        model_name = model_name or self.model_name
        model = SentenceTransformer(model_name)
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

        self._embeddings = embeddings
        self._index = index

    # ---------------------------------------------------------------- properties
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
