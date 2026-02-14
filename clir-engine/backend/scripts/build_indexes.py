from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.dataset import DatasetStore  # noqa: E402
from app.index_store import IndexStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute CLIR embeddings + FAISS index")
    parser.add_argument("--data-dir", default=ROOT / "data" / "processed", type=Path)
    parser.add_argument("--storage-dir", default=ROOT / "storage", type=Path)
    parser.add_argument("--batch-size", default=128, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = DatasetStore(Path(args.data_dir))
    store = IndexStore(dataset, Path(args.storage_dir))
    store.build_and_persist(batch_size=args.batch_size)
    print(f"Embeddings stored in {store.emb_path}")
    print(f"FAISS index stored in {store.index_path}")


if __name__ == "__main__":
    main()
