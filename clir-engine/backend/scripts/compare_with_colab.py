from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.colab_core.search import search as backend_search  # noqa: E402


def load_colab_results(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Colab results file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "queries" in data:
        data = data["queries"]
    if not isinstance(data, list):
        raise ValueError("Expected a list of query entries in the Colab JSON.")
    return data


def overlap_at_k(colab_urls: List[str], backend_urls: List[str], k: int) -> float:
    top_colab = [u.strip() for u in colab_urls[:k]]
    top_backend = [u.strip() for u in backend_urls[:k]]
    return len(set(top_colab) & set(top_backend)) / float(k or 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare backend results against Colab notebook outputs."
    )
    parser.add_argument("--colab-results", required=True, type=Path)
    parser.add_argument("--lang", default="all")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    entries = load_colab_results(args.colab_results)
    scores = []
    for entry in entries:
        query = entry.get("query", "")
        colab_urls = [r.get("url", "") for r in entry.get("results", [])]
        backend = backend_search(query, lang=args.lang, k=args.k, debug=False)
        backend_urls = [r.get("url", "") for r in backend["results"]]
        overlap = overlap_at_k(colab_urls, backend_urls, args.k)
        scores.append(overlap)
        print(f"{query!r}: overlap@{args.k} = {overlap:.2f}")
        if overlap < 1.0:
            missing = set(colab_urls[: args.k]) - set(backend_urls[: args.k])
            extra = set(backend_urls[: args.k]) - set(colab_urls[: args.k])
            print("  Missing:", missing)
            print("  Extra:", extra)

    if scores:
        avg = sum(scores) / len(scores)
        print(f"\nAverage overlap@{args.k}: {avg:.3f}")
    else:
        print("No queries loaded from Colab JSON.")


if __name__ == "__main__":
    main()
