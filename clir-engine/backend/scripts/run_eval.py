from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.eval_metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402
from app.colab_core.search import get_components, search as backend_search  # noqa: E402


def load_labels(path: Path, url_to_doc: Dict[str, int]) -> Dict[str, Dict[int, int]]:
    labels: Dict[str, Dict[int, int]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"query", "doc_url", "relevant"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Label file missing required columns: {required}")

        for row in reader:
            query = (row.get("query") or "").strip()
            url = (row.get("doc_url") or "").strip()
            rel = row.get("relevant", "0").strip()
            if not query or not url:
                continue
            doc_id = url_to_doc.get(url)
            if doc_id is None:
                continue
            relevance = 1 if rel in {"1", "true", "True"} else 0
            labels.setdefault(query, {})[doc_id] = relevance
    return labels


def main() -> None:
    data_dir = ROOT / "data" / "processed"
    dataset, _, _, _ = get_components()
    url_to_doc = {doc.url: doc.doc_id for doc in dataset.iter_documents()}
    labels_path = data_dir / "labels_fiiled.csv"
    if not labels_path.exists():
        alt = data_dir / "labels.csv"
        if alt.exists():
            labels_path = alt
    labels = load_labels(labels_path, url_to_doc)
    if not labels:
        raise RuntimeError("No labels found to evaluate.")

    rows: List[Dict[str, float]] = []
    for query, doc_labels in labels.items():
        search = backend_search(query, lang="all", k=50, debug=True)
        doc_ids = [item["_doc_id"] for item in search["results"] if "_doc_id" in item]
        relevances = [doc_labels.get(doc_id, 0) for doc_id in doc_ids]
        total_relevant = sum(doc_labels.values())
        metrics = {
            "query": query,
            "P@10": precision_at_k(relevances, 10),
            "Recall@50": recall_at_k(relevances, 50, total_relevant),
            "nDCG@10": ndcg_at_k(relevances, 10),
            "MRR": mrr(relevances),
        }
        rows.append(metrics)

    out_path = data_dir / "eval_results.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "P@10", "Recall@50", "nDCG@10", "MRR"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    avg = {
        "P@10": sum(r["P@10"] for r in rows) / len(rows),
        "Recall@50": sum(r["Recall@50"] for r in rows) / len(rows),
        "nDCG@10": sum(r["nDCG@10"] for r in rows) / len(rows),
        "MRR": sum(r["MRR"] for r in rows) / len(rows),
    }

    print(f"Saved evaluation results to {out_path}")
    print("Averages:")
    for key, value in avg.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
