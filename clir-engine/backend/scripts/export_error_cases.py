from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABEL_FILES = [
    ROOT / "data/processed/labels_fiiled.csv",
    ROOT / "data/processed/labels.csv",
]

import sys

sys.path.append(str(ROOT))

from app.colab_core.search import search as backend_search  # noqa: E402


def detect_column(fieldnames, candidates):
    for name in candidates:
        if name in fieldnames:
            return name
    raise ValueError(f"Could not determine column out of {candidates}")


def normalize_url(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


def load_labels(path: Path) -> Dict[str, Dict[str, bool]]:
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    with path.open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Label CSV missing headers.")
        query_col = detect_column(reader.fieldnames, ["query", "question"])
        url_col = detect_column(reader.fieldnames, ["doc_url", "url", "link"])
        rel_col = detect_column(reader.fieldnames, ["relevant", "is_relevant", "rel"])

        relevance_map: Dict[str, Dict[str, bool]] = {}
        for row in reader:
            query = (row.get(query_col) or "").strip()
            url = normalize_url(row.get(url_col) or "")
            rel_text = (row.get(rel_col) or "").strip().lower()
            if not query or not url:
                continue
            relevant = rel_text in {"1", "true", "yes", "y", "relevant"}
            relevance_map.setdefault(query, {})[url] = relevant

    if not relevance_map:
        raise ValueError(f"No labeled entries found in {path}")
    return relevance_map


def get_labels_path(explicit: Optional[Path]) -> Path:
    if explicit:
        return explicit
    for candidate in DEFAULT_LABEL_FILES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find labels_fiiled.csv or labels.csv. "
        "Provide --labels PATH explicitly."
    )


def collect_cases(
    labels: Dict[str, Dict[str, bool]],
    queries: list[str],
    target_n: int,
    mode: str,
    lang_filter: str,
    topk: int,
    min_score: Optional[float],
) -> list[Dict[str, str]]:
    cases: list[Dict[str, str]] = []
    for query in queries:
        if len(cases) >= target_n:
            break
        response = backend_search(query, lang=lang_filter, k=topk, debug=True)
        results = response.get("results", [])
        if not results:
            continue
        rows = results[:1] if mode == "top1" else results[:topk]
        for res in rows:
            if len(cases) >= target_n:
                break
            score = res.get("score")
            if min_score is not None and score is not None and float(score) < min_score:
                continue
            url_key = normalize_url(res.get("url") or "")
            label_dict = labels.get(query, {})
            label_val = label_dict.get(url_key)
            record = {
                "case_id": str(len(cases) + 1),
                "query": query,
                "detected_lang": response.get("detected_lang", ""),
                "query_en": response.get("query_variants", {}).get("en", ""),
                "query_bn": response.get("query_variants", {}).get("bn", ""),
                "rank": str(res.get("rank")),
                "title": res.get("title", ""),
                "url": res.get("url", ""),
                "doc_language": res.get("language", ""),
                "source": res.get("source", ""),
                "score": res.get("score"),
                "is_relevant": (
                    "yes" if label_val else ("no" if label_val is False else "")
                ),
                "error_type": "",
                "notes": "",
                "annotator": "",
            }
            cases.append(record)
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export failure cases for manual error analysis."
    )
    parser.add_argument("--labels", type=Path, default=None, help="Path to labels CSV.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/processed/error_cases_template.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--n", type=int, default=100, help="Number of cases to export.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lang", default="all", choices=["all", "en", "bn"], help="Language filter."
    )
    parser.add_argument("--k", type=int, default=10, help="Top-k results per query.")
    parser.add_argument(
        "--mode",
        choices=["top1", "top10"],
        default="top10",
        help="Use only top1 per query or top-k rows.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Skip cases whose top score falls below this threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_path = get_labels_path(args.labels)
    labels = load_labels(labels_path)

    rng = random.Random(args.seed)
    queries = list(labels.keys())
    rng.shuffle(queries)

    cases = collect_cases(
        labels=labels,
        queries=queries,
        target_n=args.n,
        mode=args.mode,
        lang_filter=args.lang,
        topk=args.k,
        min_score=args.min_score,
    )

    if len(cases) < args.n:
        print(f"Warning: only collected {len(cases)} cases out of requested {args.n}.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "query",
        "detected_lang",
        "query_en",
        "query_bn",
        "rank",
        "title",
        "url",
        "doc_language",
        "source",
        "score",
        "is_relevant",
        "error_type",
        "notes",
        "annotator",
    ]
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in cases:
            writer.writerow(row)

    print(f"Exported {len(cases)} cases to {args.out} (labels from {labels_path})")
    print("Next steps: copy the template to error_cases_annotated.csv, fill any blank")
    print("is_relevant values, and label each failure with the required error_type.")


if __name__ == "__main__":
    main()
