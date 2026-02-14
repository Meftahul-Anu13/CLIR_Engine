from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from app.error_analysis import (
    SUPPORTED_ERROR_TYPES,
    normalize_error_type,
    stable_round_percentages,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute error distribution percentages from annotated cases."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/processed/error_cases_annotated.csv",
        help="Annotated CSV with error_type column.",
    )
    parser.add_argument(
        "--min-cases",
        type=int,
        default=100,
        help="Minimum number of annotated cases required.",
    )
    parser.add_argument(
        "--latex-out",
        type=Path,
        default=ROOT / "data/processed/error_distribution_table.tex",
        help="Path to save LaTeX table snippet.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=ROOT / "data/processed/error_distribution_summary.csv",
        help="Where to save percentage summary as CSV.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "data/processed/error_distribution_summary.json",
        help="Where to save percentage summary as JSON.",
    )
    return parser.parse_args()


def load_error_counts(path: Path) -> Dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(
            f"Annotated error file not found: {path}. "
            "Run export_error_cases.py and annotate the template first."
        )
    counts = {etype: 0 for etype in SUPPORTED_ERROR_TYPES}
    with path.open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "error_type" not in (reader.fieldnames or []):
            raise ValueError("CSV must contain an 'error_type' column.")
        for idx, row in enumerate(reader, start=1):
            label_raw = (row.get("error_type") or "").strip()
            if not label_raw:
                raise ValueError(f"Row {idx} missing error_type. Annotate all cases.")
            label = normalize_error_type(label_raw)
            counts[label] = counts.get(label, 0) + 1
    return counts


def print_table(counts: Dict[str, int], percentages: Dict[str, int], total: int) -> None:
    print(f"Error Distribution ({total} cases)")
    print("-" * 50)
    print(f"{'Error Type':30} | Percentage")
    print("-" * 50)
    for etype in SUPPORTED_ERROR_TYPES:
        pct = percentages.get(etype, 0)
        print(f"{etype:30} | {pct:3d}%")
    print("-" * 50)


def write_csv(path: Path, counts: Dict[str, int], percentages: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["error_type", "count", "percentage"])
        for etype in SUPPORTED_ERROR_TYPES:
            writer.writerow([etype, counts.get(etype, 0), percentages.get(etype, 0)])


def write_json(path: Path, counts: Dict[str, int], percentages: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: List[dict] = []
    for etype in SUPPORTED_ERROR_TYPES:
        payload.append(
            {
                "error_type": etype,
                "count": counts.get(etype, 0),
                "percentage": percentages.get(etype, 0),
            }
        )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_latex(path: Path, counts: Dict[str, int], percentages: Dict[str, int], total: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Error Type & Percentage \\\\",
        "\\midrule",
    ]
    for etype in SUPPORTED_ERROR_TYPES:
        pct = percentages.get(etype, 0)
        lines.append(f"{etype} & {pct}\\% \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{Error Frequency ({total} cases)}}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    counts = load_error_counts(args.input)
    total_cases = sum(counts.values())
    if total_cases < args.min_cases:
        raise ValueError(
            f"Annotated cases ({total_cases}) below required minimum ({args.min_cases})."
        )

    percentages = stable_round_percentages(counts)
    print_table(counts, percentages, total_cases)
    write_csv(args.csv_out, counts, percentages)
    write_json(args.json_out, counts, percentages)
    write_latex(args.latex_out, counts, percentages, total_cases)
    print(f"Saved summaries to {args.csv_out}, {args.json_out}, {args.latex_out}")


if __name__ == "__main__":
    main()
