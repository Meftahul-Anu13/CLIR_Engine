from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        data = json.load(f_in)
        for record in data:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Converted {src} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSON array to JSONL")
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == "__main__":
    main()
