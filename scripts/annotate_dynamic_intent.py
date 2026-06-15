from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterointent.data.dynamic_intent import annotate_processed_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate processed Qilin samples with dynamic intent transition labels.")
    parser.add_argument("--processed-dir", default="data/processed/qilin_full")
    parser.add_argument("--qilin-dir", default=None, help="Optional raw Qilin directory for complete note item_type/taxonomy lookup.")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--max-history", type=int, default=None)
    args = parser.parse_args()

    summary = annotate_processed_directory(
        processed_dir=args.processed_dir,
        qilin_dir=args.qilin_dir,
        splits=args.splits,
        max_history=args.max_history,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
