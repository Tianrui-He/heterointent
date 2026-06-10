from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterointent.data.preprocess import preprocess_flat_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize flattened Qilin-like samples into train/valid/test parquet.")
    parser.add_argument("--input", required=True, help="CSV/JSON/JSONL/Parquet with one row per request-item candidate.")
    parser.add_argument("--output-dir", default="data/processed/qilin")
    parser.add_argument("--max-history", type=int, default=20)
    args = parser.parse_args()
    metadata = preprocess_flat_samples(args.input, args.output_dir, max_history=args.max_history)
    print(f"wrote processed data to {args.output_dir}")
    print(metadata)


if __name__ == "__main__":
    main()
