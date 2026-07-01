from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterointent.data.qilin import convert_qilin_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert official THUIR/Qilin parquet folders into HeteroIntent format.")
    parser.add_argument("--qilin-dir", default=str(ROOT / "data" / "raw" / "Qilin"), help="Directory containing recommendation_train, recommendation_test, notes, user_feat.")
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "run_latest" / "processed" / "qilin_base"))
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--text-hash-dim", type=int, default=0, help="0 disables cheap hashed text features. Use build_text_embeddings.py for strong text embeddings.")
    args = parser.parse_args()

    metadata = convert_qilin_directory(
        qilin_dir=args.qilin_dir,
        output_dir=args.output_dir,
        max_history=args.max_history,
        text_hash_dim=args.text_hash_dim,
    )
    print(f"wrote processed Qilin data to {args.output_dir}")
    print(metadata)


if __name__ == "__main__":
    main()
