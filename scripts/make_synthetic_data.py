from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterointent.data.synthetic import make_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Qilin-like synthetic dataset for smoke tests.")
    parser.add_argument("--output-dir", default="data/processed/synthetic")
    parser.add_argument("--num-requests", type=int, default=900)
    parser.add_argument("--num-users", type=int, default=120)
    parser.add_argument("--num-items", type=int, default=600)
    parser.add_argument("--candidates-per-request", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    metadata = make_synthetic_dataset(
        output_dir=args.output_dir,
        num_requests=args.num_requests,
        num_users=args.num_users,
        num_items=args.num_items,
        candidates_per_request=args.candidates_per_request,
        seed=args.seed,
    )
    print(f"wrote synthetic dataset to {args.output_dir}")
    print(metadata)


if __name__ == "__main__":
    main()
