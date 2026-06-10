from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterointent.inference import rank_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidates and export Top-20 predictions.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", default="outputs/submission_top20.csv")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()
    ranked = rank_file(
        checkpoint_path=args.checkpoint,
        samples_path=args.samples,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
        topk=args.topk,
    )
    print(f"wrote {len(ranked)} ranked rows to {args.output}")


if __name__ == "__main__":
    main()
