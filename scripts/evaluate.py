from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterointent.evaluation.metrics import compute_ranking_metrics
from heterointent.inference.rank import load_model
from heterointent.training.trainer import predict_frame
from heterointent.data.dataset import build_dataloader


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a parquet sample file.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fast-loader", action="store_true", help="Use tensor batch loader for faster GPU inference")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    model, metadata, _, device = load_model(args.checkpoint, device=args.device)
    loader = build_dataloader(
        args.samples,
        metadata,
        batch_size=args.batch_size,
        shuffle=False,
        fast_loader=bool(args.fast_loader),
        pin_memory=device.type == "cuda",
    )
    pred = predict_frame(model, loader, device)
    metrics = compute_ranking_metrics(pred, topk=args.topk)
    print(metrics)


if __name__ == "__main__":
    main()
