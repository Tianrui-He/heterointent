from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterointent.data.dataset import build_dataloader
from heterointent.evaluation.metrics import compute_ranking_metrics
from heterointent.inference.rank import load_model
from heterointent.training.trainer import predict_frame

README_GROUPS = {
    "graph": ("graph",),
    "dense": ("item_dense", "ratio"),
    "text": ("text_fused", "text", "text_title", "text_content"),
    "video-meta": ("video_meta",),
    "image-meta": ("image_meta",),
    "image-emb": ("image_emb",),
}


def grouped_top20(metrics: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, parts in README_GROUPS.items():
        out[label] = sum(float(metrics.get(f"top20_mean_gate_{part}", 0.0)) for part in parts)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Report Top-20 gate diagnostics on a split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    model, metadata, _, device = load_model(args.checkpoint, device="auto")
    loader = build_dataloader(
        args.samples,
        metadata,
        batch_size=int(args.batch_size),
        shuffle=False,
        fast_loader=True,
        pin_memory=device.type == "cuda",
    )
    pred = predict_frame(model, loader, device)
    metrics = compute_ranking_metrics(pred, topk=int(args.topk), include_diagnostics=True)
    grouped = grouped_top20(metrics)
    payload = {
        "checkpoint": str(args.checkpoint),
        "samples": str(args.samples),
        "part_names": list(getattr(model.item_encoder, "part_names", [])),
        "grouped_top20": grouped,
        "raw_top20": {
            key.removeprefix("top20_mean_gate_"): float(value)
            for key, value in metrics.items()
            if key.startswith("top20_mean_gate_")
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
