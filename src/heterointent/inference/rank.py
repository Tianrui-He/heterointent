from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from heterointent.data.dataset import build_dataloader
from heterointent.evaluation.metrics import TASKS, deduplicate_request_items
from heterointent.models import HeteroIntentPLE
from heterointent.training.trainer import predict_frame
from heterointent.utils import resolve_device


def load_model(checkpoint_path: str | Path, device: str = "auto") -> tuple[HeteroIntentPLE, dict, dict, torch.device]:
    resolved = resolve_device(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=resolved, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=resolved)
    metadata = checkpoint["metadata"]
    config = checkpoint["config"]
    model = HeteroIntentPLE(metadata, config)
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as exc:
        message = str(exc)
        legacy_missing_only = "Missing key(s)" in message and "size mismatch" not in message and "Unexpected key(s)" not in message
        if "transition_head" not in message and "type_transition_head" not in message and not legacy_missing_only:
            raise
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        print(
            "Loaded a legacy checkpoint with newly initialized compatible heads/aliases; "
            f"missing={list(incompatible.missing_keys)}, unexpected={list(incompatible.unexpected_keys)}"
        )
    model.to(resolved)
    model.eval()
    return model, metadata, config, resolved


def rank_predictions(pred: pd.DataFrame, topk: int = 20) -> pd.DataFrame:
    pred = deduplicate_request_items(pred)
    ranked_rows = []
    for request_id, group in pred.groupby("request_id", sort=False):
        ranked = group.sort_values("score", ascending=False).head(topk).copy()
        ranked["rank"] = range(1, len(ranked) + 1)
        ranked_rows.append(ranked)
    if not ranked_rows:
        return pd.DataFrame(columns=["request_id", "rank", "item_id", "score", *[f"p_{t}" for t in TASKS]])
    cols = ["request_id", "rank", "item_id", "score", *[f"p_{t}" for t in TASKS]]
    return pd.concat(ranked_rows, ignore_index=True)[cols]


def rank_file(
    checkpoint_path: str | Path,
    samples_path: str | Path,
    output_path: str | Path,
    device: str = "auto",
    batch_size: int | None = None,
    topk: int = 20,
) -> pd.DataFrame:
    model, metadata, config, resolved = load_model(checkpoint_path, device=device)
    if batch_size is None:
        batch_size = int(config["data"].get("batch_size", 256))
    loader = build_dataloader(samples_path, metadata, batch_size=batch_size, shuffle=False)
    pred = predict_frame(model, loader, resolved)
    ranked = rank_predictions(pred, topk=topk)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        ranked.to_parquet(output_path, index=False)
    else:
        ranked.to_csv(output_path, index=False)
    return ranked
