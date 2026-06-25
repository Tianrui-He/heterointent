from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from heterointent.config import save_config, to_plain_dict
from heterointent.data.dataset import build_dataloader
from heterointent.evaluation.metrics import TASKS, compute_ranking_metrics
from heterointent.models import HeteroIntentPLE
from heterointent.training.losses import compute_loss
from heterointent.utils import count_parameters, read_json, resolve_device, set_seed, write_json


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    sample = next(iter(batch.values()))
    if sample.device.type == device.type:
        return batch
    non_blocking = device.type == "cuda"
    return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}


def _topk_hits(logits: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, logits.size(-1))
    top = torch.topk(logits, k=k, dim=-1).indices
    return top.eq(target.clamp(min=0).unsqueeze(1)).any(dim=1) & target.gt(0)


def _target_mrr(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    safe_target = target.clamp(min=0, max=logits.size(-1) - 1)
    target_logits = logits.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    rank = logits.gt(target_logits.unsqueeze(1)).sum(dim=1).float() + 1.0
    return torch.where(target.gt(0), rank.reciprocal(), torch.zeros_like(rank))


def _attention_mass(attn: torch.Tensor, history_values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    match = history_values.eq(target.unsqueeze(1)) & target.gt(0).unsqueeze(1)
    return (attn.float() * match.float()).sum(dim=1)


@torch.no_grad()
def predict_frame(model: nn.Module, loader, device: torch.device) -> pd.DataFrame:
    model.eval()
    frames = []
    for batch in loader:
        batch = _move_batch(batch, device)
        outputs = model(batch)
        probs = torch.nan_to_num(outputs["probs"], nan=0.0, posinf=1.0, neginf=0.0).detach().cpu().numpy()
        scores = torch.nan_to_num(outputs["final_score"], nan=0.0).detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()
        frame = pd.DataFrame(
            {
                "request_id": batch["request_id"].detach().cpu().numpy(),
                "item_id": batch["item_id"].detach().cpu().numpy(),
                "score": scores,
            }
        )
        for task_idx, task in enumerate(TASKS):
            frame[task] = labels[:, task_idx].astype("int8", copy=False)
            frame[f"p_{task}"] = probs[:, task_idx]

        if "modality_gate" in outputs:
            gate = outputs["modality_gate"].detach().cpu().numpy()
            part_names = list(getattr(getattr(model, "item_encoder", None), "part_names", []))
            if len(part_names) != gate.shape[1]:
                part_names = [f"part_{idx}" for idx in range(gate.shape[1])]
            for part_idx, part_name in enumerate(part_names):
                frame[f"gate_{part_name}"] = gate[:, part_idx].astype("float32", copy=False)

        dynamic_cols = [
            "target_item_type",
            "target_taxonomy_id",
            "hist_dominant_item_type",
            "hist_dominant_taxonomy_id",
            "is_type_shift",
            "is_taxonomy_shift",
            "has_intent_target",
        ]
        for col in dynamic_cols:
            if col in batch:
                frame[col] = batch[col].detach().cpu().numpy().astype("int64", copy=False)

        if "type_transition_logits" in outputs and "target_item_type" in batch:
            type_logits = outputs["type_transition_logits"]
            target_type = batch["target_item_type"]
            frame["intent_type_hit@1"] = _topk_hits(type_logits, target_type, k=1).detach().cpu().numpy().astype("float32")
            frame["intent_type_hit@2"] = _topk_hits(type_logits, target_type, k=2).detach().cpu().numpy().astype("float32")
        if "taxonomy_transition_logits" in outputs and "target_taxonomy_id" in batch:
            taxonomy_logits = outputs["taxonomy_transition_logits"]
            target_taxonomy = batch["target_taxonomy_id"]
            frame["intent_taxonomy_hit@1"] = (
                _topk_hits(taxonomy_logits, target_taxonomy, k=1).detach().cpu().numpy().astype("float32")
            )
            frame["intent_taxonomy_hit@5"] = (
                _topk_hits(taxonomy_logits, target_taxonomy, k=5).detach().cpu().numpy().astype("float32")
            )
            frame["intent_taxonomy_mrr"] = _target_mrr(taxonomy_logits, target_taxonomy).detach().cpu().numpy().astype("float32")
        if "history_attention" in outputs:
            attn = outputs["history_attention"]
            if "history_item_types" in batch and "target_item_type" in batch:
                frame["attention_type_target_mass"] = (
                    _attention_mass(attn, batch["history_item_types"], batch["target_item_type"]).detach().cpu().numpy().astype("float32")
                )
            if "history_taxonomy_ids" in batch and "target_taxonomy_id" in batch:
                frame["attention_taxonomy_target_mass"] = (
                    _attention_mass(attn, batch["history_taxonomy_ids"], batch["target_taxonomy_id"])
                    .detach()
                    .cpu()
                    .numpy()
                    .astype("float32")
                )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["request_id", "item_id", "score", *TASKS, *[f"p_{t}" for t in TASKS]])
    return pd.concat(frames, ignore_index=True)


def _load_history(output_dir: Path) -> list[dict]:
    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists():
        return []
    return pd.read_csv(metrics_path).to_dict("records")


def _metric_value(row: dict, metric_name: str) -> float:
    value = row.get(metric_name)
    if value is None or not pd.notna(value):
        value = row.get("weighted_hit@20", -1.0)
    value = float(value)
    return value if np.isfinite(value) else -1.0


def _best_metric_from_history(history: list[dict], metric_name: str = "quality_score") -> float:
    values = [_metric_value(row, metric_name) for row in history]
    return max(values) if values else -1.0


def _best_record_from_history(history: list[dict], metric_name: str) -> dict:
    if not history:
        return {}
    return max(history, key=lambda row: _metric_value(row, metric_name))


def train(config: dict, resume_path: str | None = None) -> dict:
    set_seed(int(config.get("seed", 2026)))
    device = resolve_device(str(config.get("device", "auto")))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    data_cfg = config["data"]
    output_dir = Path(config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    processed_dir = Path(data_cfg["processed_dir"])
    metadata = read_json(processed_dir / data_cfg.get("metadata_file", "metadata.json"))
    train_loader = build_dataloader(
        processed_dir / data_cfg.get("train_file", "train.parquet"),
        metadata,
        batch_size=int(data_cfg.get("batch_size", 256)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", device.type == "cuda")),
        fast_loader=bool(data_cfg.get("fast_loader", False)),
        request_preserving=bool(data_cfg.get("request_preserving_train", data_cfg.get("request_preserving", False))),
        tensor_device=str(data_cfg.get("tensor_device", "")) or None,
    )
    eval_batch_size = int(data_cfg.get("eval_batch_size", data_cfg.get("batch_size", 256)))
    valid_loader = build_dataloader(
        processed_dir / data_cfg.get("valid_file", "valid.parquet"),
        metadata,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", device.type == "cuda")),
        fast_loader=bool(data_cfg.get("eval_fast_loader", data_cfg.get("fast_loader", False))),
        tensor_device=str(data_cfg.get("tensor_device", "")) or None,
    )

    model = HeteroIntentPLE(metadata, config).to(device)
    graph_path = processed_dir / "graph_embedding.npy"
    if graph_path.exists():
        graph_values = torch.tensor(np.load(graph_path), dtype=torch.float32, device=device)
        model.item_encoder.load_graph_embedding(graph_values)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["train"].get("lr", 1e-3)),
        weight_decay=float(config["train"].get("weight_decay", 1e-4)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["train"].get("amp", False)) and device.type == "cuda")

    patience = int(config["train"].get("patience", 3))
    min_epochs = int(config["train"].get("min_epochs", 0))
    stale = 0
    history: list[dict] = []
    epochs = int(config["train"].get("epochs", 10))
    topk = int(config["evaluation"].get("topk", 20))
    selection_metric = str(config.get("evaluation", {}).get("selection_metric", "quality_score"))
    start_epoch = 1
    best_metric = -1.0
    save_every_epoch = bool(config["train"].get("save_every_epoch", False))

    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        history = list(checkpoint.get("history") or _load_history(output_dir))
        checkpoint_selection_metric = str(checkpoint.get("selection_metric", checkpoint.get("selection_metric_name", selection_metric)))
        if checkpoint_selection_metric == selection_metric:
            best_metric = float(checkpoint.get("best_metric", _best_metric_from_history(history, selection_metric)))
        else:
            best_metric = _best_metric_from_history(history, selection_metric)
        stale = int(checkpoint.get("stale", 0))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("Resume checkpoint has no optimizer state; continuing from model weights with a fresh optimizer.")
        if "scaler" in checkpoint and scaler.is_enabled() and checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        if "epoch" in checkpoint:
            start_epoch = int(checkpoint["epoch"]) + 1
        elif history:
            start_epoch = int(max(row.get("epoch", 0) for row in history)) + 1
        print(f"Resumed from {resume_path}; next epoch = {start_epoch}; best_{selection_metric} = {best_metric:.6f}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_log_totals: dict[str, torch.Tensor] = {}
        epoch_steps = 0
        start = time.perf_counter()
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                outputs = model(batch)
                loss, logs = compute_loss(outputs, batch, config["loss"])
            scaler.scale(loss).backward()
            if float(config["train"].get("grad_clip", 0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            for key, value in logs.items():
                epoch_log_totals[key] = epoch_log_totals.get(key, torch.zeros((), device=device)) + value
            epoch_steps += 1

        pred = predict_frame(model, valid_loader, device)
        metrics = compute_ranking_metrics(pred, topk=topk)
        mean_logs = {
            f"train_{key}": float((value / max(epoch_steps, 1)).detach().cpu())
            for key, value in epoch_log_totals.items()
        }
        record = {
            "epoch": epoch,
            "train_loss": mean_logs.get("train_loss", 0.0),
            "seconds": time.perf_counter() - start,
            **mean_logs,
            **metrics,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(output_dir / "metrics.csv", index=False)

        metric = _metric_value(metrics, selection_metric)
        improved = metric > best_metric
        if improved:
            best_metric = metric
            stale = 0
        else:
            stale += 1

        checkpoint_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "metadata": metadata,
            "config": to_plain_dict(config),
            "epoch": epoch,
            "best_metric": best_metric,
            "selection_metric": selection_metric,
            "stale": stale,
            "history": history,
        }
        torch.save(checkpoint_payload, output_dir / "last.pt")
        if save_every_epoch:
            torch.save(checkpoint_payload, output_dir / f"epoch_{epoch:03d}.pt")

        if improved:
            torch.save(checkpoint_payload, output_dir / "best.pt")
            pred.to_parquet(output_dir / "valid_predictions.parquet", index=False)
        elif epoch >= min_epochs and stale >= patience:
            break

    best_record = _best_record_from_history(history, selection_metric)
    summary = {
        "selection_metric": selection_metric,
        "best_metric": best_metric,
        "best_selection_metric": _metric_value(best_record, selection_metric) if best_record else best_metric,
        "best_epoch": int(best_record.get("epoch", 0)) if best_record else 0,
        "best_weighted_hit@20": float(best_record.get("weighted_hit@20", float("nan"))) if best_record else float("nan"),
        "best_quality_score": float(best_record.get("quality_score", float("nan"))) if best_record else float("nan"),
        "num_parameters": count_parameters(model),
        "device": str(device),
        "last_epoch": int(history[-1]["epoch"]) if history else 0,
    }
    write_json(summary, output_dir / "summary.json")
    return {"output_dir": str(output_dir), **summary}
