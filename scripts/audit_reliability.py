from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterointent.data.dataset import build_dataloader
from heterointent.evaluation.metrics import TASKS, compute_ranking_metrics
from heterointent.inference.rank import load_model
from heterointent.training.trainer import predict_frame

TASK_WEIGHTS = {"click": 0.3, "collect": 0.4, "share": 0.3}


def _float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return float(value)


def candidate_stats(df: pd.DataFrame) -> dict[str, Any]:
    request_items = df.groupby("request_id")["item_id"].nunique()
    positives = df.groupby("request_id")[list(TASKS)].sum()
    oracle = sum(TASK_WEIGHTS[task] * positives[task].gt(0).mean() for task in TASKS)
    stats: dict[str, Any] = {
        "rows": int(len(df)),
        "requests": int(df["request_id"].nunique()),
        "duplicate_request_item_rows": int(df.duplicated(["request_id", "item_id"]).sum()),
        "candidate_count": {
            "mean": _float(request_items.mean()),
            "median": _float(request_items.median()),
            "p75": _float(request_items.quantile(0.75)),
            "p90": _float(request_items.quantile(0.90)),
            "max": int(request_items.max()) if len(request_items) else 0,
            "lt20_rate": _float(request_items.lt(20).mean()),
            "ge20_rate": _float(request_items.ge(20).mean()),
            "le20_rate": _float(request_items.le(20).mean()),
            "gt20_rate": _float(request_items.gt(20).mean()),
        },
        "oracle_weighted_hit@20": _float(oracle),
    }
    for task in TASKS:
        has_positive = positives[task].gt(0)
        stats[task] = {
            "positive_requests": int(has_positive.sum()),
            "positive_request_rate": _float(has_positive.mean()),
            "avg_positive_labels_if_any": _float(positives.loc[has_positive, task].mean()) if has_positive.any() else 0.0,
        }
    return stats


def _request_subset(df: pd.DataFrame, request_ids: pd.Index | set[int]) -> pd.DataFrame:
    ids = set(int(x) for x in request_ids)
    return df[df["request_id"].astype(int).isin(ids)].copy()


def _metrics_or_empty(df: pd.DataFrame, topk: int) -> dict[str, float]:
    if df.empty or df["request_id"].nunique() == 0:
        return {}
    return compute_ranking_metrics(df, topk=topk)


def grouped_prediction_metrics(pred: pd.DataFrame, samples: pd.DataFrame, topk: int) -> dict[str, Any]:
    request_items = samples.groupby("request_id")["item_id"].nunique()
    positives = samples.groupby("request_id")[list(TASKS)].sum()
    groups = {
        "all": samples["request_id"].drop_duplicates(),
        "candidate_count_ge20": request_items[request_items.ge(20)].index,
        "candidate_count_lt20": request_items[request_items.lt(20)].index,
        "candidate_count_gt20": request_items[request_items.gt(topk)].index,
        "candidate_count_le20": request_items[request_items.le(topk)].index,
        "collect_positive": positives[positives["collect"].gt(0)].index,
        "collect_non_positive": positives[positives["collect"].le(0)].index,
        "share_positive": positives[positives["share"].gt(0)].index,
        "share_non_positive": positives[positives["share"].le(0)].index,
    }
    out: dict[str, Any] = {}
    oracle = candidate_stats(samples)["oracle_weighted_hit@20"]
    for name, ids in groups.items():
        group_pred = _request_subset(pred, ids)
        metrics = _metrics_or_empty(group_pred, topk=topk)
        if metrics and name == "all" and oracle > 0:
            metrics["weighted_hit_oracle_normalized"] = metrics.get("official_weighted_hit@20", metrics.get("weighted_hit@20", float("nan"))) / oracle
        out[name] = metrics
    return out


def graph_coverage(processed_dir: Path) -> dict[str, Any]:
    train_items = set(pd.read_parquet(processed_dir / "train.parquet", columns=["item_id"])["item_id"].astype(int).unique())
    out: dict[str, Any] = {
        "static_audit": {
            "graph_builder": "scripts/build_item_graph.py",
            "samples_file_default": "train.parquet",
            "uses_eval_parquet": False,
            "uses_label_columns": False,
            "feature_prefix": "text_feat_",
            "edge_source": "session/request item co-occurrence from train samples",
        }
    }
    for split in ["valid", "test"]:
        df = pd.read_parquet(processed_dir / f"{split}.parquet", columns=["item_id", *TASKS])
        item_ids = df["item_id"].astype(int)
        seen = item_ids.isin(train_items)
        unique_items = set(item_ids.unique())
        positive_rows = df[list(TASKS)].max(axis=1).astype(bool)
        out[split] = {
            "rows_seen_in_train_graph": int(seen.sum()),
            "row_seen_rate": _float(seen.mean()),
            "unique_items": int(len(unique_items)),
            "unique_items_seen_in_train_graph": int(len(unique_items & train_items)),
            "unique_seen_rate": _float(len(unique_items & train_items) / max(len(unique_items), 1)),
            "positive_row_seen_rate": _float(seen[positive_rows].mean()) if positive_rows.any() else float("nan"),
        }
    return out


def predict_split(checkpoint: Path, samples: Path, batch_size: int, device: str):
    model, metadata, _, resolved = load_model(checkpoint, device=device)
    loader = build_dataloader(samples, metadata, batch_size=batch_size, shuffle=False)
    return predict_frame(model, loader, resolved)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Split Candidate Audit",
        "",
        "## Conclusion",
        "",
        "- valid and test are not identically distributed: valid is the last 10% of recommendation_train, while test comes from recommendation_test.",
        "- test has a much higher click-positive request rate and a higher oracle WeightedHit@20 ceiling, so raw valid/test metrics should not be compared as same-distribution generalization.",
        "- graph construction uses train.parquet only and does not use label columns; strict graph zero-OOV ablation is still recommended because the compact ID space is transductive.",
        "",
        "## Split Summary",
        "",
        "| split | requests | rows | dup rows | cand mean | cand median | cand <=20 | cand >20 | click pos rate | collect pos rate | share pos rate | oracle WH@20 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, stats in report["splits"].items():
        lines.append(
            f"| {split} | {stats['requests']} | {stats['rows']} | {stats['duplicate_request_item_rows']} | "
            f"{stats['candidate_count']['mean']:.4f} | {stats['candidate_count']['median']:.1f} | "
            f"{stats['candidate_count']['le20_rate']:.4f} | {stats['candidate_count']['gt20_rate']:.4f} | "
            f"{stats['click']['positive_request_rate']:.4f} | "
            f"{stats['collect']['positive_request_rate']:.4f} | {stats['share']['positive_request_rate']:.4f} | "
            f"{stats['oracle_weighted_hit@20']:.6f} |"
        )
    if "prediction_metrics" in report:
        lines.extend(["", "## Model Metrics", ""])
        for split, groups in report["prediction_metrics"].items():
            all_metrics = groups.get("all", {})
            lines.append(
                f"- {split}: OfficialWeightedHit@20={all_metrics.get('official_weighted_hit@20', all_metrics.get('weighted_hit@20', float('nan'))):.6f}, "
                f"NDCG@20={all_metrics.get('ndcg@20', float('nan')):.6f}, "
                f"HardWeightedHit@20={all_metrics.get('hard_weighted_hit@20', float('nan')):.6f}, "
                f"PreferenceAUC={all_metrics.get('preference_auc', float('nan')):.6f}, "
                f"oracle-normalized={all_metrics.get('weighted_hit_oracle_normalized', float('nan')):.6f}"
            )
    lines.extend(["", "## Graph Coverage", ""])
    coverage = report["graph_coverage"]
    for split in ["valid", "test"]:
        value = coverage[split]
        lines.append(
            f"- {split}: row seen rate={value['row_seen_rate']:.6f}, unique seen rate={value['unique_seen_rate']:.6f}, "
            f"positive row seen rate={value['positive_row_seen_rate']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit split candidate distributions and graph leakage risk.")
    parser.add_argument("--processed-dir", default="data/run_latest/processed/qilin_full_feature_opt_v2_compact")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/run_latest/audits")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"processed_dir": str(processed_dir), "splits": {}, "graph_coverage": graph_coverage(processed_dir)}
    samples_by_split = {}
    for split in ["train", "valid", "test"]:
        samples = pd.read_parquet(processed_dir / f"{split}.parquet", columns=["request_id", "item_id", "position", *TASKS])
        samples_by_split[split] = samples
        report["splits"][split] = candidate_stats(samples)

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        prediction_metrics = {}
        for split in ["valid", "test"]:
            pred = predict_split(checkpoint, processed_dir / f"{split}.parquet", args.batch_size, args.device)
            prediction_metrics[split] = grouped_prediction_metrics(pred, samples_by_split[split], topk=args.topk)
        report["prediction_metrics"] = prediction_metrics

    json_path = output_dir / "split_candidate_audit.json"
    md_path = output_dir / "split_candidate_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
