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

from heterointent.evaluation.metrics import TASKS, compute_ranking_metrics
from heterointent.inference.rank import rank_predictions

DEFAULT_METRIC_KEYS = [
    "native_selection_score",
    "official_weighted_hit@20",
    "hard_official_capture",
    "sparse_ap",
    "sparse_recall",
    "rare_score",
    "hard_weighted_hit@20",
    "hard_ndcg@20",
    "hard_preference_auc",
    "hard_request_ap_collect",
    "hard_request_ap_share",
    "hard_recall_collect@20",
    "hard_recall_share@20",
    "candidate_count",
    "candidate_count_gt_topk_rate",
    "num_requests",
    "num_rows",
]


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def read_unique_candidates(path: Path) -> pd.DataFrame:
    samples = pd.read_parquet(path, columns=["request_id", "item_id", *TASKS])
    return samples.groupby(["request_id", "item_id"], as_index=False, sort=False)[list(TASKS)].max()


def random_predictions(samples: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pred = samples.copy()
    score = rng.random(len(pred)).astype("float32")
    pred["score"] = score
    for task in TASKS:
        pred[f"p_{task}"] = score
    return pred


def summarize_runs(runs: list[dict[str, float]]) -> dict[str, dict[str, float | None]]:
    keys = sorted({key for run in runs for key in run if key != "seed"})
    summary: dict[str, dict[str, float | None]] = {}
    for key in keys:
        values = np.array([run[key] for run in runs if key in run and np.isfinite(run[key])], dtype="float64")
        if len(values) == 0:
            summary[key] = {"mean": None, "std": None}
            continue
        summary[key] = {
            "mean": _json_float(values.mean()),
            "std": _json_float(values.std(ddof=0)),
        }
    return summary


def write_summary_csv(summary_by_split: dict[str, dict[str, dict[str, float | None]]], path: Path) -> None:
    rows = []
    for split, summary in summary_by_split.items():
        for metric, values in summary.items():
            rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "mean": values["mean"],
                    "std": values["std"],
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a pure random ranking baseline on Qilin processed splits.")
    parser.add_argument("--processed-dir", default="data/run_latest/processed/qilin_full_feature_opt_v2_compact")
    parser.add_argument("--output-dir", default="outputs/run_latest/random_baseline")
    parser.add_argument("--splits", nargs="+", default=["valid", "test"], choices=["train", "valid", "test"])
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--export-top20-split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--export-top20-seed", type=int, default=2026)
    parser.add_argument("--no-export-top20", action="store_true")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed + offset for offset in range(args.num_seeds)]

    report: dict[str, Any] = {
        "processed_dir": str(processed_dir),
        "topk": args.topk,
        "seeds": seeds,
        "deduplication": "request_id,item_id labels are max-pooled before random scoring",
        "splits": {},
    }
    summary_by_split: dict[str, dict[str, dict[str, float | None]]] = {}

    cached_samples: dict[str, pd.DataFrame] = {}
    for split in args.splits:
        samples = read_unique_candidates(processed_dir / f"{split}.parquet")
        cached_samples[split] = samples
        runs: list[dict[str, float]] = []
        for seed in seeds:
            pred = random_predictions(samples, seed)
            metrics = compute_ranking_metrics(pred, topk=args.topk, include_diagnostics=True)
            selected = {key: float(metrics[key]) for key in DEFAULT_METRIC_KEYS if key in metrics}
            selected["seed"] = float(seed)
            runs.append(selected)
        summary = summarize_runs(runs)
        report["splits"][split] = {"runs": runs, "summary": summary}
        summary_by_split[split] = summary

    if not args.no_export_top20:
        split = args.export_top20_split
        samples = cached_samples[split] if split in cached_samples else read_unique_candidates(processed_dir / f"{split}.parquet")
        pred = random_predictions(samples, args.export_top20_seed)
        ranked = rank_predictions(pred, topk=args.topk)
        top20_path = output_dir / f"random_{split}_top{args.topk}_seed{args.export_top20_seed}.csv"
        ranked.to_csv(top20_path, index=False)
        report["exported_top20"] = str(top20_path)

    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "metrics_summary.csv"
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(summary_by_split, summary_path)
    print(f"wrote {metrics_path}")
    print(f"wrote {summary_path}")
    if "exported_top20" in report:
        print(f"wrote {report['exported_top20']}")


if __name__ == "__main__":
    main()
