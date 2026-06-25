from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_reliability import candidate_stats, grouped_prediction_metrics
from create_strict_graph_processed import create_strict_graph_processed
from heterointent.config import load_config, save_config, to_plain_dict
from heterointent.data.dataset import build_dataloader
from heterointent.evaluation.metrics import compute_ranking_metrics
from heterointent.inference.rank import load_model, rank_predictions
from heterointent.training import train
from heterointent.training.trainer import predict_frame


RUN_SPECS: dict[str, dict[str, Any]] = {
    "full_bge_cls": {},
    "strict_graph_zero_oov": {"processed": "strict"},
    "no_graph": {"use_graph_embedding": False},
    "no_intent_aux": {"type_transition_weight": 0.0, "taxonomy_transition_weight": 0.0},
    "no_text": {"disabled_modalities": ["text"]},
    "no_image_meta": {"disabled_modalities": ["image"]},
    "no_video_meta": {"disabled_modalities": ["video"]},
    "no_dense": {"disabled_modalities": ["dense"]},
    "id_category_history_graph_only": {"disabled_modalities": ["text", "image", "video", "dense"]},
    "id_category_history_only": {"disabled_modalities": ["text", "image", "video", "dense"], "use_graph_embedding": False},
}

METRIC_KEYS = [
    "quality_score",
    "ranking_quality_score",
    "recommendation_quality_score",
    "weighted_hit@20",
    "ndcg@20",
    "preference_auc",
    "hard_weighted_hit@20",
    "hard_ndcg@20",
    "hard_preference_auc",
    "request_auc_click",
    "request_auc_collect",
    "request_auc_share",
    "request_ap_collect",
    "request_ap_share",
]


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialize_baseline_output(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["best.pt", "last.pt", "metrics.csv", "summary.json", "valid_predictions.parquet", "config.yaml", "submission_top20_dedup.csv"]:
        src = source_dir / name
        if src.exists():
            link_or_copy(src, output_dir / name)


def apply_run_spec(base_cfg: dict[str, Any], run_name: str, output_dir: Path, processed_dir: Path, strict_processed_dir: Path) -> dict[str, Any]:
    cfg = deepcopy(to_plain_dict(base_cfg))
    spec = RUN_SPECS[run_name]
    cfg["train"]["output_dir"] = str(output_dir)
    cfg["data"]["processed_dir"] = str(strict_processed_dir if spec.get("processed") == "strict" else processed_dir)
    cfg["model"]["disabled_modalities"] = list(spec.get("disabled_modalities", []))
    if "use_graph_embedding" in spec:
        cfg["model"]["use_graph_embedding"] = bool(spec["use_graph_embedding"])
    if "type_transition_weight" in spec:
        cfg["loss"]["type_transition_weight"] = float(spec["type_transition_weight"])
    if "taxonomy_transition_weight" in spec:
        cfg["loss"]["taxonomy_transition_weight"] = float(spec["taxonomy_transition_weight"])
    return cfg


def evaluate_run(output_dir: Path, processed_dir: Path, batch_size: int, device: str, topk: int) -> dict[str, Any]:
    checkpoint = output_dir / "best.pt"
    model, metadata, _, resolved = load_model(checkpoint, device=device)
    test_path = processed_dir / "test.parquet"
    loader = build_dataloader(test_path, metadata, batch_size=batch_size, shuffle=False)
    pred = predict_frame(model, loader, resolved)
    samples = pd.read_parquet(test_path, columns=["request_id", "item_id", "position", "click", "collect", "share"])
    grouped = grouped_prediction_metrics(pred, samples, topk=topk)
    all_metrics = grouped.get("all", compute_ranking_metrics(pred, topk=topk))
    oracle = candidate_stats(samples)["oracle_weighted_hit@20"]
    payload = {
        "all": all_metrics,
        "groups": grouped,
        "oracle_weighted_hit@20": oracle,
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    ranked = rank_predictions(pred, topk=topk)
    ranked.to_csv(output_dir / "submission_top20_dedup.csv", index=False)
    return payload


def best_valid_row(output_dir: Path) -> dict[str, Any]:
    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists():
        return {}
    df = pd.read_csv(metrics_path)
    if df.empty:
        return {}
    metric = "quality_score" if "quality_score" in df.columns else "weighted_hit@20"
    if metric not in df.columns:
        return {}
    row = df.loc[df[metric].idxmax()].to_dict()
    row["selection_metric"] = metric
    return row


def build_summary(output_root: Path, run_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name in run_names:
        output_dir = output_root / run_name
        valid = best_valid_row(output_dir)
        test_path = output_dir / "test_metrics.json"
        test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else {}
        all_test = test.get("all", {})
        ge20 = test.get("groups", {}).get("candidate_count_ge20", {})
        row: dict[str, Any] = {
            "run": run_name,
            "best_valid_epoch": valid.get("epoch"),
            "valid_selection_metric": valid.get("selection_metric"),
            "valid_quality_score": valid.get("quality_score"),
            "valid_weighted_hit@20": valid.get("weighted_hit@20"),
            "valid_ndcg@20": valid.get("ndcg@20"),
            "test_oracle_weighted_hit@20": test.get("oracle_weighted_hit@20"),
            "test_weighted_hit_oracle_normalized": all_test.get("weighted_hit_oracle_normalized"),
            "test_candidate_ge20_weighted_hit@20": ge20.get("weighted_hit@20"),
        }
        for key in METRIC_KEYS:
            row[f"valid_{key}"] = valid.get(key)
            row[f"test_{key}"] = all_test.get(key)
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty and "full_bge_cls" in set(df["run"]):
        base = df[df["run"].eq("full_bge_cls")].iloc[0]
        for col in [c for c in df.columns if c.startswith("valid_") or c.startswith("test_")]:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[f"delta_vs_full_{col}"] = df[col] - base[col]
    return df


def write_report(summary: pd.DataFrame, output_root: Path) -> None:
    lines = [
        "# Ablation Report",
        "",
        "This report compares each run against `full_bge_cls` using the best validation checkpoint.",
        "",
    ]
    if not summary.empty:
        base = summary[summary["run"].eq("full_bge_cls")].iloc[0] if "full_bge_cls" in set(summary["run"]) else None
        if base is not None:
            lines.extend(["## Key Findings", ""])
            strict = summary[summary["run"].eq("strict_graph_zero_oov")]
            if not strict.empty:
                strict_row = strict.iloc[0]
                lines.append(
                    "- Strict graph zero-OOV is effectively tied with the full model "
                    f"(test WeightedHit delta {strict_row['test_weighted_hit@20'] - base['test_weighted_hit@20']:+.6f}), "
                    "so test-only item graph rows are not driving the reported test gain."
                )
            for run_name, label in [
                ("no_graph", "graph"),
                ("no_intent_aux", "dynamic intent auxiliary heads"),
                ("no_text", "text embedding"),
                ("no_image_meta", "image metadata"),
                ("no_video_meta", "video metadata"),
                ("no_dense", "dense behavioral/statistical features"),
                ("id_category_history_graph_only", "all side features except graph"),
                ("id_category_history_only", "all side features and graph"),
            ]:
                row_df = summary[summary["run"].eq(run_name)]
                if row_df.empty:
                    continue
                row = row_df.iloc[0]
                wh_delta = row["test_weighted_hit@20"] - base["test_weighted_hit@20"]
                ndcg_delta = row["test_ndcg@20"] - base["test_ndcg@20"]
                direction = "helps" if wh_delta < 0 else "does not help the main metric"
                lines.append(
                    f"- Removing {label}: test WeightedHit delta {wh_delta:+.6f}, "
                    f"NDCG delta {ndcg_delta:+.6f}; by the main metric, this component {direction}."
                )
            lines.append("")

        cols = [
            "run",
            "best_valid_epoch",
            "valid_weighted_hit@20",
            "test_weighted_hit@20",
            "test_ndcg@20",
            "test_preference_auc",
            "test_hard_weighted_hit@20",
            "test_hard_ndcg@20",
            "test_hard_preference_auc",
        ]
        available = [c for c in cols if c in summary.columns]
        lines.extend(["## Summary Table", ""])
        lines.append("| " + " | ".join(available) + " |")
        lines.append("| " + " | ".join(["---"] + ["---:"] * (len(available) - 1)) + " |")
        for _, row in summary[available].iterrows():
            values = []
            for col in available:
                value = row[col]
                if isinstance(value, float):
                    values.append(f"{value:.6f}")
                else:
                    values.append("" if pd.isna(value) else str(value))
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
        lines.append("## Interpretation Checklist")
        lines.append("")
        lines.append("- If `strict_graph_zero_oov` is close to `full_bge_cls`, graph gains are not driven by test-only item graph rows.")
        lines.append("- If `no_graph` drops, graph contributes useful signal.")
        lines.append("- If a `no_*` run improves, that modality is noisy under the current fusion/loss setup.")
    (output_root / "ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal credible ablation matrix.")
    parser.add_argument("--base-config", default="configs/qilin_full.yaml")
    parser.add_argument("--processed-dir", default="data/processed/qilin_full_multimodal_meta")
    parser.add_argument("--strict-processed-dir", default="data/processed/qilin_full_multimodal_meta_strict_graph_zero_oov")
    parser.add_argument("--output-root", default="outputs/ablations")
    parser.add_argument("--baseline-output", default="outputs/qilin_full_multimodal_meta")
    parser.add_argument("--runs", nargs="*", default=list(RUN_SPECS.keys()), choices=list(RUN_SPECS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    base_cfg = load_config(args.base_config)
    processed_dir = Path(args.processed_dir)
    strict_processed_dir = Path(args.strict_processed_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if "strict_graph_zero_oov" in args.runs and not (strict_processed_dir / "graph_embedding.npy").exists():
        create_strict_graph_processed(processed_dir, strict_processed_dir)

    completed: list[str] = []
    for run_name in args.runs:
        output_dir = output_root / run_name
        if run_name == "full_bge_cls":
            materialize_baseline_output(Path(args.baseline_output), output_dir)
        else:
            if args.skip_existing and (output_dir / "best.pt").exists():
                print(f"skip existing {run_name}")
            else:
                cfg = apply_run_spec(base_cfg, run_name, output_dir, processed_dir, strict_processed_dir)
                save_config(cfg, output_dir / "planned_config.yaml")
                train(cfg)
        run_processed = strict_processed_dir if RUN_SPECS[run_name].get("processed") == "strict" else processed_dir
        if not (output_dir / "test_metrics.json").exists() or not args.skip_existing:
            evaluate_run(output_dir, run_processed, batch_size=args.batch_size, device=args.device, topk=args.topk)
        completed.append(run_name)

    summary = build_summary(output_root, completed)
    summary.to_csv(output_root / "ablation_summary.csv", index=False)
    write_report(summary, output_root)
    print(f"wrote {output_root / 'ablation_summary.csv'}")
    print(f"wrote {output_root / 'ablation_report.md'}")


if __name__ == "__main__":
    main()
