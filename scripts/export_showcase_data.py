from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterointent.evaluation.metrics import TASKS

TASK_WEIGHTS = {"click": 0.3, "collect": 0.4, "share": 0.3}
GATE_COLUMNS = [
    "gate_item_id",
    "gate_item_type",
    "gate_taxonomy",
    "gate_position",
    "gate_taxonomy1",
    "gate_taxonomy2",
    "gate_text_fused",
    "gate_text_stat",
    "gate_image_meta",
    "gate_video_meta",
    "gate_image_emb",
    "gate_item_dense",
    "gate_ratio",
    "gate_cold_stage",
    "gate_graph",
]
ATTENTION_COLUMNS = ["attention_type_target_mass", "attention_taxonomy_target_mass"]
THUMBNAIL_COLUMNS = ["thumbnail_path", "thumbnail_source"]


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    return float(value)


def _score_weights(config: dict[str, Any]) -> dict[str, float]:
    values = config.get("evaluation", {}).get("score_weights") or config.get("loss", {}).get("task_weights") or TASK_WEIGHTS
    return {task: float(values.get(task, TASK_WEIGHTS[task])) for task in TASKS}


def _candidate_stats(samples: pd.DataFrame) -> dict[str, Any]:
    positives = samples.groupby("request_id")[list(TASKS)].sum()
    request_items = samples.groupby("request_id")["item_id"].nunique()
    row_rates = {task: float(samples[task].mean()) for task in TASKS}
    request_rates = {task: float(positives[task].gt(0).mean()) for task in TASKS}
    return {
        "rows": int(len(samples)),
        "requests": int(samples["request_id"].nunique()),
        "candidate_mean": float(request_items.mean()),
        "candidate_median": float(request_items.median()),
        "candidate_gt20_rate": float(request_items.gt(20).mean()),
        "row_positive_rate": row_rates,
        "request_positive_rate": request_rates,
    }


def _enrich_predictions(pred: pd.DataFrame, samples: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    meta_cols = ["request_id", "item_id", "item_type", "taxonomy_id", "position"]
    meta = samples[meta_cols].drop_duplicates(["request_id", "item_id"], keep="first")
    out = pred.merge(meta, on=["request_id", "item_id"], how="left")
    weights = _score_weights(config)
    out["weighted_prob_score"] = sum(weights[task] * out[f"p_{task}"].astype(float) for task in TASKS)
    out["final_score"] = out["score"].astype(float)
    return out


def _first_thumbnail_path(row: pd.Series) -> str:
    path = str(row.get("path") or "")
    if path:
        return path
    try:
        paths = json.loads(str(row.get("paths_json") or "[]"))
    except json.JSONDecodeError:
        paths = []
    return str(paths[0]) if paths else ""


def _read_thumbnail_index(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["item_id", f"{source}_thumbnail_path"])
    df = pd.read_parquet(path)
    if "item_id" not in df:
        return pd.DataFrame(columns=["item_id", f"{source}_thumbnail_path"])
    if "status" in df:
        df = df[df["status"].astype(str).isin(["encoded", "found"])]
    if df.empty:
        return pd.DataFrame(columns=["item_id", f"{source}_thumbnail_path"])
    out = df[["item_id"]].copy()
    out[f"{source}_thumbnail_path"] = df.apply(_first_thumbnail_path, axis=1)
    out = out[out[f"{source}_thumbnail_path"].astype(str).ne("")]
    return out.drop_duplicates("item_id", keep="first")


def _load_thumbnail_lookup(processed_dir: Path) -> pd.DataFrame:
    image = _read_thumbnail_index(processed_dir / "image_embedding_items.parquet", "image")
    if image.empty:
        return pd.DataFrame(columns=["item_id", *THUMBNAIL_COLUMNS])
    lookup = image.copy()
    lookup["thumbnail_path"] = lookup["image_thumbnail_path"].fillna("")
    lookup["thumbnail_source"] = np.where(lookup["thumbnail_path"].astype(str).ne(""), "image", "")
    return lookup[["item_id", *THUMBNAIL_COLUMNS]]


def _attach_thumbnails(df: pd.DataFrame, thumbnail_lookup: pd.DataFrame) -> pd.DataFrame:
    if thumbnail_lookup.empty:
        return df
    out = df.merge(thumbnail_lookup, on="item_id", how="left")
    out["thumbnail_path"] = out["thumbnail_path"].fillna("")
    out["thumbnail_source"] = out["thumbnail_source"].fillna("")
    return out


def _topk(pred: pd.DataFrame, topk: int) -> pd.DataFrame:
    frames = []
    for request_id, group in pred.groupby("request_id", sort=False):
        ranked = group.sort_values("score", ascending=False).head(topk).copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        frames.append(ranked)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _dominant_gate(row: pd.Series) -> str:
    values = {col.replace("gate_", ""): _safe_float(row.get(col), -1.0) for col in GATE_COLUMNS if col in row}
    if not values:
        return "unknown"
    return max(values, key=values.get)


def _case_explanation(row: pd.Series) -> str:
    parts = []
    if row["top20_share_hits"] > 0:
        parts.append("该请求的 Top-20 命中了 share 正样本，适合展示高价值分享行为。")
    elif row["top20_collect_hits"] > 0:
        parts.append("该请求的 Top-20 命中了 collect 正样本，适合展示收藏行为识别。")
    elif row["share_positive"] > 0:
        parts.append("该请求包含 share 正样本，可用于分析模型对分享意图的排序表现。")
    elif row["collect_positive"] > 0:
        parts.append("该请求包含 collect 正样本，可用于分析模型对收藏意图的排序表现。")
    elif row["candidate_count"] > 20:
        parts.append("该请求候选数大于 20，是更接近真实精排压力的 hard case。")
    else:
        parts.append("该请求展示了常规 Top-20 推荐排序与多目标概率输出。")
    if row["intent_strength"] >= 0.5:
        parts.append("attention target mass 较高，用户历史意图信号比较清晰。")
    if row["dominant_gate"] in {"graph", "text_fused", "text_stat", "item_dense", "ratio"}:
        parts.append(f"{row['dominant_gate']} gate 较高，说明该模态对排序贡献明显。")
    if row["dominant_gate"] in {"image_meta", "video_meta", "image_emb"}:
        parts.append(f"{row['dominant_gate']} gate is high, showing real thumbnail contribution.")
    elif row.get("thumbnail_hits", 0) > 0:
        parts.append("This request has local thumbnails in Top-20, so it is suitable for visual-result display.")
    return "".join(parts)


def _build_cases(valid_top20: pd.DataFrame, valid_samples: pd.DataFrame, limit: int) -> pd.DataFrame:
    positives = valid_samples.groupby("request_id")[list(TASKS)].sum().rename(columns={task: f"{task}_positive" for task in TASKS})
    candidate_counts = valid_samples.groupby("request_id")["item_id"].nunique().rename("candidate_count")
    top_hits = valid_top20.groupby("request_id")[list(TASKS)].sum().rename(
        columns={task: f"top20_{task}_hits" for task in TASKS}
    )
    gate_cols = [col for col in GATE_COLUMNS if col in valid_top20.columns]
    attn_cols = [col for col in ATTENTION_COLUMNS if col in valid_top20.columns]
    gate_mean = valid_top20.groupby("request_id")[gate_cols].mean() if gate_cols else pd.DataFrame(index=top_hits.index)
    attn_mean = valid_top20.groupby("request_id")[attn_cols].mean() if attn_cols else pd.DataFrame(index=top_hits.index)
    extra_frames = []
    if "thumbnail_path" in valid_top20:
        thumbnail_hits = (
            valid_top20.assign(_has_thumbnail=valid_top20["thumbnail_path"].fillna("").astype(str).ne("").astype(int))
            .groupby("request_id")["_has_thumbnail"]
            .sum()
            .rename("thumbnail_hits")
        )
        extra_frames.append(thumbnail_hits)
    cases = pd.concat([positives, candidate_counts, top_hits, gate_mean, attn_mean, *extra_frames], axis=1).fillna(0).reset_index()
    if attn_cols:
        cases["intent_strength"] = cases[attn_cols].mean(axis=1)
    else:
        cases["intent_strength"] = 0.0
    visual_gate_cols = [col for col in ["gate_image_meta", "gate_video_meta", "gate_image_emb"] if col in cases]
    cases["visual_gate_strength"] = cases[visual_gate_cols].sum(axis=1) if visual_gate_cols else 0.0
    cases["dominant_gate"] = cases.apply(_dominant_gate, axis=1)
    cases["case_score"] = (
        100 * cases["top20_share_hits"].gt(0).astype(int)
        + 80 * cases["top20_collect_hits"].gt(0).astype(int)
        + 45 * cases["share_positive"].gt(0).astype(int)
        + 35 * cases["collect_positive"].gt(0).astype(int)
        + 20 * cases["candidate_count"].gt(20).astype(int)
        + 15 * cases["dominant_gate"].isin(["image_meta", "video_meta", "image_emb"]).astype(int)
        + 15 * cases.get("thumbnail_hits", pd.Series(0, index=cases.index)).gt(0).astype(int)
        + 10 * cases["visual_gate_strength"].clip(0, 1)
        + 10 * cases["intent_strength"].clip(0, 1)
    )
    cases = cases.sort_values(["case_score", "candidate_count", "intent_strength"], ascending=False).head(limit).copy()
    cases["tags"] = cases.apply(_case_tags, axis=1)
    cases["explanation"] = cases.apply(_case_explanation, axis=1)
    keep = [
        "request_id",
        "tags",
        "explanation",
        "case_score",
        "candidate_count",
        "click_positive",
        "collect_positive",
        "share_positive",
        "top20_click_hits",
        "top20_collect_hits",
        "top20_share_hits",
        "thumbnail_hits",
        "visual_gate_strength",
        "intent_strength",
        "dominant_gate",
    ]
    return cases[keep]


def _case_tags(row: pd.Series) -> str:
    tags = []
    if row["top20_share_hits"] > 0 or row["share_positive"] > 0:
        tags.append("share")
    if row["top20_collect_hits"] > 0 or row["collect_positive"] > 0:
        tags.append("collect")
    if row["candidate_count"] > 20:
        tags.append("hard")
    if row["intent_strength"] >= 0.5:
        tags.append("intent")
    if row.get("dominant_gate", "") in {"image_meta", "video_meta", "image_emb"} or row.get("thumbnail_hits", 0) > 0:
        tags.append("visual")
    if not tags:
        tags.append("general")
    return ",".join(tags)


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_bar_chart(path: Path, title: str, labels: list[str], values: list[float], color: str = "#2f6fed") -> None:
    width, height = 980, 560
    margin_l, margin_r, margin_t, margin_b = 150, 70, 80, 80
    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)
    title_font = _load_font(30)
    label_font = _load_font(20)
    small_font = _load_font(18)
    draw.text((margin_l, 25), title, fill="#172033", font=title_font)
    max_value = max(max(values), 1e-9)
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b
    bar_gap = 18
    bar_h = max(22, int((chart_h - bar_gap * (len(labels) - 1)) / max(len(labels), 1)))
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = margin_t + idx * (bar_h + bar_gap)
        bar_w = int(chart_w * value / max_value)
        draw.text((20, y + 4), label, fill="#172033", font=label_font)
        draw.rectangle((margin_l, y, margin_l + chart_w, y + bar_h), fill="#eef2f7")
        draw.rectangle((margin_l, y, margin_l + bar_w, y + bar_h), fill=color)
        draw.text((margin_l + bar_w + 10, y + 3), f"{value:.4f}", fill="#172033", font=small_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _write_charts(output_dir: Path, overview: dict[str, Any], valid_top20: pd.DataFrame) -> dict[str, str]:
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    train_stats = overview["dataset"]["train"]
    imbalance_path = charts_dir / "imbalance.png"
    _draw_bar_chart(
        imbalance_path,
        "训练集正样本比例",
        [f"{task} 行级" for task in TASKS] + [f"{task} 请求级" for task in TASKS],
        [train_stats["row_positive_rate"][task] for task in TASKS]
        + [train_stats["request_positive_rate"][task] for task in TASKS],
        color="#d94f70",
    )

    valid_metrics = overview["metrics"]["valid_best"]
    metrics_path = charts_dir / "metrics.png"
    metric_names = ["ndcg@20", "preference_auc", "request_auc_collect", "request_auc_share", "request_ap_collect", "request_ap_share"]
    _draw_bar_chart(
        metrics_path,
        "温和版核心解释指标",
        metric_names,
        [_safe_float(valid_metrics.get(name), 0.0) for name in metric_names],
        color="#2f6fed",
    )

    gate_cols = [col for col in GATE_COLUMNS if col in valid_top20.columns]
    gate_path = charts_dir / "modality_gate.png"
    if gate_cols:
        means = valid_top20[gate_cols].mean().sort_values(ascending=False)
        _draw_bar_chart(
            gate_path,
            "Top-20 平均模态贡献",
            [col.replace("gate_", "") for col in means.index.tolist()],
            [float(v) for v in means.tolist()],
            color="#1f9d75",
        )

    label_path = charts_dir / "top20_labels.png"
    _draw_bar_chart(
        label_path,
        "Valid Top-20 正样本命中行数",
        [f"top20 {task}" for task in TASKS],
        [float(valid_top20[task].sum()) for task in TASKS],
        color="#7d5fff",
    )
    return {
        "imbalance": str(imbalance_path.relative_to(output_dir)).replace("\\", "/"),
        "metrics": str(metrics_path.relative_to(output_dir)).replace("\\", "/"),
        "modality_gate": str(gate_path.relative_to(output_dir)).replace("\\", "/"),
        "top20_labels": str(label_path.relative_to(output_dir)).replace("\\", "/"),
    }


def _best_metric_row(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    df = pd.read_csv(metrics_path)
    if df.empty:
        return {}
    metric = "native_selection_score" if "native_selection_score" in df.columns else "official_weighted_hit@20"
    if metric not in df:
        return {}
    row = df.loc[df[metric].idxmax()].to_dict()
    out = {key: _safe_float(value, float("nan")) for key, value in row.items()}
    out["selection_metric"] = metric
    return out


def _write_report(output_dir: Path, overview: dict[str, Any], cases: pd.DataFrame) -> None:
    valid = overview["metrics"]["valid_best"]
    train = overview["dataset"]["train"]
    lines = [
        "# HeteroIntent 推荐诊断台答辩摘要",
        "",
        "## 展示主线",
        "",
        "- 任务不是单纯预测点击，而是在同一 request 候选集中完成 click / collect / share 多目标 Top-20 排序。",
        "- collect/share 是极稀疏高价值行为，因此需要展示模型如何识别高价值意图，而不只展示一个总分。",
        "- 当前主线使用官方线性分数、native 选模指标和 hard/rare 分档诊断，避免被简单请求的 Top-20 饱和现象误导。",
        "",
        "## 数据挑战",
        "",
        "| 行为 | 行级正样本率 | 请求级正样本率 |",
        "| --- | ---: | ---: |",
    ]
    for task in TASKS:
        lines.append(f"| {task} | {train['row_positive_rate'][task]:.6f} | {train['request_positive_rate'][task]:.6f} |")
    lines.extend(
        [
            "",
            "## 当前模型验证指标",
            "",
            "| 指标 | 数值 |",
            "| --- | ---: |",
        ]
    )
    metric_keys = [
        "weighted_hit@20",
        "ndcg@20",
        "preference_auc",
        "hard_weighted_hit@20",
        "hard_ndcg@20",
        "request_auc_collect",
        "request_auc_share",
        "request_ap_collect",
        "request_ap_share",
    ]
    for key in metric_keys:
        if key in valid and not math.isnan(_safe_float(valid[key], float("nan"))):
            lines.append(f"| {key} | {_safe_float(valid[key]):.6f} |")
    lines.extend(["", "## 精选展示案例", "", "| request_id | 标签 | 说明 |", "| ---: | --- | --- |"])
    for row in cases.head(6).itertuples(index=False):
        lines.append(f"| {int(row.request_id)} | {row.tags} | {row.explanation} |")
    lines.extend(
        [
            "",
            "## 现场演示建议",
            "",
            "1. 先打开 Overview，展示样本不均衡和核心指标。",
            "2. 切到 Showcase Cases，选择 share 或 collect 标签案例。",
            "3. 在 Request Explorer 中点击 Top-20 item，展示三任务概率、官方线性分数和模态 gate。",
            "4. 用 Why This Rank 说明模型不是黑箱，而是能解释分数来源和意图信号。",
        ]
    )
    (output_dir / "defense_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_showcase(args: argparse.Namespace) -> dict[str, Any]:
    processed_dir = Path(args.processed_dir)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    config = _read_yaml(run_dir / "config.yaml") or _read_yaml(Path(args.config))
    summary = _read_json(run_dir / "summary.json", {})
    valid_pred_path = run_dir / args.valid_predictions
    submission_path = run_dir / args.submission
    if not valid_pred_path.exists():
        raise FileNotFoundError(f"Missing valid predictions: {valid_pred_path}")
    if not submission_path.exists():
        raise FileNotFoundError(f"Missing submission CSV: {submission_path}")

    valid_pred = pd.read_parquet(valid_pred_path)
    valid_samples = pd.read_parquet(
        processed_dir / "valid.parquet",
        columns=["request_id", "item_id", "item_type", "taxonomy_id", "position", *TASKS],
    )
    train_samples = pd.read_parquet(processed_dir / "train.parquet", columns=["request_id", "item_id", *TASKS])
    test_samples = pd.read_parquet(processed_dir / "test.parquet", columns=["request_id", "item_id", "item_type", "taxonomy_id", "position"])
    valid_enriched = _enrich_predictions(valid_pred, valid_samples, config)
    valid_top20 = _topk(valid_enriched, args.topk)
    test_top20 = pd.read_csv(submission_path)
    test_top20 = test_top20.merge(test_samples.drop_duplicates(["request_id", "item_id"]), on=["request_id", "item_id"], how="left")
    test_top20["final_score"] = test_top20["score"].astype(float)
    thumbnail_dir = Path(args.thumbnail_index_dir) if args.thumbnail_index_dir else processed_dir
    thumbnail_lookup = _load_thumbnail_lookup(thumbnail_dir)
    valid_top20 = _attach_thumbnails(valid_top20, thumbnail_lookup)
    test_top20 = _attach_thumbnails(test_top20, thumbnail_lookup)

    cases = _build_cases(valid_top20, valid_samples, args.max_cases)
    valid_best = _best_metric_row(run_dir / "metrics.csv")
    overview = {
        "title": "HeteroIntent 推荐诊断台",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "processed_dir": str(processed_dir),
        "topk": int(args.topk),
        "score_weights": _score_weights(config),
        "dataset": {
            "train": _candidate_stats(train_samples),
            "valid": _candidate_stats(valid_samples),
            "test": {
                "rows": int(len(test_samples)),
                "requests": int(test_samples["request_id"].nunique()),
                "candidate_mean": float(test_samples.groupby("request_id")["item_id"].nunique().mean()),
            },
        },
        "metrics": {"valid_best": valid_best, "summary": summary},
        "visual": {
            "thumbnail_index_dir": str(thumbnail_dir),
            "thumbnail_items": int(thumbnail_lookup["thumbnail_path"].astype(str).ne("").sum()) if not thumbnail_lookup.empty else 0,
        },
        "artifacts": {},
    }
    charts = _write_charts(output_dir, overview, valid_top20)
    overview["artifacts"]["charts"] = charts

    valid_cols = _showcase_columns(valid_top20)
    test_cols = [
        col
        for col in [
            "request_id",
            "rank",
            "item_id",
            "score",
            "final_score",
            "p_click",
            "p_collect",
            "p_share",
            "item_type",
            "taxonomy_id",
            "position",
            *THUMBNAIL_COLUMNS,
        ]
        if col in test_top20
    ]
    valid_top20[valid_cols].to_parquet(output_dir / "valid_top20.parquet", index=False)
    test_top20[test_cols].to_parquet(output_dir / "test_top20.parquet", index=False)
    cases.to_parquet(output_dir / "showcase_cases.parquet", index=False)
    cases.to_json(output_dir / "showcase_cases.json", orient="records", force_ascii=False, indent=2)
    (output_dir / "overview.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(output_dir, overview, cases)
    print(f"wrote showcase data to {output_dir}")
    print(f"cases: {len(cases)}, valid_top20 rows: {len(valid_top20)}, test_top20 rows: {len(test_top20)}")
    return overview


def _showcase_columns(df: pd.DataFrame) -> list[str]:
    required = [
        "request_id",
        "rank",
        "item_id",
        "score",
        "final_score",
        "weighted_prob_score",
        "p_click",
        "p_collect",
        "p_share",
        "click",
        "collect",
        "share",
        "item_type",
        "taxonomy_id",
        "position",
        "target_item_type",
        "target_taxonomy_id",
        "hist_dominant_item_type",
        "hist_dominant_taxonomy_id",
        "is_type_shift",
        "is_taxonomy_shift",
        "has_intent_target",
        *ATTENTION_COLUMNS,
        *GATE_COLUMNS,
        *THUMBNAIL_COLUMNS,
    ]
    return [col for col in required if col in df.columns]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export lightweight offline showcase data for the HeteroIntent demo app.")
    parser.add_argument("--processed-dir", default="data/run_latest/processed/qilin_full_feature_opt_v2_compact")
    parser.add_argument("--run-dir", default="outputs/run_latest/qilin_feature_opt_v2_history_compact")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default="configs/qilin_feature_opt_v2_history_compact.yaml")
    parser.add_argument("--output-dir", default="outputs/showcase_run_latest")
    parser.add_argument("--valid-predictions", default="valid_predictions.parquet")
    parser.add_argument("--submission", default="submission_top20_dedup.csv")
    parser.add_argument("--thumbnail-index-dir", default=None, help="Directory containing image_embedding_items.parquet.")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--max-cases", type=int, default=120)
    args = parser.parse_args()
    export_showcase(args)


if __name__ == "__main__":
    main()
