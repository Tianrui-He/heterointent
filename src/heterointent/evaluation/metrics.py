from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

TASKS = ("click", "collect", "share")
TASK_WEIGHTS = {"click": 0.3, "collect": 0.4, "share": 0.3}
RANKING_QUALITY_WEIGHTS = {
    "hard_ndcg@20": 0.35,
    "hard_preference_auc": 0.30,
    "ndcg@20": 0.20,
    "preference_auc": 0.15,
}
RECOMMENDATION_QUALITY_WEIGHTS = {
    "request_ap_collect": 0.30,
    "request_ap_share": 0.30,
    "request_auc_collect": 0.20,
    "request_auc_share": 0.15,
    "request_auc_click": 0.05,
}
CORE_METRIC_KEYS = [
    "topk",
    "num_requests",
    "num_rows",
    "candidate_count",
    "candidate_count_gt_topk_rate",
    "quality_score",
    "ranking_quality_score",
    "recommendation_quality_score",
    "weighted_hit@20",
    "hard_weighted_hit@20",
    "ndcg@20",
    "hard_ndcg@20",
    "preference_auc",
    "hard_preference_auc",
    "request_auc_click",
    "request_auc_collect",
    "request_auc_share",
    "request_ap_collect",
    "request_ap_share",
]


def _finite_mask(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    return np.isfinite(y_score) & np.isfinite(y_true)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    mask = _finite_mask(y_true, y_score)
    if not bool(mask.any()):
        return float("nan")
    y_true = y_true[mask]
    y_score = y_score[mask]
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    mask = _finite_mask(y_true, y_score)
    if not bool(mask.any()):
        return float("nan")
    y_true = y_true[mask]
    y_score = y_score[mask]
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _binary_pair_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool, copy=False)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    comparisons = pos[:, None] - neg[None, :]
    return float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())


def _graded_pair_auc(scores: np.ndarray, relevance: np.ndarray) -> float:
    rel_diff = relevance[:, None] - relevance[None, :]
    score_diff = scores[:, None] - scores[None, :]
    mask = rel_diff > 0
    if not bool(mask.any()):
        return float("nan")
    return float((score_diff[mask] > 0).mean() + 0.5 * (score_diff[mask] == 0).mean())


def _dcg(relevances: list[float]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def _safe_mean(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(values.mean())


def _weighted_available(metrics: dict[str, float], prefix: str) -> float:
    total = 0.0
    weight_sum = 0.0
    for task, weight in TASK_WEIGHTS.items():
        value = metrics.get(f"{prefix}_{task}")
        if value is None or not np.isfinite(value):
            continue
        total += weight * value
        weight_sum += weight
    return float(total / weight_sum) if weight_sum > 0 else float("nan")


def _weighted_metric_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    total = 0.0
    weight_sum = 0.0
    for key, weight in weights.items():
        value = metrics.get(key)
        if value is None or not np.isfinite(value):
            continue
        total += weight * value
        weight_sum += weight
    return float(total / weight_sum) if weight_sum > 0 else float("nan")


def _filter_core_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: metrics[key] for key in CORE_METRIC_KEYS if key in metrics}


def deduplicate_request_items(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one scored row per request-item pair."""

    keys = ["request_id", "item_id"]
    if df.empty or not set(keys).issubset(df.columns):
        return df
    if not df.duplicated(keys).any():
        return df

    label_cols = [task for task in TASKS if task in df.columns]
    sort_cols = keys + (["score"] if "score" in df.columns else [])
    ascending = [True, True] + ([False] if "score" in df.columns else [])
    ordered = df.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    first_cols = [col for col in ordered.columns if col not in label_cols]
    first = ordered.drop_duplicates(keys, keep="first")[first_cols]
    if not label_cols:
        return first.reset_index(drop=True)
    labels = df.groupby(keys, as_index=False)[label_cols].max()
    return first.merge(labels, on=keys, how="left").reset_index(drop=True)


def compute_ranking_metrics(df: pd.DataFrame, topk: int = 20, include_diagnostics: bool = False) -> dict[str, float]:
    """Compute representative request-level metrics for ranking quality."""

    if df.empty:
        return {}
    required = {"request_id", "item_id", "score", *TASKS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}")

    raw_rows = len(df)
    df = deduplicate_request_items(df)
    duplicate_rows_removed = raw_rows - len(df)
    has_dynamic = {"has_intent_target", "is_type_shift", "is_taxonomy_shift"}.issubset(df.columns)
    gate_cols = sorted(col for col in df.columns if col.startswith("gate_"))

    per_request = []
    for _, group in df.groupby("request_id", sort=False):
        ranked = group.sort_values("score", ascending=False).head(topk)
        first = group.iloc[0]
        row: dict[str, float] = {}
        row["candidate_count"] = float(len(group))
        row["is_trivial_topk_request"] = float(len(group) <= topk)
        row["is_hard_topk_request"] = float(len(group) > topk)
        weighted_rel_all = (
            TASK_WEIGHTS["click"] * group["click"].to_numpy()
            + TASK_WEIGHTS["collect"] * group["collect"].to_numpy()
            + TASK_WEIGHTS["share"] * group["share"].to_numpy()
        )
        weighted_rel_top = (
            TASK_WEIGHTS["click"] * ranked["click"].to_numpy()
            + TASK_WEIGHTS["collect"] * ranked["collect"].to_numpy()
            + TASK_WEIGHTS["share"] * ranked["share"].to_numpy()
        )
        ideal = sorted(weighted_rel_all.tolist(), reverse=True)[:topk]
        row["ndcg@20"] = _dcg(weighted_rel_top.tolist()) / max(_dcg(ideal), 1e-12)
        row["request_pair_auc_weighted"] = _graded_pair_auc(group["score"].to_numpy(), weighted_rel_all)
        for task in TASKS:
            positives = int(group[task].sum())
            top_positives = int(ranked[task].sum())
            has_positive = positives > 0
            hit = float(top_positives > 0)
            recall = float(top_positives / positives) if has_positive else np.nan
            row[f"hit_{task}@20"] = hit
            row[f"hit_{task}_positive@20"] = hit if has_positive else np.nan
            row[f"recall_{task}@20"] = recall
            row[f"recall_{task}_overall@20"] = recall if has_positive else 0.0
            row[f"has_{task}_positive"] = float(has_positive)
            score_col = f"p_{task}"
            if score_col in group.columns:
                task_scores = group[score_col].to_numpy()
                task_labels = group[task].to_numpy()
                row[f"request_auc_{task}"] = _binary_pair_auc(task_scores, task_labels)
                row[f"request_ap_{task}"] = (
                    _safe_ap(task_labels, task_scores)
                    if len(np.unique(task_labels)) >= 2
                    else float("nan")
                )
            else:
                row[f"request_auc_{task}"] = np.nan
                row[f"request_ap_{task}"] = np.nan
        row["weighted_hit@20"] = sum(TASK_WEIGHTS[t] * row[f"hit_{t}@20"] for t in TASKS)
        for col in gate_cols:
            row[f"top20_mean_{col}"] = float(ranked[col].mean())

        if has_dynamic:
            has_target = float(first.get("has_intent_target", 0)) > 0
            is_type_shift = float(first.get("is_type_shift", 0)) > 0
            is_taxonomy_shift = float(first.get("is_taxonomy_shift", 0)) > 0
            row["has_intent_target"] = float(has_target)
            row["is_type_shift"] = float(is_type_shift)
            row["is_taxonomy_shift"] = float(is_taxonomy_shift)
            row["is_intent_shift"] = float(is_type_shift or is_taxonomy_shift)
            if has_target:
                for metric_col, output_col in [
                    ("intent_type_hit@1", "intent_type_acc@1"),
                    ("intent_type_hit@2", "intent_type_acc@2"),
                    ("intent_taxonomy_hit@1", "intent_taxonomy_acc@1"),
                    ("intent_taxonomy_hit@5", "intent_taxonomy_acc@5"),
                    ("intent_taxonomy_mrr", "intent_taxonomy_mrr"),
                ]:
                    row[output_col] = float(first[metric_col]) if metric_col in first else np.nan
                if "attention_type_target_mass" in ranked:
                    row["attention_type_target_mass@20"] = float(ranked["attention_type_target_mass"].mean())
                if "attention_taxonomy_target_mass" in ranked:
                    row["attention_taxonomy_target_mass@20"] = float(ranked["attention_taxonomy_target_mass"].mean())
                if "attention_type_target_mass@20" in row and "attention_taxonomy_target_mass@20" in row:
                    row["attention_target_mass"] = (
                        row["attention_type_target_mass@20"] + row["attention_taxonomy_target_mass@20"]
                    ) * 0.5
            else:
                row["intent_type_acc@1"] = np.nan
                row["intent_type_acc@2"] = np.nan
                row["intent_taxonomy_acc@1"] = np.nan
                row["intent_taxonomy_acc@5"] = np.nan
                row["intent_taxonomy_mrr"] = np.nan
                row["attention_type_target_mass@20"] = np.nan
                row["attention_taxonomy_target_mass@20"] = np.nan
                row["attention_target_mass"] = np.nan
        per_request.append(row)

    per_request_df = pd.DataFrame(per_request)
    metrics = per_request_df.mean(numeric_only=True).to_dict()
    metrics["topk"] = float(topk)
    metrics["candidate_count_le_topk_rate"] = float(per_request_df["candidate_count"].le(topk).mean())
    metrics["candidate_count_gt_topk_rate"] = float(per_request_df["candidate_count"].gt(topk).mean())
    metrics["candidate_count_le20_rate"] = metrics["candidate_count_le_topk_rate"]
    metrics["candidate_count_gt20_rate"] = metrics["candidate_count_gt_topk_rate"]
    hard_requests = per_request_df[per_request_df["is_hard_topk_request"].gt(0)]
    metrics["hard_topk_requests"] = float(len(hard_requests))
    metrics["hard_topk_request_rate"] = float(len(hard_requests) / max(len(per_request_df), 1))
    hard_metric_cols = [
        "weighted_hit@20",
        "ndcg@20",
        "request_pair_auc_weighted",
        *[f"hit_{task}@20" for task in TASKS],
        *[f"hit_{task}_positive@20" for task in TASKS],
        *[f"recall_{task}@20" for task in TASKS],
        *[f"request_auc_{task}" for task in TASKS],
        *[f"request_ap_{task}" for task in TASKS],
    ]
    for col in hard_metric_cols:
        if col in hard_requests:
            metrics[f"hard_{col}"] = _safe_mean(hard_requests[col])
    for task in TASKS:
        score_col = f"p_{task}"
        if score_col in df.columns:
            metrics[f"auc_{task}"] = _safe_auc(df[task].to_numpy(), df[score_col].to_numpy())
            metrics[f"ap_{task}"] = _safe_ap(df[task].to_numpy(), df[score_col].to_numpy())
        metrics[f"positive_requests_{task}"] = float(df.groupby("request_id")[task].sum().gt(0).sum())
        metrics[f"positive_request_rate_{task}"] = float(metrics[f"has_{task}_positive"])
        request_auc_col = f"request_auc_{task}"
        request_ap_col = f"request_ap_{task}"
        metrics[f"request_auc_requests_{task}"] = float(per_request_df[request_auc_col].notna().sum())
        metrics[f"request_auc_request_rate_{task}"] = float(per_request_df[request_auc_col].notna().mean())
        metrics[f"request_ap_requests_{task}"] = float(per_request_df[request_ap_col].notna().sum())
        metrics[f"request_ap_request_rate_{task}"] = float(per_request_df[request_ap_col].notna().mean())
    for col in gate_cols:
        metrics[f"mean_{col}"] = float(df[col].mean())
    metrics["request_auc_weighted"] = _weighted_available(metrics, "request_auc")
    metrics["request_ap_weighted"] = _weighted_available(metrics, "request_ap")
    metrics["hard_request_auc_weighted"] = _weighted_available(metrics, "hard_request_auc")
    metrics["hard_request_ap_weighted"] = _weighted_available(metrics, "hard_request_ap")
    metrics["preference_auc"] = metrics.get("request_pair_auc_weighted", float("nan"))
    metrics["hard_preference_auc"] = metrics.get("hard_request_pair_auc_weighted", float("nan"))
    metrics["ranking_quality_score"] = _weighted_metric_score(metrics, RANKING_QUALITY_WEIGHTS)
    metrics["recommendation_quality_score"] = _weighted_metric_score(metrics, RECOMMENDATION_QUALITY_WEIGHTS)
    metrics["quality_score"] = _weighted_metric_score(
        metrics,
        {
            "ranking_quality_score": 0.60,
            "recommendation_quality_score": 0.40,
        },
    )

    if has_dynamic:
        target_requests = per_request_df[per_request_df["has_intent_target"].gt(0)]
        shift_requests = target_requests[target_requests["is_intent_shift"].gt(0)]
        stable_requests = target_requests[target_requests["is_intent_shift"].le(0)]
        type_shift_requests = target_requests[target_requests["is_type_shift"].gt(0)]
        taxonomy_shift_requests = target_requests[target_requests["is_taxonomy_shift"].gt(0)]
        metrics["intent_target_requests"] = float(len(target_requests))
        metrics["intent_target_request_rate"] = float(len(target_requests) / max(len(per_request_df), 1))
        metrics["intent_shift_requests"] = float(len(shift_requests))
        metrics["intent_shift_request_rate"] = (
            float(len(shift_requests) / max(len(target_requests), 1)) if len(target_requests) else float("nan")
        )
        metrics["type_shift_requests"] = float(len(type_shift_requests))
        metrics["taxonomy_shift_requests"] = float(len(taxonomy_shift_requests))
        metrics["shift_type_hit@1"] = (
            _safe_mean(type_shift_requests["intent_type_acc@1"]) if "intent_type_acc@1" in type_shift_requests else float("nan")
        )
        metrics["shift_taxonomy_hit@5"] = (
            _safe_mean(taxonomy_shift_requests["intent_taxonomy_acc@5"])
            if "intent_taxonomy_acc@5" in taxonomy_shift_requests
            else float("nan")
        )
        metrics["ranking_weighted_hit@20_shift"] = _safe_mean(shift_requests["weighted_hit@20"])
        metrics["ranking_weighted_hit@20_stable"] = _safe_mean(stable_requests["weighted_hit@20"])

    metrics["num_requests"] = float(df["request_id"].nunique())
    metrics["num_rows"] = float(len(df))
    metrics["num_raw_rows"] = float(raw_rows)
    metrics["num_duplicate_rows_removed"] = float(duplicate_rows_removed)
    metrics = {k: float(v) for k, v in metrics.items()}
    if include_diagnostics:
        return metrics
    return _filter_core_metrics(metrics)
