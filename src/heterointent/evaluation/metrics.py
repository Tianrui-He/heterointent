from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

TASKS = ("click", "collect", "share")
TASK_WEIGHTS = {"click": 0.3, "collect": 0.4, "share": 0.3}
NATIVE_SELECTION_WEIGHTS = {
    "hard_official_capture": 0.30,
    "hard_ndcg@20": 0.25,
    "hard_preference_auc": 0.20,
    "sparse_ap": 0.15,
    "sparse_recall": 0.10,
}
CORE_METRIC_KEYS = [
    "topk",
    "num_requests",
    "num_rows",
    "candidate_count",
    "candidate_count_gt_topk_rate",
    "native_selection_score",
    "official_weighted_hit@20",
    "hard_weighted_hit@20",
    "hard_official_capture",
    "hard_ndcg@20",
    "hard_preference_auc",
    "sparse_ap",
    "sparse_recall",
    "rare_score",
    "rare_requests",
    "oracle_score",
    "oracle_score_eligible",
    "topk_boundary_success_click",
    "topk_boundary_success_collect",
    "topk_boundary_success_share",
    "topk_boundary_margin_click",
    "topk_boundary_margin_collect",
    "topk_boundary_margin_share",
    "hard_request_ap_collect",
    "hard_request_ap_share",
    "hard_recall_collect@20",
    "hard_recall_share@20",
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
        total += float(weight) * float(value)
        weight_sum += float(weight)
    return float(total / weight_sum) if weight_sum > 0 else float("nan")


def _filter_core_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: metrics[key] for key in CORE_METRIC_KEYS if key in metrics}


def _combine_available(parts: list[tuple[float | None, float]]) -> float:
    total = 0.0
    weight_sum = 0.0
    for value, weight in parts:
        if value is None or not np.isfinite(value):
            continue
        total += float(value) * float(weight)
        weight_sum += float(weight)
    return float(total / weight_sum) if weight_sum > 0 else float("nan")


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


def per_request_metric_frame(df: pd.DataFrame, topk: int = 20) -> pd.DataFrame:
    """Compute request-level native metrics.

    Qilin has many requests whose candidate count is no larger than Top-K. This
    frame keeps those requests visible while marking the truly competitive hard
    requests separately.
    """

    if df.empty:
        return pd.DataFrame()
    required = {"request_id", "item_id", "score", *TASKS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}")

    df = deduplicate_request_items(df)
    has_dynamic = {"has_intent_target", "is_type_shift", "is_taxonomy_shift"}.issubset(df.columns)
    gate_cols = sorted(col for col in df.columns if col.startswith("gate_"))
    per_request = []

    for request_id, group in df.groupby("request_id", sort=False):
        ranked = group.sort_values("score", ascending=False).head(topk)
        first = group.iloc[0]
        row: dict[str, float] = {}
        row["request_id"] = float(request_id)
        candidate_count = len(group)
        row["candidate_count"] = float(candidate_count)
        row["is_trivial_topk_request"] = float(candidate_count <= topk)
        row["is_hard_topk_request"] = float(candidate_count > topk)
        row["candidate_bucket_le20"] = float(candidate_count <= topk)
        row["candidate_bucket_21_40"] = float(topk < candidate_count <= 40)
        row["candidate_bucket_gt40"] = float(candidate_count > 40)

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
                row[f"request_ap_{task}"] = _safe_ap(task_labels, task_scores) if len(np.unique(task_labels)) >= 2 else float("nan")
            else:
                row[f"request_auc_{task}"] = np.nan
                row[f"request_ap_{task}"] = np.nan

            if candidate_count > topk and has_positive:
                threshold = float(ranked["score"].iloc[-1])
                max_pos_score = float(group.loc[group[task].gt(0), "score"].max())
                margin = max_pos_score - threshold
                row[f"topk_boundary_margin_{task}"] = margin
                row[f"topk_boundary_success_{task}"] = float(margin >= -1e-12)
            else:
                row[f"topk_boundary_margin_{task}"] = np.nan
                row[f"topk_boundary_success_{task}"] = np.nan

        row["weighted_hit@20"] = sum(TASK_WEIGHTS[t] * row[f"hit_{t}@20"] for t in TASKS)
        row["official_weighted_hit@20"] = row["weighted_hit@20"]

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

    return pd.DataFrame(per_request)


def summarize_per_request_metrics(per_request_df: pd.DataFrame) -> dict[str, float]:
    if per_request_df.empty:
        return {}

    metrics = per_request_df.drop(columns=["request_id"], errors="ignore").mean(numeric_only=True).to_dict()
    hard_requests = per_request_df[per_request_df["is_hard_topk_request"].gt(0)]
    metrics["hard_topk_requests"] = float(len(hard_requests))
    metrics["hard_topk_request_rate"] = float(len(hard_requests) / max(len(per_request_df), 1))

    eligible_mask = per_request_df["is_hard_topk_request"].gt(0)
    has_click = per_request_df["has_click_positive"].gt(0)
    has_collect = per_request_df["has_collect_positive"].gt(0)
    has_share = per_request_df["has_share_positive"].gt(0)
    rare_mask = eligible_mask & (has_collect | has_share)
    oracle = (
        TASK_WEIGHTS["click"] * has_click.astype(float)
        + TASK_WEIGHTS["collect"] * has_collect.astype(float)
        + TASK_WEIGHTS["share"] * has_share.astype(float)
    )
    metrics["eligible_score"] = _safe_mean(per_request_df.loc[eligible_mask, "weighted_hit@20"])
    metrics["rare_requests"] = float(int(rare_mask.sum()))
    if bool(rare_mask.any()):
        rare_hit = (
            TASK_WEIGHTS["collect"] * per_request_df.loc[rare_mask, "hit_collect@20"]
            + TASK_WEIGHTS["share"] * per_request_df.loc[rare_mask, "hit_share@20"]
        ) / (TASK_WEIGHTS["collect"] + TASK_WEIGHTS["share"])
        metrics["rare_score"] = _safe_mean(rare_hit)
    else:
        metrics["rare_score"] = float("nan")
    metrics["oracle_score"] = _safe_mean(oracle)
    metrics["oracle_score_eligible"] = _safe_mean(oracle[eligible_mask])
    metrics["hard_official_capture"] = (
        float(metrics["eligible_score"] / metrics["oracle_score_eligible"])
        if np.isfinite(metrics["eligible_score"]) and np.isfinite(metrics["oracle_score_eligible"]) and metrics["oracle_score_eligible"] > 0
        else float("nan")
    )

    hard_metric_cols = [
        "weighted_hit@20",
        "official_weighted_hit@20",
        "ndcg@20",
        "request_pair_auc_weighted",
        *[f"hit_{task}@20" for task in TASKS],
        *[f"hit_{task}_positive@20" for task in TASKS],
        *[f"recall_{task}@20" for task in TASKS],
        *[f"request_auc_{task}" for task in TASKS],
        *[f"request_ap_{task}" for task in TASKS],
        *[f"topk_boundary_success_{task}" for task in TASKS],
        *[f"topk_boundary_margin_{task}" for task in TASKS],
    ]
    for col in hard_metric_cols:
        if col in hard_requests:
            metrics[f"hard_{col}"] = _safe_mean(hard_requests[col])

    metrics["official_weighted_hit@20"] = metrics.get("weighted_hit@20", float("nan"))
    metrics["preference_auc"] = metrics.get("request_pair_auc_weighted", float("nan"))
    metrics["hard_preference_auc"] = metrics.get("hard_request_pair_auc_weighted", float("nan"))
    for task in TASKS:
        metrics[f"positive_request_rate_{task}"] = float(metrics.get(f"has_{task}_positive", float("nan")))
    metrics["sparse_ap"] = _combine_available(
        [
            (metrics.get("hard_request_ap_collect"), TASK_WEIGHTS["collect"]),
            (metrics.get("hard_request_ap_share"), TASK_WEIGHTS["share"]),
        ]
    )
    metrics["sparse_recall"] = _combine_available(
        [
            (metrics.get("hard_recall_collect@20"), TASK_WEIGHTS["collect"]),
            (metrics.get("hard_recall_share@20"), TASK_WEIGHTS["share"]),
        ]
    )
    metrics["native_selection_score"] = _weighted_metric_score(metrics, NATIVE_SELECTION_WEIGHTS)
    metrics["acceptance_guard_score"] = _combine_available(
        [
            (metrics.get("hard_weighted_hit@20"), 0.5),
            (metrics.get("sparse_ap"), 0.25),
            (metrics.get("sparse_recall"), 0.25),
        ]
    )
    return {k: float(v) for k, v in metrics.items()}


def compute_ranking_metrics(df: pd.DataFrame, topk: int = 20, include_diagnostics: bool = False) -> dict[str, float]:
    """Compute native Qilin ranking metrics."""

    if df.empty:
        return {}
    required = {"request_id", "item_id", "score", *TASKS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}")

    raw_rows = len(df)
    df = deduplicate_request_items(df)
    duplicate_rows_removed = raw_rows - len(df)
    per_request_df = per_request_metric_frame(df, topk=topk)
    metrics = summarize_per_request_metrics(per_request_df)
    metrics["topk"] = float(topk)
    metrics["candidate_count_le_topk_rate"] = float(per_request_df["candidate_count"].le(topk).mean())
    metrics["candidate_count_gt_topk_rate"] = float(per_request_df["candidate_count"].gt(topk).mean())
    metrics["candidate_count_le20_rate"] = metrics["candidate_count_le_topk_rate"]
    metrics["candidate_count_gt20_rate"] = metrics["candidate_count_gt_topk_rate"]

    for task in TASKS:
        score_col = f"p_{task}"
        if score_col in df.columns:
            metrics[f"auc_{task}"] = _safe_auc(df[task].to_numpy(), df[score_col].to_numpy())
            metrics[f"ap_{task}"] = _safe_ap(df[task].to_numpy(), df[score_col].to_numpy())
        metrics[f"positive_requests_{task}"] = float(df.groupby("request_id")[task].sum().gt(0).sum())
        request_auc_col = f"request_auc_{task}"
        request_ap_col = f"request_ap_{task}"
        metrics[f"request_auc_requests_{task}"] = float(per_request_df[request_auc_col].notna().sum())
        metrics[f"request_auc_request_rate_{task}"] = float(per_request_df[request_auc_col].notna().mean())
        metrics[f"request_ap_requests_{task}"] = float(per_request_df[request_ap_col].notna().sum())
        metrics[f"request_ap_request_rate_{task}"] = float(per_request_df[request_ap_col].notna().mean())

    gate_cols = sorted(col for col in df.columns if col.startswith("gate_"))
    for col in gate_cols:
        metrics[f"mean_{col}"] = float(df[col].mean())
    metrics["request_auc_weighted"] = _weighted_available(metrics, "request_auc")
    metrics["request_ap_weighted"] = _weighted_available(metrics, "request_ap")
    metrics["hard_request_auc_weighted"] = _weighted_available(metrics, "hard_request_auc")
    metrics["hard_request_ap_weighted"] = _weighted_available(metrics, "hard_request_ap")

    if {"has_intent_target", "is_type_shift", "is_taxonomy_shift"}.issubset(per_request_df.columns):
        target_requests = per_request_df[per_request_df["has_intent_target"].gt(0)]
        shift_requests = target_requests[target_requests["is_intent_shift"].gt(0)]
        stable_requests = target_requests[target_requests["is_intent_shift"].le(0)]
        type_shift_requests = target_requests[target_requests["is_type_shift"].gt(0)]
        taxonomy_shift_requests = target_requests[target_requests["is_taxonomy_shift"].gt(0)]
        metrics["intent_target_requests"] = float(len(target_requests))
        metrics["intent_target_request_rate"] = float(len(target_requests) / max(len(per_request_df), 1))
        metrics["intent_shift_requests"] = float(len(shift_requests))
        metrics["intent_shift_request_rate"] = float(len(shift_requests) / max(len(target_requests), 1)) if len(target_requests) else float("nan")
        metrics["type_shift_requests"] = float(len(type_shift_requests))
        metrics["taxonomy_shift_requests"] = float(len(taxonomy_shift_requests))
        metrics["shift_type_hit@1"] = _safe_mean(type_shift_requests["intent_type_acc@1"]) if "intent_type_acc@1" in type_shift_requests else float("nan")
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
