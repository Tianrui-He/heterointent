from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TASKS = ("click", "collect", "share")
TASK_WEIGHTS = {"click": 0.3, "collect": 0.4, "share": 0.3}


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _dcg(relevances: list[float]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def deduplicate_request_items(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one scored row per request-item pair.

    Qilin recommendation candidates can contain the same item more than once in a request.
    Ranking should recommend unique items, while labels are aggregated with max so any
    positive exposure makes the unique request-item pair positive for that task.
    """

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


def compute_ranking_metrics(df: pd.DataFrame, topk: int = 20) -> dict[str, float]:
    """Compute request-level Top-K metrics for click/collect/share ranking.

    `recall_<task>@20` is averaged only over requests that have at least one
    positive label for that task. Requests without positives have undefined
    recall and are excluded from this average. The `*_overall@20` variants keep
    the old zero-filled convention for comparison.
    """

    if df.empty:
        return {}
    required = {"request_id", "item_id", "score", *TASKS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}")

    raw_rows = len(df)
    df = deduplicate_request_items(df)
    duplicate_rows_removed = raw_rows - len(df)

    per_request = []
    for _, group in df.groupby("request_id", sort=False):
        ranked = group.sort_values("score", ascending=False).head(topk)
        row: dict[str, float] = {}
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
        row["weighted_hit@20"] = sum(TASK_WEIGHTS[t] * row[f"hit_{t}@20"] for t in TASKS)
        per_request.append(row)

    metrics = pd.DataFrame(per_request).mean(numeric_only=True).to_dict()
    for task in TASKS:
        score_col = f"p_{task}"
        if score_col in df.columns:
            metrics[f"auc_{task}"] = _safe_auc(df[task].to_numpy(), df[score_col].to_numpy())
        metrics[f"positive_requests_{task}"] = float(df.groupby("request_id")[task].sum().gt(0).sum())
        metrics[f"positive_request_rate_{task}"] = float(metrics[f"has_{task}_positive"])
    metrics["num_requests"] = float(df["request_id"].nunique())
    metrics["num_rows"] = float(len(df))
    metrics["num_raw_rows"] = float(raw_rows)
    metrics["num_duplicate_rows_removed"] = float(duplicate_rows_removed)
    return {k: float(v) for k, v in metrics.items()}