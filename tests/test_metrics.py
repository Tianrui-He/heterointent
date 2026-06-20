from __future__ import annotations

import pandas as pd

from heterointent.evaluation.metrics import compute_ranking_metrics


def test_weighted_hit_at_20() -> None:
    df = pd.DataFrame(
        [
            {"request_id": 1, "item_id": 1, "score": 0.9, "click": 1, "collect": 0, "share": 0, "p_click": 0.9, "p_collect": 0.1, "p_share": 0.1},
            {"request_id": 1, "item_id": 2, "score": 0.1, "click": 0, "collect": 1, "share": 0, "p_click": 0.1, "p_collect": 0.8, "p_share": 0.1},
            {"request_id": 2, "item_id": 3, "score": 0.8, "click": 0, "collect": 0, "share": 1, "p_click": 0.1, "p_collect": 0.1, "p_share": 0.8},
            {"request_id": 2, "item_id": 4, "score": 0.2, "click": 0, "collect": 0, "share": 0, "p_click": 0.1, "p_collect": 0.1, "p_share": 0.1},
        ]
    )
    df["gate_image"] = [0.1, 0.2, 0.3, 0.4]
    metrics = compute_ranking_metrics(df, topk=1)
    assert metrics["hit_click@20"] == 0.5
    assert metrics["hit_collect@20"] == 0.0
    assert metrics["hit_share@20"] == 0.5
    assert metrics["weighted_hit@20"] == 0.3
    assert abs(metrics["mean_gate_image"] - 0.25) < 1e-6
    assert abs(metrics["top20_mean_gate_image"] - 0.2) < 1e-6


def test_request_auc_skips_requests_without_task_positives() -> None:
    df = pd.DataFrame(
        [
            {"request_id": 1, "item_id": 1, "score": 0.9, "click": 0, "collect": 1, "share": 0, "p_click": 0.1, "p_collect": 0.9, "p_share": 0.1},
            {"request_id": 1, "item_id": 2, "score": 0.1, "click": 0, "collect": 0, "share": 0, "p_click": 0.1, "p_collect": 0.2, "p_share": 0.1},
            {"request_id": 2, "item_id": 3, "score": 0.8, "click": 0, "collect": 0, "share": 0, "p_click": 0.1, "p_collect": 0.8, "p_share": 0.1},
            {"request_id": 2, "item_id": 4, "score": 0.2, "click": 0, "collect": 0, "share": 0, "p_click": 0.1, "p_collect": 0.7, "p_share": 0.1},
        ]
    )

    metrics = compute_ranking_metrics(df, topk=1)

    assert metrics["request_auc_collect"] == 1.0
    assert metrics["request_auc_requests_collect"] == 1.0
    assert metrics["request_auc_request_rate_collect"] == 0.5
    assert pd.isna(metrics["request_auc_share"])
    assert metrics["request_auc_requests_share"] == 0.0


def test_hard_topk_metrics_only_use_candidate_count_above_topk() -> None:
    df = pd.DataFrame(
        [
            {"request_id": 1, "item_id": 1, "score": 0.9, "click": 1, "collect": 0, "share": 0, "p_click": 0.9, "p_collect": 0.1, "p_share": 0.1},
            {"request_id": 1, "item_id": 2, "score": 0.1, "click": 0, "collect": 0, "share": 0, "p_click": 0.1, "p_collect": 0.1, "p_share": 0.1},
            {"request_id": 2, "item_id": 3, "score": 0.8, "click": 0, "collect": 0, "share": 0, "p_click": 0.1, "p_collect": 0.1, "p_share": 0.1},
            {"request_id": 2, "item_id": 4, "score": 0.7, "click": 0, "collect": 0, "share": 0, "p_click": 0.1, "p_collect": 0.1, "p_share": 0.1},
            {"request_id": 2, "item_id": 5, "score": 0.6, "click": 1, "collect": 0, "share": 0, "p_click": 0.9, "p_collect": 0.1, "p_share": 0.1},
        ]
    )

    metrics = compute_ranking_metrics(df, topk=2)

    assert metrics["candidate_count_le20_rate"] == 0.5
    assert metrics["candidate_count_gt20_rate"] == 0.5
    assert metrics["hard_topk_requests"] == 1.0
    assert metrics["hard_hit_click@20"] == 0.0
    assert metrics["hard_weighted_hit@20"] == 0.0
