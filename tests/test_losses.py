from __future__ import annotations

import torch

from heterointent.training.losses import compute_loss, focal_bce_with_logits, request_listwise_loss


def test_positive_weights_increase_sparse_task_bce() -> None:
    logits = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    labels = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    base = focal_bce_with_logits(logits, labels, gamma=0.0)
    weighted = focal_bce_with_logits(
        logits,
        labels,
        gamma=0.0,
        positive_weights=torch.tensor([1.0, 4.0, 6.0]),
        negative_weights=torch.ones(3),
    )
    assert weighted[0] == base[0]
    assert weighted[1] > base[1]
    assert weighted[2] > weighted[1]


def test_compute_loss_reports_multi_objective_components() -> None:
    logits = torch.zeros((6, 3), requires_grad=True)
    labels = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    outputs = {
        "logits": logits,
        "final_score": torch.sigmoid(logits).matmul(torch.tensor([0.3, 0.4, 0.3])),
        "type_transition_logits": torch.zeros((6, 3), requires_grad=True),
        "taxonomy_transition_logits": torch.zeros((6, 5), requires_grad=True),
        "text_repr": torch.zeros((6, 4)),
        "image_repr": torch.zeros((6, 4)),
    }
    batch = {
        "labels": labels,
        "request_id": torch.tensor([1, 1, 1, 2, 2, 2]),
        "next_item_type": torch.zeros(6, dtype=torch.long),
        "target_item_type": torch.zeros(6, dtype=torch.long),
        "target_taxonomy_id": torch.zeros(6, dtype=torch.long),
    }
    cfg = {
        "task_weights": {"click": 0.3, "collect": 0.4, "share": 0.3},
        "positive_weights": {"click": 1.0, "collect": 4.0, "share": 6.0},
        "negative_weights": {"click": 1.0, "collect": 1.0, "share": 1.0},
        "bpr_weight": 0.03,
        "task_bpr_weight": 0.02,
        "type_transition_weight": 0.0,
        "taxonomy_transition_weight": 0.0,
        "contrastive_weight": 0.0,
    }

    loss, logs = compute_loss(outputs, batch, cfg)

    assert loss.requires_grad
    assert logs["bce_collect_loss"] > logs["bce_click_loss"]
    assert logs["bce_share_loss"] > logs["bce_collect_loss"]
    assert logs["bpr_loss"] > 0
    assert logs["task_bpr_loss"] > 0


def test_bounded_bpr_can_be_disabled_by_group_limit() -> None:
    logits = torch.zeros((4, 3), requires_grad=True)
    labels = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    outputs = {
        "logits": logits,
        "final_score": torch.sigmoid(logits).matmul(torch.tensor([0.3, 0.4, 0.3])),
        "type_transition_logits": torch.zeros((4, 3), requires_grad=True),
        "taxonomy_transition_logits": torch.zeros((4, 5), requires_grad=True),
        "text_repr": torch.zeros((4, 4)),
        "image_repr": torch.zeros((4, 4)),
    }
    batch = {
        "labels": labels,
        "request_id": torch.tensor([1, 1, 2, 2]),
        "next_item_type": torch.zeros(4, dtype=torch.long),
        "target_item_type": torch.zeros(4, dtype=torch.long),
        "target_taxonomy_id": torch.zeros(4, dtype=torch.long),
    }
    cfg = {
        "bpr_weight": 0.03,
        "task_bpr_weight": 0.02,
        "bpr_max_groups": 0,
        "type_transition_weight": 0.0,
        "taxonomy_transition_weight": 0.0,
        "contrastive_weight": 0.0,
    }

    _, logs = compute_loss(outputs, batch, cfg)

    assert logs["bpr_loss"] == 0
    assert logs["task_bpr_loss"] == 0


def test_listwise_loss_handles_edge_cases() -> None:
    weights = torch.tensor([0.3, 0.4, 0.3])
    no_positive = request_listwise_loss(
        torch.tensor([0.1, 0.2]),
        torch.zeros((2, 3)),
        torch.tensor([1, 1]),
        weights,
    )
    single_candidate = request_listwise_loss(
        torch.tensor([0.1]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([1]),
        weights,
    )
    scores = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    normal = request_listwise_loss(
        scores,
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        torch.tensor([1, 1, 1]),
        weights,
    )

    assert no_positive == 0
    assert single_candidate == 0
    assert normal > 0
    normal.backward()
    assert scores.grad is not None


def test_compute_loss_without_intent_heads_and_with_listwise() -> None:
    logits = torch.zeros((4, 3), requires_grad=True)
    labels = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    outputs = {
        "logits": logits,
        "final_score": torch.sigmoid(logits).matmul(torch.tensor([0.3, 0.4, 0.3])),
    }
    batch = {
        "labels": labels,
        "request_id": torch.tensor([1, 1, 1, 1]),
        "next_item_type": torch.zeros(4, dtype=torch.long),
        "target_item_type": torch.zeros(4, dtype=torch.long),
        "target_taxonomy_id": torch.zeros(4, dtype=torch.long),
    }
    cfg = {
        "task_weights": {"click": 0.3, "collect": 0.4, "share": 0.3},
        "positive_weights": {"click": 1.0, "collect": 8.0, "share": 12.0},
        "negative_weights": {"click": 1.0, "collect": 1.0, "share": 1.0},
        "bpr_weight": 0.0,
        "task_bpr_weight": 0.0,
        "listwise_weight": 0.03,
        "collect_listwise_weight": 0.04,
        "share_listwise_weight": 0.05,
        "type_transition_weight": 0.1,
        "taxonomy_transition_weight": 0.1,
        "contrastive_weight": 0.0,
    }

    loss, logs = compute_loss(outputs, batch, cfg)

    assert loss.requires_grad
    assert logs["listwise_loss"] > 0
    assert logs["collect_listwise_loss"] > 0
    assert logs["share_listwise_loss"] > 0
    assert logs["type_transition_loss"] == 0
    assert logs["taxonomy_transition_loss"] == 0
