from __future__ import annotations

from collections import Counter

import pandas as pd
import torch

from heterointent.data.dataset import build_dataloader
from heterointent.data.io import write_table
from heterointent.models import HeteroIntentPLE


def _metadata() -> dict:
    return {
        "num_users": 5,
        "num_items": 8,
        "num_item_types": 4,
        "num_taxonomies": 6,
        "text_dim": 0,
        "image_dim": 0,
        "video_dim": 0,
        "dense_dim": 0,
        "max_history": 3,
    }


def _config(score_weights: dict[str, float] | None = None, enable_intent_heads: bool = False) -> dict:
    return {
        "model": {
            "embed_dim": 8,
            "hidden_dim": 16,
            "dropout": 0.0,
            "max_position": 20,
            "transformer_layers": 1,
            "transformer_heads": 2,
            "ranker": "shared_bottom",
            "use_graph_embedding": False,
            "enable_intent_heads": enable_intent_heads,
        },
        "loss": {
            "task_weights": {"click": 0.3, "collect": 0.4, "share": 0.3},
        },
        "evaluation": {
            "score_weights": score_weights or {"click": 0.3, "collect": 0.4, "share": 0.3},
        },
    }


def _batch() -> dict[str, torch.Tensor]:
    return {
        "request_id": torch.tensor([1, 1]),
        "user_id": torch.tensor([1, 2]),
        "item_id": torch.tensor([1, 2]),
        "item_type": torch.tensor([1, 2]),
        "taxonomy_id": torch.tensor([1, 2]),
        "position": torch.tensor([1, 2]),
        "history_items": torch.zeros((2, 3), dtype=torch.long),
        "history_item_types": torch.zeros((2, 3), dtype=torch.long),
        "history_taxonomy_ids": torch.zeros((2, 3), dtype=torch.long),
    }


def test_model_can_disable_intent_heads() -> None:
    model = HeteroIntentPLE(_metadata(), _config(enable_intent_heads=False))
    outputs = model(_batch())

    assert not hasattr(model, "type_transition_head")
    assert "type_transition_logits" not in outputs
    assert "taxonomy_transition_logits" not in outputs


def test_score_weights_control_final_score() -> None:
    model = HeteroIntentPLE(_metadata(), _config(score_weights={"click": 1.0, "collect": 0.0, "share": 0.0}))
    outputs = model(_batch())

    assert torch.allclose(outputs["final_score"], outputs["probs"][:, 0])


def test_rank_head_can_drive_final_score() -> None:
    cfg = _config()
    cfg["model"]["use_rank_head"] = True
    cfg["model"]["rank_score_blend"] = 1.0
    model = HeteroIntentPLE(_metadata(), cfg)
    outputs = model(_batch())

    assert "rank_logit" in outputs
    assert "rank_score" in outputs
    assert torch.allclose(outputs["final_score"], outputs["rank_score"])


def test_request_preserving_loader_does_not_split_request_groups(tmp_path) -> None:
    rows = []
    for request_id, count in [(1, 2), (2, 3), (3, 1), (4, 2)]:
        for pos in range(1, count + 1):
            rows.append(
                {
                    "request_id": request_id,
                    "user_id": request_id,
                    "item_id": request_id * 10 + pos,
                    "item_type": 1,
                    "taxonomy_id": 1,
                    "position": pos,
                    "click": int(pos == 1),
                    "collect": 0,
                    "share": 0,
                }
            )
    path = tmp_path / "samples.csv"
    write_table(pd.DataFrame(rows), path)

    loader = build_dataloader(
        path,
        _metadata(),
        batch_size=4,
        shuffle=False,
        fast_loader=True,
        request_preserving=True,
    )
    full_counts = Counter(row["request_id"] for row in rows)
    seen_batches: dict[int, int] = {}
    seen_counts: Counter[int] = Counter()

    for batch_idx, batch in enumerate(loader):
        request_ids = batch["request_id"].tolist()
        for request_id in set(request_ids):
            assert request_id not in seen_batches
            seen_batches[request_id] = batch_idx
        seen_counts.update(request_ids)

    assert seen_counts == full_counts
