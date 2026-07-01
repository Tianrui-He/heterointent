from __future__ import annotations

import torch

from heterointent.models import HeteroIntentPLE
from heterointent.training.losses import compute_loss


def _minimal_metadata() -> dict:
    return {
        "num_users": 4,
        "num_items": 8,
        "num_item_types": 3,
        "num_taxonomies": 4,
        "num_taxonomy1": 3,
        "num_taxonomy2": 3,
        "num_genders": 3,
        "num_platforms": 3,
        "num_ages": 3,
        "num_locations": 3,
        "max_history": 3,
        "text_stat_dim": 3,
        "query_dim": 4,
        "cross_dim": 5,
        "history_text_dim": 4,
        "history_text_last_dim": 4,
        "history_ratio_dim": 2,
        "num_cold_stages": 5,
    }


def _minimal_batch(metadata: dict) -> dict[str, torch.Tensor]:
    batch_size = 2
    return {
        "request_id": torch.tensor([1, 1], dtype=torch.long),
        "user_id": torch.tensor([1, 1], dtype=torch.long),
        "item_id": torch.tensor([1, 2], dtype=torch.long),
        "item_type": torch.tensor([1, 2], dtype=torch.long),
        "taxonomy_id": torch.tensor([1, 2], dtype=torch.long),
        "taxonomy1_id": torch.tensor([1, 2], dtype=torch.long),
        "taxonomy2_id": torch.tensor([1, 2], dtype=torch.long),
        "gender_id": torch.tensor([1, 1], dtype=torch.long),
        "platform_id": torch.tensor([1, 1], dtype=torch.long),
        "age_id": torch.tensor([1, 1], dtype=torch.long),
        "location_id": torch.tensor([1, 1], dtype=torch.long),
        "position": torch.tensor([1, 2], dtype=torch.long),
        "cold_stage_id": torch.tensor([2, 3], dtype=torch.long),
        "has_query": torch.tensor([1, 0], dtype=torch.long),
        "has_image_emb": torch.tensor([1, 0], dtype=torch.long),
        "history_items": torch.tensor([[1, 2, 0], [2, 0, 0]], dtype=torch.long),
        "history_item_types": torch.tensor([[1, 2, 0], [2, 0, 0]], dtype=torch.long),
        "history_taxonomy_ids": torch.tensor([[1, 2, 0], [2, 0, 0]], dtype=torch.long),
        "history_taxonomy1_ids": torch.tensor([[1, 2, 0], [2, 0, 0]], dtype=torch.long),
        "history_taxonomy2_ids": torch.tensor([[1, 2, 0], [2, 0, 0]], dtype=torch.long),
        "labels": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
        "aux_labels": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "page_time_log": torch.tensor([2.0, 0.0], dtype=torch.float32),
        "query_feat": torch.tensor([[1.0, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "cross_feat": torch.ones(batch_size, metadata["cross_dim"], dtype=torch.float32) * 0.1,
        "text_stat_feat": torch.tensor([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]], dtype=torch.float32),
        "history_text_feat": torch.ones(batch_size, metadata["history_text_dim"], dtype=torch.float32) * 0.2,
        "history_text_last_feat": torch.ones(batch_size, metadata["history_text_last_dim"], dtype=torch.float32) * 0.1,
        "history_ratio_feat": torch.ones(batch_size, metadata["history_ratio_dim"], dtype=torch.float32) * 0.05,
        "user_dense_feat": torch.zeros(batch_size, 0, dtype=torch.float32),
    }


def test_feature_opt_v2_forward_backward_smoke() -> None:
    metadata = _minimal_metadata()
    config = {
        "model": {
            "embed_dim": 8,
            "hidden_dim": 16,
            "dropout": 0.0,
            "ranker": "ple",
            "shared_experts": 2,
            "task_experts": 2,
            "ple_layers": 1,
            "use_graph_embedding": False,
            "enable_intent_heads": False,
            "enable_aux_heads": True,
            "use_computed_cross": True,
            "use_query_interaction": True,
            "use_text_fusion_gate": True,
            "use_cold_stage_gate": True,
            "use_history_semantic": True,
            "task_text_residual_weight": 0.1,
        },
        "loss": {
            "task_weights": {"click": 0.3, "collect": 0.4, "share": 0.3},
            "aux_like_weight": 0.02,
            "aux_comment_weight": 0.01,
            "aux_page_time_weight": 0.01,
            "focal_gamma": 1.5,
        },
        "evaluation": {"score_weights": {"click": 0.3, "collect": 0.4, "share": 0.3}},
    }
    model = HeteroIntentPLE(metadata, config)
    batch = _minimal_batch(metadata)
    outputs = model(batch)
    loss, _ = compute_loss(outputs, batch, config["loss"])
    loss.backward()
    assert outputs["logits"].shape == (2, 3)
    assert "query_interactions" in outputs
