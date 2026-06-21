from __future__ import annotations

import torch
from torch import nn

from heterointent.models.encoders import ItemEncoder, UserInterestEncoder
from heterointent.models.layers import MMoERanker, PLERanker, SharedBottomRanker, mlp

TASK_NAMES = ("click", "collect", "share")
DEFAULT_SCORE_WEIGHTS = (0.3, 0.4, 0.3)


def _score_weight_tensor(config: dict) -> torch.Tensor:
    values = config.get("evaluation", {}).get("score_weights")
    if values is None:
        values = config.get("loss", {}).get("task_weights")
    if isinstance(values, dict):
        weights = [float(values.get(task, default)) for task, default in zip(TASK_NAMES, DEFAULT_SCORE_WEIGHTS)]
    elif isinstance(values, (list, tuple)):
        if len(values) != len(TASK_NAMES):
            raise ValueError(f"score_weights must contain {len(TASK_NAMES)} values")
        weights = [float(value) for value in values]
    else:
        weights = list(DEFAULT_SCORE_WEIGHTS)
    return torch.tensor(weights, dtype=torch.float32)


class HeteroIntentPLE(nn.Module):
    def __init__(self, metadata: dict, config: dict):
        super().__init__()
        model_cfg = config["model"]
        embed_dim = int(model_cfg["embed_dim"])
        hidden_dim = int(model_cfg["hidden_dim"])
        dropout = float(model_cfg.get("dropout", 0.1))
        use_graph = bool(model_cfg.get("use_graph_embedding", False))
        self.enable_intent_heads = bool(model_cfg.get("enable_intent_heads", True))
        self.use_rank_head = bool(model_cfg.get("use_rank_head", False))
        default_rank_blend = 0.5 if self.use_rank_head else 0.0
        self.rank_score_blend = min(max(float(model_cfg.get("rank_score_blend", default_rank_blend)), 0.0), 1.0)

        self.item_encoder = ItemEncoder(
            metadata=metadata,
            embed_dim=embed_dim,
            max_position=int(model_cfg.get("max_position", 200)),
            dropout=dropout,
            use_graph_embedding=use_graph,
            disabled_modalities=list(model_cfg.get("disabled_modalities", [])),
        )
        self.item_encoder.set_graph_trainable(bool(model_cfg.get("graph_embedding_trainable", False)))
        self.user_encoder = UserInterestEncoder(
            num_users=int(metadata["num_users"]),
            item_embedding=self.item_encoder.item_embedding,
            embed_dim=embed_dim,
            max_history=int(metadata.get("max_history", 20)),
            num_layers=int(model_cfg.get("transformer_layers", 2)),
            num_heads=int(model_cfg.get("transformer_heads", 4)),
            dropout=dropout,
        )

        feature_dim = embed_dim * 6
        self.match_projection = mlp(feature_dim, [hidden_dim], hidden_dim, dropout=dropout)
        ranker = str(model_cfg.get("ranker", "ple")).lower()
        if ranker == "shared_bottom":
            self.ranker = SharedBottomRanker(hidden_dim, hidden_dim, dropout=dropout)
        elif ranker == "mmoe":
            self.ranker = MMoERanker(
                hidden_dim,
                hidden_dim,
                num_experts=int(model_cfg.get("shared_experts", 4)),
                dropout=dropout,
            )
        elif ranker == "ple":
            self.ranker = PLERanker(
                hidden_dim,
                hidden_dim,
                shared_experts=int(model_cfg.get("shared_experts", 4)),
                task_experts=int(model_cfg.get("task_experts", 2)),
                ple_layers=int(model_cfg.get("ple_layers", 2)),
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown ranker: {ranker}")

        if self.use_rank_head:
            self.rank_score_head = nn.Linear(hidden_dim, 1)
        if self.enable_intent_heads:
            self.type_transition_head = mlp(embed_dim, [hidden_dim], int(metadata["num_item_types"]), dropout=dropout)
            self.taxonomy_transition_head = mlp(embed_dim, [hidden_dim], int(metadata["num_taxonomies"]), dropout=dropout)
        self.register_buffer("score_weights", _score_weight_tensor(config), persistent=False)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        item_repr, item_extra = self.item_encoder(batch)
        user_repr, user_extra = self.user_encoder(batch, item_repr)
        match = torch.cat(
            [
                user_repr,
                item_repr,
                user_repr * item_repr,
                torch.abs(user_repr - item_repr),
                item_extra["item_id_repr"],
                user_repr + item_repr,
            ],
            dim=-1,
        )
        h = self.match_projection(match)
        logits, ranker_extra = self.ranker(h)
        probs = torch.sigmoid(logits)
        score_weights = self.score_weights.to(probs.device)
        weighted_prob_score = (probs * score_weights).sum(dim=-1)
        final_score = weighted_prob_score
        result = {
            "logits": logits,
            "probs": probs,
            "weighted_prob_score": weighted_prob_score,
            "final_score": final_score,
            **item_extra,
            **user_extra,
            **ranker_extra,
        }
        if self.use_rank_head:
            rank_logit = self.rank_score_head(h).squeeze(-1)
            rank_score = torch.sigmoid(rank_logit)
            final_score = (1.0 - self.rank_score_blend) * weighted_prob_score + self.rank_score_blend * rank_score
            result.update(
                {
                    "rank_logit": rank_logit,
                    "rank_score": rank_score,
                    "final_score": final_score,
                }
            )
        if self.enable_intent_heads:
            history_intent = user_extra["history_intent_repr"]
            type_transition_logits = self.type_transition_head(history_intent)
            taxonomy_transition_logits = self.taxonomy_transition_head(history_intent)
            result.update(
                {
                    "transition_logits": type_transition_logits,
                    "type_transition_logits": type_transition_logits,
                    "taxonomy_transition_logits": taxonomy_transition_logits,
                }
            )
        return result
