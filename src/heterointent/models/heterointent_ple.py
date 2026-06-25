from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from heterointent.models.encoders import ItemEncoder, UserInterestEncoder
from heterointent.models.feature_modules import QueryInteractionModule, TaskTextResidualHeads
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
        self.enable_aux_heads = bool(model_cfg.get("enable_aux_heads", False))
        self.use_computed_cross = bool(model_cfg.get("use_computed_cross", False))
        self.use_query_interaction = bool(model_cfg.get("use_query_interaction", False))
        self.task_text_residual_weight = float(model_cfg.get("task_text_residual_weight", 0.0))
        self.use_rank_head = bool(model_cfg.get("use_rank_head", False))
        default_rank_blend = 0.5 if self.use_rank_head else 0.0
        self.rank_score_blend = min(max(float(model_cfg.get("rank_score_blend", default_rank_blend)), 0.0), 1.0)
        encoder_cfg = {
            "use_text_fusion_gate": bool(model_cfg.get("use_text_fusion_gate", False)),
            "use_cold_stage_gate": bool(model_cfg.get("use_cold_stage_gate", False)),
            "use_history_semantic": bool(model_cfg.get("use_history_semantic", False)),
        }

        self.item_encoder = ItemEncoder(
            metadata=metadata,
            embed_dim=embed_dim,
            max_position=int(model_cfg.get("max_position", 200)),
            dropout=dropout,
            use_graph_embedding=use_graph,
            disabled_modalities=list(model_cfg.get("disabled_modalities", [])),
            encoder_cfg=encoder_cfg,
        )
        self.item_encoder.set_graph_trainable(bool(model_cfg.get("graph_embedding_trainable", False)))
        self.user_encoder = UserInterestEncoder(
            metadata=metadata,
            item_embedding=self.item_encoder.item_embedding,
            type_embedding=self.item_encoder.type_embedding,
            taxonomy_embedding=self.item_encoder.taxonomy_embedding,
            taxonomy1_embedding=self.item_encoder.taxonomy1_embedding,
            taxonomy2_embedding=self.item_encoder.taxonomy2_embedding,
            embed_dim=embed_dim,
            max_history=int(metadata.get("max_history", 20)),
            num_layers=int(model_cfg.get("transformer_layers", 2)),
            num_heads=int(model_cfg.get("transformer_heads", 4)),
            dropout=dropout,
            encoder_cfg=encoder_cfg,
        )

        self.query_projection = (
            nn.Sequential(nn.Linear(int(metadata.get("query_dim", 0)), embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(), nn.Dropout(dropout))
            if int(metadata.get("query_dim", 0)) > 0
            else None
        )
        self.query_interaction = (
            QueryInteractionModule(embed_dim, embed_dim, dropout)
            if self.use_query_interaction and self.query_projection is not None
            else None
        )
        self.cross_projection = (
            nn.Sequential(nn.Linear(int(metadata.get("cross_dim", 0)), embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(), nn.Dropout(dropout))
            if int(metadata.get("cross_dim", 0)) > 0
            else None
        )
        self.computed_cross_projection = (
            nn.Sequential(nn.Linear(5, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(), nn.Dropout(dropout))
            if self.use_computed_cross
            else None
        )

        feature_dim = embed_dim * 6
        if self.query_projection is not None and self.query_interaction is None:
            feature_dim += embed_dim
        if self.query_interaction is not None:
            feature_dim += embed_dim
        if self.cross_projection is not None:
            feature_dim += embed_dim
        if self.computed_cross_projection is not None:
            feature_dim += embed_dim
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

        residual_dim = embed_dim + int(metadata.get("history_text_dim", 0))
        self.task_text_residual = (
            TaskTextResidualHeads(residual_dim)
            if self.task_text_residual_weight > 0
            else None
        )
        if self.use_rank_head:
            self.rank_score_head = nn.Linear(hidden_dim, 1)
        if self.enable_aux_heads:
            self.aux_like_head = nn.Linear(hidden_dim, 1)
            self.aux_comment_head = nn.Linear(hidden_dim, 1)
            self.aux_page_time_head = nn.Linear(hidden_dim, 1)
        if self.enable_intent_heads:
            self.type_transition_head = mlp(embed_dim, [hidden_dim], int(metadata["num_item_types"]), dropout=dropout)
            self.taxonomy_transition_head = mlp(embed_dim, [hidden_dim], int(metadata["num_taxonomies"]), dropout=dropout)
        self.register_buffer("score_weights", _score_weight_tensor(config), persistent=False)

    @staticmethod
    def _history_match_rate(history_values: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        denom = valid.float().sum(dim=1).clamp_min(1.0)
        matches = history_values.eq(target.unsqueeze(1)) & target.gt(0).unsqueeze(1) & valid
        return matches.float().sum(dim=1) / denom

    def _computed_cross(self, batch: dict[str, torch.Tensor], query_item_dot: torch.Tensor | None) -> torch.Tensor:
        valid = batch["history_items"].gt(0)
        if query_item_dot is None:
            query_item_dot = valid.new_zeros(valid.size(0), dtype=torch.float32)
        zero_history = torch.zeros_like(batch["history_items"])
        zero_item = torch.zeros_like(batch["item_id"])
        values = [
            query_item_dot.float(),
            self._history_match_rate(batch.get("history_item_types", zero_history), batch.get("item_type", zero_item), valid),
            self._history_match_rate(batch.get("history_taxonomy1_ids", zero_history), batch.get("taxonomy1_id", zero_item), valid),
            self._history_match_rate(batch.get("history_taxonomy2_ids", zero_history), batch.get("taxonomy2_id", zero_item), valid),
            self._history_match_rate(batch.get("history_taxonomy_ids", zero_history), batch.get("taxonomy_id", zero_item), valid),
        ]
        return torch.stack(values, dim=-1)

    def _query_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if "has_query" in batch:
            return batch["has_query"].gt(0)
        if "query_feat" in batch:
            return batch["query_feat"].abs().sum(dim=-1).gt(0)
        return torch.zeros_like(batch["item_id"], dtype=torch.bool)

    def _task_text_residual_features(
        self,
        batch: dict[str, torch.Tensor],
        item_extra: dict[str, torch.Tensor],
        user_extra: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        item_text = item_extra.get("item_text_repr", item_extra["text_repr"])
        history_text = batch.get("history_text_feat")
        if history_text is None or history_text.size(-1) == 0:
            return item_text
        return torch.cat([item_text, history_text.float()], dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        item_repr, item_extra = self.item_encoder(batch)
        user_repr, user_extra = self.user_encoder(batch, item_repr)
        match_parts = [
            user_repr,
            item_repr,
            user_repr * item_repr,
            torch.abs(user_repr - item_repr),
            item_extra["item_id_repr"],
            user_repr + item_repr,
        ]
        query_item_dot = None
        query_repr = None
        query_mask = self._query_mask(batch)
        if self.query_projection is not None:
            query_repr = self.query_projection(batch["query_feat"].float())
            text_repr = item_extra.get("item_text_repr", item_extra["text_repr"])
            query_item_dot = (F.normalize(query_repr, dim=-1) * F.normalize(text_repr, dim=-1)).sum(dim=-1)
            query_item_dot = query_item_dot * query_mask.float()
            if self.query_interaction is not None:
                history_text = user_extra.get("history_text_repr")
                if history_text is None:
                    history_text = torch.zeros_like(text_repr)
                elif history_text.size(-1) != text_repr.size(-1):
                    history_text = torch.zeros_like(text_repr)
                query_cross, query_interactions = self.query_interaction(
                    query_repr,
                    item_extra.get("text_title_repr", text_repr),
                    item_extra.get("text_content_repr", text_repr),
                    item_extra.get("text_joint_repr", text_repr),
                    text_repr,
                    history_text,
                    query_mask,
                )
                match_parts.append(query_cross)
                item_extra["query_interactions"] = query_interactions
            else:
                match_parts.append(query_repr * query_mask.unsqueeze(-1).float())
        if self.cross_projection is not None:
            match_parts.append(self.cross_projection(batch["cross_feat"].float()))
        if self.computed_cross_projection is not None:
            match_parts.append(self.computed_cross_projection(self._computed_cross(batch, query_item_dot)))

        match = torch.cat(match_parts, dim=-1)
        h = self.match_projection(match)
        logits, ranker_extra = self.ranker(h)
        if self.task_text_residual is not None:
            residual_features = self._task_text_residual_features(batch, item_extra, user_extra)
            logits = logits + self.task_text_residual_weight * self.task_text_residual(residual_features)
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
        if self.enable_aux_heads:
            result.update(
                {
                    "aux_like_logit": self.aux_like_head(h).squeeze(-1),
                    "aux_comment_logit": self.aux_comment_head(h).squeeze(-1),
                    "aux_page_time": self.aux_page_time_head(h).squeeze(-1),
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


def _masked_dot_tensor(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    value = (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1)
    return value * mask.float()
