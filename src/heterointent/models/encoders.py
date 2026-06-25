from __future__ import annotations

import math

import torch
from torch import nn

from heterointent.models.feature_modules import TextFusionGate


ITEM_FEATURE_GROUPS = (
    "text",
    "text_title",
    "text_content",
    "text_stat",
    "image",
    "video",
    "dense",
    "image_meta",
    "video_meta",
    "image_emb",
    "video_emb",
    "item_dense",
    "ratio",
)


def _projection(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim), nn.ReLU(), nn.Dropout(dropout))


def _clamp_ids(values: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
    return values.clamp(min=0, max=embedding.num_embeddings - 1)


def _zeros_like_batch(batch: dict[str, torch.Tensor], key: str, reference: torch.Tensor) -> torch.Tensor:
    return batch.get(key, torch.zeros_like(reference))


class ItemEncoder(nn.Module):
    def __init__(
        self,
        metadata: dict,
        embed_dim: int,
        max_position: int,
        dropout: float,
        use_graph_embedding: bool,
        disabled_modalities: list[str] | None = None,
        encoder_cfg: dict | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_graph_embedding = use_graph_embedding
        self.disabled_modalities = set(disabled_modalities or [])
        encoder_cfg = encoder_cfg or {}
        self.use_text_fusion_gate = bool(encoder_cfg.get("use_text_fusion_gate", False))
        self.use_cold_stage_gate = bool(encoder_cfg.get("use_cold_stage_gate", False))
        supported = {
            "text",
            "text_title",
            "text_content",
            "text_stat",
            "image",
            "video",
            "dense",
            "image_meta",
            "video_meta",
            "image_emb",
            "video_emb",
            "item_dense",
            "ratio",
        }
        unsupported = self.disabled_modalities - supported
        if unsupported:
            raise ValueError(f"Unsupported disabled modalities: {sorted(unsupported)}")
        self.item_embedding = nn.Embedding(int(metadata["num_items"]), embed_dim, padding_idx=0)
        self.type_embedding = nn.Embedding(int(metadata["num_item_types"]), embed_dim, padding_idx=0)
        self.taxonomy_embedding = nn.Embedding(int(metadata["num_taxonomies"]), embed_dim, padding_idx=0)
        self.taxonomy1_embedding = (
            nn.Embedding(int(metadata.get("num_taxonomy1", 1)), embed_dim, padding_idx=0)
            if int(metadata.get("num_taxonomy1", 1)) > 1
            else None
        )
        self.taxonomy2_embedding = (
            nn.Embedding(int(metadata.get("num_taxonomy2", 1)), embed_dim, padding_idx=0)
            if int(metadata.get("num_taxonomy2", 1)) > 1
            else None
        )
        self.position_embedding = nn.Embedding(max_position + 1, embed_dim, padding_idx=0)
        self.graph_embedding = nn.Embedding(int(metadata["num_items"]), embed_dim, padding_idx=0)
        nn.init.zeros_(self.graph_embedding.weight)

        self.projections = nn.ModuleDict()
        for name in ITEM_FEATURE_GROUPS:
            dim = int(metadata.get(f"{name}_dim", 0))
            if dim > 0 and not self._is_disabled(name):
                self.projections[name] = _projection(dim, embed_dim, dropout)

        self.text_fusion_gate = (
            TextFusionGate(embed_dim, int(metadata.get("text_stat_dim", 0)), dropout)
            if self.use_text_fusion_gate
            and any(name in self.projections for name in ("text", "text_title", "text_content"))
            else None
        )
        self.cold_stage_embedding = (
            nn.Embedding(int(metadata.get("num_cold_stages", 1)), embed_dim, padding_idx=0)
            if self.use_cold_stage_gate and int(metadata.get("num_cold_stages", 1)) > 1
            else None
        )

        gate_text_parts = ["text_fused"] if self.use_text_fusion_gate else ["text", "text_title", "text_content"]
        self.part_names = ["item_id", "item_type", "taxonomy", "position"]
        if self.taxonomy1_embedding is not None:
            self.part_names.append("taxonomy1")
        if self.taxonomy2_embedding is not None:
            self.part_names.append("taxonomy2")
        for name in self.projections.keys():
            if name in {"text", "text_title", "text_content"}:
                continue
            self.part_names.append(name)
        if self.use_text_fusion_gate and any(k in self.projections for k in ("text", "text_title", "text_content")):
            self.part_names.extend([p for p in gate_text_parts if p not in self.part_names])
        elif not self.use_text_fusion_gate:
            for name in ("text", "text_title", "text_content"):
                if name in self.projections and name not in self.part_names:
                    self.part_names.append(name)
        if self.cold_stage_embedding is not None:
            self.part_names.append("cold_stage")
        if use_graph_embedding:
            self.part_names.append("graph")
        self.gate = nn.Linear(embed_dim * len(self.part_names), len(self.part_names))
        self.output = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(dropout))

    def _is_disabled(self, name: str) -> bool:
        if name in self.disabled_modalities:
            return True
        if name in {"text_title", "text_content"} and "text" in self.disabled_modalities:
            return True
        if name in {"image_meta", "image_emb"} and "image" in self.disabled_modalities:
            return True
        if name in {"video_meta", "video_emb"} and "video" in self.disabled_modalities:
            return True
        if name in {"item_dense", "ratio"} and "dense" in self.disabled_modalities:
            return True
        return False

    def set_graph_trainable(self, trainable: bool) -> None:
        self.graph_embedding.weight.requires_grad_(trainable)

    def load_graph_embedding(self, values: torch.Tensor) -> None:
        if values.shape != self.graph_embedding.weight.shape:
            raise ValueError(
                f"Graph embedding shape mismatch: expected {tuple(self.graph_embedding.weight.shape)}, "
                f"got {tuple(values.shape)}"
            )
        with torch.no_grad():
            self.graph_embedding.weight.copy_(values)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        position = batch["position"].clamp(min=0, max=self.position_embedding.num_embeddings - 1)
        parts_by_name = {
            "item_id": self.item_embedding(_clamp_ids(batch["item_id"], self.item_embedding)),
            "item_type": self.type_embedding(_clamp_ids(batch["item_type"], self.type_embedding)),
            "taxonomy": self.taxonomy_embedding(_clamp_ids(batch["taxonomy_id"], self.taxonomy_embedding)),
            "position": self.position_embedding(position),
        }
        presence_by_name = {
            "item_id": torch.ones_like(batch["item_id"], dtype=torch.bool),
            "item_type": torch.ones_like(batch["item_id"], dtype=torch.bool),
            "taxonomy": torch.ones_like(batch["item_id"], dtype=torch.bool),
            "position": torch.ones_like(batch["item_id"], dtype=torch.bool),
        }
        if self.taxonomy1_embedding is not None:
            parts_by_name["taxonomy1"] = self.taxonomy1_embedding(_clamp_ids(batch["taxonomy1_id"], self.taxonomy1_embedding))
            presence_by_name["taxonomy1"] = batch["taxonomy1_id"].gt(0)
        if self.taxonomy2_embedding is not None:
            parts_by_name["taxonomy2"] = self.taxonomy2_embedding(_clamp_ids(batch["taxonomy2_id"], self.taxonomy2_embedding))
            presence_by_name["taxonomy2"] = batch["taxonomy2_id"].gt(0)

        projected: dict[str, torch.Tensor] = {}
        for name, projection in self.projections.items():
            feat = batch[f"{name}_feat"].float()
            presence = feat.abs().sum(dim=-1).gt(0)
            if name == "image_emb" and "has_image_emb" in batch:
                presence = presence & batch["has_image_emb"].gt(0)
            if name == "video_emb" and "has_video_emb" in batch:
                presence = presence & batch["has_video_emb"].gt(0)
            presence_by_name[name] = presence
            repr_ = projection(feat)
            parts_by_name[name] = repr_
            projected[name] = repr_

        title_repr = projected.get("text_title", projected.get("text", torch.zeros_like(parts_by_name["item_id"])))
        content_repr = projected.get("text_content", projected.get("text", torch.zeros_like(parts_by_name["item_id"])))
        joint_repr: torch.Tensor
        if "text" in projected:
            joint_repr = projected["text"]
        elif "text_title" in projected or "text_content" in projected:
            values = [projected[k] for k in ("text_title", "text_content") if k in projected]
            joint_repr = torch.stack(values, dim=0).mean(dim=0) if values else torch.zeros_like(parts_by_name["item_id"])
        else:
            joint_repr = torch.zeros_like(parts_by_name["item_id"])

        text_gate = None
        if self.text_fusion_gate is not None:
            text_stats = batch.get("text_stat_feat")
            if text_stats is None:
                text_stats = torch.zeros(batch["item_id"].size(0), int(self.text_fusion_gate.gate[0].in_features - self.embed_dim * 3), device=batch["item_id"].device)
            else:
                text_stats = text_stats.float()
            fused_text, text_gate = self.text_fusion_gate(
                title_repr,
                content_repr,
                joint_repr,
                text_stats,
                projected.get("text_title", torch.zeros_like(title_repr)).abs().sum(dim=-1).gt(0) if "text_title" in projected else torch.zeros_like(batch["item_id"], dtype=torch.bool),
                projected.get("text_content", torch.zeros_like(content_repr)).abs().sum(dim=-1).gt(0) if "text_content" in projected else torch.zeros_like(batch["item_id"], dtype=torch.bool),
                (projected.get("text", joint_repr)).abs().sum(dim=-1).gt(0),
            )
            parts_by_name["text_fused"] = fused_text
            presence_by_name["text_fused"] = text_gate.sum(dim=-1).gt(0)

        if self.cold_stage_embedding is not None:
            cold_stage = batch.get("cold_stage_id", torch.zeros_like(batch["item_id"]))
            parts_by_name["cold_stage"] = self.cold_stage_embedding(_clamp_ids(cold_stage, self.cold_stage_embedding))
            presence_by_name["cold_stage"] = cold_stage.gt(0)

        if self.use_graph_embedding:
            parts_by_name["graph"] = self.graph_embedding(_clamp_ids(batch["item_id"], self.graph_embedding))
            presence_by_name["graph"] = batch["item_id"].gt(0)

        parts = [parts_by_name[name] for name in self.part_names]
        presence_masks = [presence_by_name[name] for name in self.part_names]
        gate_logits = self.gate(torch.cat(parts, dim=-1))
        modality_mask = torch.stack(presence_masks, dim=-1)
        gate_logits = gate_logits.masked_fill(~modality_mask, torch.finfo(gate_logits.dtype).min)
        weights = torch.softmax(gate_logits, dim=-1)
        stacked = torch.stack(parts, dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)

        def mean_projected(names: tuple[str, ...]) -> torch.Tensor:
            values = [projected[name] for name in names if name in projected]
            if not values:
                return torch.zeros_like(fused)
            return torch.stack(values, dim=0).mean(dim=0)

        item_text_repr = parts_by_name["text_fused"] if self.text_fusion_gate is not None else mean_projected(("text", "text_title", "text_content"))
        extras = {
            "modality_gate": weights,
            "modality_gate_mask": modality_mask.float(),
            "item_id_repr": parts_by_name["item_id"],
            "text_repr": item_text_repr,
            "text_title_repr": title_repr,
            "text_content_repr": content_repr,
            "text_joint_repr": joint_repr,
            "image_repr": mean_projected(("image", "image_meta", "image_emb")),
            "item_text_repr": item_text_repr,
        }
        if text_gate is not None:
            extras["text_gate"] = text_gate
        return self.output(fused), extras


def _metadata_cardinality(metadata: dict, key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    count = int(value)
    return count if count > 1 else None


def _optional_user_embedding(metadata: dict, key: str, embed_dim: int) -> nn.Embedding | None:
    count = _metadata_cardinality(metadata, key)
    if count is None:
        return None
    return nn.Embedding(count, embed_dim, padding_idx=0)


class UserInterestEncoder(nn.Module):
    def __init__(
        self,
        metadata: dict,
        item_embedding: nn.Embedding,
        type_embedding: nn.Embedding,
        taxonomy_embedding: nn.Embedding,
        taxonomy1_embedding: nn.Embedding | None,
        taxonomy2_embedding: nn.Embedding | None,
        embed_dim: int,
        max_history: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        encoder_cfg: dict | None = None,
    ):
        super().__init__()
        encoder_cfg = encoder_cfg or {}
        self.use_history_semantic = bool(encoder_cfg.get("use_history_semantic", False))
        self.user_embedding = nn.Embedding(int(metadata["num_users"]), embed_dim, padding_idx=0)
        self.gender_embedding = _optional_user_embedding(metadata, "num_genders", embed_dim)
        self.platform_embedding = _optional_user_embedding(metadata, "num_platforms", embed_dim)
        self.age_embedding = _optional_user_embedding(metadata, "num_ages", embed_dim)
        self.location_embedding = _optional_user_embedding(metadata, "num_locations", embed_dim)
        self.item_embedding = item_embedding
        self.type_embedding = type_embedding
        self.taxonomy_embedding = taxonomy_embedding
        self.taxonomy1_embedding = taxonomy1_embedding
        self.taxonomy2_embedding = taxonomy2_embedding
        user_dense_dim = int(metadata.get("user_dense_dim", 0))
        self.user_dense_projection = _projection(user_dense_dim, embed_dim, dropout) if user_dense_dim > 0 else None
        history_text_dim = int(metadata.get("history_text_dim", 0))
        history_ratio_dim = int(metadata.get("history_ratio_dim", 0))
        history_text_last_dim = int(metadata.get("history_text_last_dim", 0))
        self.history_text_projection = (
            _projection(history_text_dim + history_text_last_dim, embed_dim, dropout)
            if self.use_history_semantic and (history_text_dim + history_text_last_dim) > 0
            else None
        )
        self.history_ratio_projection = (
            _projection(history_ratio_dim, embed_dim, dropout)
            if self.use_history_semantic and history_ratio_dim > 0
            else None
        )
        has_user_context = any(
            module is not None
            for module in (
                self.gender_embedding,
                self.platform_embedding,
                self.age_embedding,
                self.location_embedding,
                self.user_dense_projection,
                self.history_text_projection,
                self.history_ratio_projection,
            )
        )
        self.user_context_norm = nn.LayerNorm(embed_dim) if has_user_context else None
        self.positional = nn.Parameter(torch.randn(max_history, embed_dim) / math.sqrt(embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.attention = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self.history_output = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(dropout))
        self.output = nn.Sequential(nn.Linear(embed_dim * 3, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())

    def forward(self, batch: dict[str, torch.Tensor], candidate_repr: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history = batch["history_items"]
        mask = history.eq(0)
        all_empty = mask.all(dim=1)
        transformer_mask = mask.clone()
        if all_empty.any():
            transformer_mask[all_empty, 0] = False
        hist_emb = self.item_embedding(_clamp_ids(history, self.item_embedding))
        hist_emb = hist_emb + self.type_embedding(_clamp_ids(batch["history_item_types"], self.type_embedding))
        hist_emb = hist_emb + self.taxonomy_embedding(_clamp_ids(batch["history_taxonomy_ids"], self.taxonomy_embedding))
        if self.taxonomy1_embedding is not None:
            hist_taxonomy1 = _zeros_like_batch(batch, "history_taxonomy1_ids", history)
            hist_emb = hist_emb + self.taxonomy1_embedding(_clamp_ids(hist_taxonomy1, self.taxonomy1_embedding))
        if self.taxonomy2_embedding is not None:
            hist_taxonomy2 = _zeros_like_batch(batch, "history_taxonomy2_ids", history)
            hist_emb = hist_emb + self.taxonomy2_embedding(_clamp_ids(hist_taxonomy2, self.taxonomy2_embedding))
        hist_emb = hist_emb + self.positional[: history.size(1)].unsqueeze(0)
        encoded = self.transformer(hist_emb, src_key_padding_mask=transformer_mask)
        encoded = torch.nan_to_num(encoded)
        valid = (~mask).float().unsqueeze(-1)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        history_intent = self.history_output(pooled)

        cand = candidate_repr.unsqueeze(1).expand_as(encoded)
        attn_input = torch.cat([encoded, cand, encoded * cand, torch.abs(encoded - cand)], dim=-1)
        attn_logits = self.attention(attn_input).squeeze(-1)
        mask_value = torch.finfo(attn_logits.dtype).min
        attn_logits = attn_logits.masked_fill(mask, mask_value)
        attn = torch.softmax(attn_logits, dim=-1)
        attn = torch.where(all_empty.unsqueeze(1), torch.zeros_like(attn), attn)
        din_interest = (encoded * attn.unsqueeze(-1)).sum(dim=1)
        user_base = self.user_embedding(_clamp_ids(batch["user_id"], self.user_embedding))
        zero_user_feature = torch.zeros_like(batch["user_id"])
        if self.gender_embedding is not None:
            user_base = user_base + self.gender_embedding(_clamp_ids(batch.get("gender_id", zero_user_feature), self.gender_embedding))
        if self.platform_embedding is not None:
            user_base = user_base + self.platform_embedding(_clamp_ids(batch.get("platform_id", zero_user_feature), self.platform_embedding))
        if self.age_embedding is not None:
            user_base = user_base + self.age_embedding(_clamp_ids(batch.get("age_id", zero_user_feature), self.age_embedding))
        if self.location_embedding is not None:
            user_base = user_base + self.location_embedding(_clamp_ids(batch.get("location_id", zero_user_feature), self.location_embedding))
        if self.user_dense_projection is not None:
            user_dense = batch["user_dense_feat"].float()
            user_base = user_base + self.user_dense_projection(user_dense)
        if self.history_text_projection is not None:
            history_text_parts = [batch["history_text_feat"].float()]
            if "history_text_last_feat" in batch and batch["history_text_last_feat"].size(-1) > 0:
                history_text_parts.append(batch["history_text_last_feat"].float())
            history_text = torch.cat(history_text_parts, dim=-1)
            user_base = user_base + self.history_text_projection(history_text)
        if self.history_ratio_projection is not None:
            history_ratio = batch["history_ratio_feat"].float()
            user_base = user_base + self.history_ratio_projection(history_ratio)
        if self.user_context_norm is not None:
            user_base = self.user_context_norm(user_base)
        user_repr = self.output(torch.cat([user_base, pooled, din_interest], dim=-1))
        return user_repr, {
            "history_attention": attn,
            "history_intent_repr": history_intent,
            "history_text_repr": batch.get("history_text_feat"),
        }
