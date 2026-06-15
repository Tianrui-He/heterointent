from __future__ import annotations

import math

import torch
from torch import nn


class ItemEncoder(nn.Module):
    def __init__(self, metadata: dict, embed_dim: int, max_position: int, dropout: float, use_graph_embedding: bool):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_graph_embedding = use_graph_embedding
        self.item_embedding = nn.Embedding(int(metadata["num_items"]), embed_dim, padding_idx=0)
        self.type_embedding = nn.Embedding(int(metadata["num_item_types"]), embed_dim, padding_idx=0)
        self.taxonomy_embedding = nn.Embedding(int(metadata["num_taxonomies"]), embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_position + 1, embed_dim, padding_idx=0)
        self.graph_embedding = nn.Embedding(int(metadata["num_items"]), embed_dim, padding_idx=0)
        nn.init.zeros_(self.graph_embedding.weight)

        dims = {
            "text": int(metadata.get("text_dim", 0)),
            "image": int(metadata.get("image_dim", 0)),
            "video": int(metadata.get("video_dim", 0)),
            "dense": int(metadata.get("dense_dim", 0)),
        }
        self.projections = nn.ModuleDict()
        for name, dim in dims.items():
            if dim > 0:
                self.projections[name] = nn.Sequential(nn.Linear(dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        gate_inputs = embed_dim * (4 + len(self.projections) + int(use_graph_embedding))
        self.gate = nn.Linear(gate_inputs, 4 + len(self.projections) + int(use_graph_embedding))
        self.output = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(dropout))
        self.part_names = ["item_id", "item_type", "taxonomy", "position", *self.projections.keys()]
        if use_graph_embedding:
            self.part_names.append("graph")

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
        parts = [
            self.item_embedding(batch["item_id"]),
            self.type_embedding(batch["item_type"]),
            self.taxonomy_embedding(batch["taxonomy_id"]),
            self.position_embedding(position),
        ]
        presence_masks = [torch.ones_like(batch["item_id"], dtype=torch.bool) for _ in parts]
        text_repr = None
        image_repr = None
        for name, projection in self.projections.items():
            feat = batch[f"{name}_feat"].float()
            presence_masks.append(feat.abs().sum(dim=-1).gt(0))
            if feat.is_cuda:
                with torch.amp.autocast("cuda", enabled=False):
                    repr_ = projection(feat)
            else:
                repr_ = projection(feat)
            if name == "text":
                text_repr = repr_
            if name == "image":
                image_repr = repr_
            parts.append(repr_)
        if self.use_graph_embedding:
            parts.append(self.graph_embedding(batch["item_id"]))
            presence_masks.append(batch["item_id"].gt(0))
        gate_logits = self.gate(torch.cat(parts, dim=-1))
        modality_mask = torch.stack(presence_masks, dim=-1)
        gate_logits = gate_logits.masked_fill(~modality_mask, torch.finfo(gate_logits.dtype).min)
        weights = torch.softmax(gate_logits, dim=-1)
        stacked = torch.stack(parts, dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        extras = {
            "modality_gate": weights,
            "modality_gate_mask": modality_mask.float(),
            "item_id_repr": parts[0],
            "text_repr": text_repr if text_repr is not None else torch.zeros_like(fused),
            "image_repr": image_repr if image_repr is not None else torch.zeros_like(fused),
        }
        return self.output(fused), extras


class UserInterestEncoder(nn.Module):
    def __init__(
        self,
        num_users: int,
        item_embedding: nn.Embedding,
        embed_dim: int,
        max_history: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim, padding_idx=0)
        self.item_embedding = item_embedding
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
        hist_emb = self.item_embedding(history) + self.positional[: history.size(1)].unsqueeze(0)
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
        user_base = self.user_embedding(batch["user_id"])
        user_repr = self.output(torch.cat([user_base, pooled, din_interest], dim=-1))
        return user_repr, {"history_attention": attn, "history_intent_repr": history_intent}
