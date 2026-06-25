from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _masked_dot(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    value = (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1)
    if mask is not None:
        value = value * mask.float()
    return value


class TextFusionGate(nn.Module):
    """Fuse title/content/joint text projections with length and similarity stats."""

    def __init__(self, embed_dim: int, stat_dim: int, dropout: float):
        super().__init__()
        gate_in = embed_dim * 3 + stat_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_in, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 3),
        )

    def forward(
        self,
        title_repr: torch.Tensor,
        content_repr: torch.Tensor,
        joint_repr: torch.Tensor,
        text_stats: torch.Tensor,
        title_present: torch.Tensor,
        content_present: torch.Tensor,
        joint_present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.gate(torch.cat([title_repr, content_repr, joint_repr, text_stats], dim=-1))
        presence = torch.stack([title_present, content_present, joint_present], dim=-1)
        gate_logits = gate_logits.masked_fill(~presence, torch.finfo(gate_logits.dtype).min)
        weights = torch.softmax(gate_logits, dim=-1)
        stacked = torch.stack([title_repr, content_repr, joint_repr], dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        has_any = presence.any(dim=-1, keepdim=True)
        fused = torch.where(has_any, fused, torch.zeros_like(fused))
        return fused, weights


class QueryInteractionModule(nn.Module):
    """Multi-path query semantic interactions for ranking."""

    def __init__(self, embed_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.output = nn.Sequential(
            nn.Linear(5, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query_repr: torch.Tensor,
        title_repr: torch.Tensor,
        content_repr: torch.Tensor,
        joint_repr: torch.Tensor,
        item_text_repr: torch.Tensor,
        history_text_repr: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        interactions = torch.stack(
            [
                _masked_dot(query_repr, title_repr, query_mask),
                _masked_dot(query_repr, content_repr, query_mask),
                _masked_dot(query_repr, joint_repr, query_mask),
                _masked_dot(query_repr, item_text_repr, query_mask),
                _masked_dot(query_repr, history_text_repr, query_mask),
            ],
            dim=-1,
        )
        return self.output(interactions), interactions


class TaskTextResidualHeads(nn.Module):
    """Zero-initialized per-task text residual logits."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(3)])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(features) for head in self.heads], dim=-1)
