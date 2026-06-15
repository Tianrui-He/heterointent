from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_bce_with_logits(logits: torch.Tensor, labels: torch.Tensor, gamma: float = 1.5) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(labels > 0, probs, 1.0 - probs)
    focal = (1.0 - pt).pow(gamma) * bce
    return focal.mean(dim=0)


def request_bpr_loss(scores: torch.Tensor, labels: torch.Tensor, request_ids: torch.Tensor) -> torch.Tensor:
    rel = labels @ torch.tensor([0.3, 0.4, 0.3], dtype=labels.dtype, device=labels.device)
    losses = []
    for request_id in request_ids.unique():
        mask = request_ids.eq(request_id)
        group_scores = scores[mask]
        group_rel = rel[mask]
        pos = group_scores[group_rel > 0]
        neg = group_scores[group_rel <= 0]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        pairwise = neg.unsqueeze(0) - pos.unsqueeze(1)
        losses.append(F.softplus(pairwise).mean())
    if not losses:
        return scores.new_tensor(0.0)
    return torch.stack(losses).mean()


def info_nce_loss(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0 or a.size(0) < 2:
        return a.new_tensor(0.0)
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.T / temperature
    target = torch.arange(a.size(0), device=a.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) * 0.5


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = target.gt(0)
    if not bool(mask.any()):
        return logits.new_tensor(0.0)
    target = target.clamp(min=0, max=logits.size(-1) - 1)
    return F.cross_entropy(logits[mask], target[mask])


def compute_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    labels = batch["labels"]
    logits = outputs["logits"]
    bce = focal_bce_with_logits(logits, labels, gamma=float(cfg.get("focal_gamma", 1.5)))
    task_weights = torch.tensor(
        [float(cfg.get("click_weight", 0.3)), float(cfg.get("collect_weight", 0.4)), float(cfg.get("share_weight", 0.3))],
        dtype=labels.dtype,
        device=labels.device,
    )
    task_loss = (bce * task_weights).sum()

    bpr_weight = float(cfg.get("bpr_weight", 0.1))
    legacy_transition_weight = float(cfg.get("transition_weight", 0.0))
    type_transition_weight = float(cfg.get("type_transition_weight", legacy_transition_weight))
    taxonomy_transition_weight = float(cfg.get("taxonomy_transition_weight", 0.0))
    contrastive_weight = float(cfg.get("contrastive_weight", 0.05))

    bpr = request_bpr_loss(outputs["final_score"], labels, batch["request_id"]) if bpr_weight > 0 else logits.new_tensor(0.0)
    type_transition = (
        masked_cross_entropy(outputs["type_transition_logits"], batch.get("target_item_type", batch["next_item_type"]))
        if type_transition_weight > 0
        else logits.new_tensor(0.0)
    )
    taxonomy_transition = (
        masked_cross_entropy(outputs["taxonomy_transition_logits"], batch["target_taxonomy_id"])
        if taxonomy_transition_weight > 0
        else logits.new_tensor(0.0)
    )
    contrastive = (
        info_nce_loss(outputs["text_repr"], outputs["image_repr"])
        if contrastive_weight > 0
        else logits.new_tensor(0.0)
    )
    total = (
        task_loss
        + bpr_weight * bpr
        + type_transition_weight * type_transition
        + taxonomy_transition_weight * taxonomy_transition
        + contrastive_weight * contrastive
    )
    logs = {
        "loss": total.detach(),
        "task_loss": task_loss.detach(),
        "bpr_loss": bpr.detach(),
        "transition_loss": type_transition.detach(),
        "type_transition_loss": type_transition.detach(),
        "taxonomy_transition_loss": taxonomy_transition.detach(),
        "contrastive_loss": contrastive.detach(),
    }
    return total, logs
