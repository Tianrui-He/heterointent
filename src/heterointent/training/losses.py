from __future__ import annotations

import torch
import torch.nn.functional as F

TASK_NAMES = ("click", "collect", "share")
DEFAULT_TASK_WEIGHTS = (0.3, 0.4, 0.3)
DEFAULT_BPR_MAX_GROUPS = 128
DEFAULT_BPR_MAX_PAIRS_PER_GROUP = 64
DEFAULT_LISTWISE_MAX_GROUPS = 128


def _task_tensor(
    cfg: dict,
    nested_key: str,
    legacy_prefix: str,
    defaults: tuple[float, float, float],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    values = cfg.get(nested_key)
    if isinstance(values, dict):
        return torch.tensor([float(values.get(task, default)) for task, default in zip(TASK_NAMES, defaults)], dtype=dtype, device=device)
    if isinstance(values, (list, tuple)):
        if len(values) != len(TASK_NAMES):
            raise ValueError(f"{nested_key} must contain {len(TASK_NAMES)} values")
        return torch.tensor([float(v) for v in values], dtype=dtype, device=device)
    key_for = (lambda task: f"{task}_weight") if not legacy_prefix else (lambda task: f"{legacy_prefix}_{task}_weight")
    return torch.tensor(
        [float(cfg.get(key_for(task), default)) for task, default in zip(TASK_NAMES, defaults)],
        dtype=dtype,
        device=device,
    )


def focal_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 1.5,
    positive_weights: torch.Tensor | None = None,
    negative_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(labels > 0, probs, 1.0 - probs)
    focal = (1.0 - pt).pow(gamma) * bce
    if positive_weights is not None or negative_weights is not None:
        if positive_weights is None:
            positive_weights = torch.ones(logits.size(-1), dtype=logits.dtype, device=logits.device)
        if negative_weights is None:
            negative_weights = torch.ones(logits.size(-1), dtype=logits.dtype, device=logits.device)
        example_weights = torch.where(labels > 0, positive_weights.view(1, -1), negative_weights.view(1, -1))
        focal = focal * example_weights
    return focal.mean(dim=0)


def _group_bpr_loss(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    request_ids: torch.Tensor,
    max_groups: int = DEFAULT_BPR_MAX_GROUPS,
    max_pairs_per_group: int = DEFAULT_BPR_MAX_PAIRS_PER_GROUP,
) -> torch.Tensor:
    """Bounded request-level BPR.

    Sorting once avoids repeatedly scanning the whole batch for every request.
    Pair sampling keeps the auxiliary ranking loss from dominating epoch time on
    requests with many positive/negative candidates.
    """

    if scores.numel() == 0:
        return scores.new_tensor(0.0)
    max_groups = max(int(max_groups), 0)
    max_pairs_per_group = max(int(max_pairs_per_group), 0)
    if max_groups == 0 or max_pairs_per_group == 0:
        return scores.new_tensor(0.0)

    order = torch.argsort(request_ids)
    sorted_request_ids = request_ids[order]
    sorted_scores = scores[order]
    sorted_relevance = relevance[order]
    _, counts = torch.unique_consecutive(sorted_request_ids, return_counts=True)
    starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    starts_list = starts.detach().cpu().tolist()
    counts_list = counts.detach().cpu().tolist()

    losses = []
    used_groups = 0
    for start, count in zip(starts_list, counts_list):
        if count <= 1:
            continue
        end = start + count
        group_scores = sorted_scores[start:end]
        group_rel = sorted_relevance[start:end]
        pos = group_scores[group_rel > 0]
        neg = group_scores[group_rel <= 0]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        pair_count = pos.numel() * neg.numel()
        if pair_count > max_pairs_per_group:
            pos_idx = torch.randint(pos.numel(), (max_pairs_per_group,), device=scores.device)
            neg_idx = torch.randint(neg.numel(), (max_pairs_per_group,), device=scores.device)
            pairwise = neg[neg_idx] - pos[pos_idx]
        else:
            pairwise = neg.unsqueeze(0) - pos.unsqueeze(1)
        losses.append(F.softplus(pairwise).mean())
        used_groups += 1
        if used_groups >= max_groups:
            break
    if not losses:
        return scores.new_tensor(0.0)
    return torch.stack(losses).mean()


def request_bpr_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    request_ids: torch.Tensor,
    task_weights: torch.Tensor | None = None,
    max_groups: int = DEFAULT_BPR_MAX_GROUPS,
    max_pairs_per_group: int = DEFAULT_BPR_MAX_PAIRS_PER_GROUP,
) -> torch.Tensor:
    if task_weights is None:
        task_weights = torch.tensor(DEFAULT_TASK_WEIGHTS, dtype=labels.dtype, device=labels.device)
    rel = labels @ task_weights
    return _group_bpr_loss(scores, rel, request_ids, max_groups=max_groups, max_pairs_per_group=max_pairs_per_group)


def request_task_bpr_loss(
    task_scores: torch.Tensor,
    labels: torch.Tensor,
    request_ids: torch.Tensor,
    task_weights: torch.Tensor,
    max_groups: int = DEFAULT_BPR_MAX_GROUPS,
    max_pairs_per_group: int = DEFAULT_BPR_MAX_PAIRS_PER_GROUP,
) -> torch.Tensor:
    losses = []
    for task_idx in range(labels.size(-1)):
        losses.append(
            _group_bpr_loss(
                task_scores[:, task_idx],
                labels[:, task_idx],
                request_ids,
                max_groups=max_groups,
                max_pairs_per_group=max_pairs_per_group,
            )
        )
    if not losses:
        return task_scores.new_tensor(0.0)
    return (torch.stack(losses) * task_weights[: len(losses)]).sum() / task_weights[: len(losses)].sum().clamp_min(1e-12)


def _group_listwise_loss(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    request_ids: torch.Tensor,
    max_groups: int = DEFAULT_LISTWISE_MAX_GROUPS,
    temperature: float = 1.0,
) -> torch.Tensor:
    if scores.numel() == 0:
        return scores.new_tensor(0.0)
    max_groups = max(int(max_groups), 0)
    if max_groups == 0:
        return scores.new_tensor(0.0)
    temperature = max(float(temperature), 1e-6)

    order = torch.argsort(request_ids)
    sorted_request_ids = request_ids[order]
    sorted_scores = scores[order]
    sorted_relevance = relevance[order].clamp_min(0)
    _, counts = torch.unique_consecutive(sorted_request_ids, return_counts=True)
    starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    starts_list = starts.detach().cpu().tolist()
    counts_list = counts.detach().cpu().tolist()

    losses = []
    used_groups = 0
    for start, count in zip(starts_list, counts_list):
        if count <= 1:
            continue
        end = start + count
        group_scores = sorted_scores[start:end] / temperature
        group_rel = sorted_relevance[start:end]
        rel_sum = group_rel.sum()
        if rel_sum <= 0:
            continue
        target = group_rel / rel_sum.clamp_min(1e-12)
        losses.append(-(target * F.log_softmax(group_scores, dim=0)).sum())
        used_groups += 1
        if used_groups >= max_groups:
            break
    if not losses:
        return scores.new_tensor(0.0)
    return torch.stack(losses).mean()


def request_listwise_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    request_ids: torch.Tensor,
    task_weights: torch.Tensor | None = None,
    max_groups: int = DEFAULT_LISTWISE_MAX_GROUPS,
    temperature: float = 1.0,
) -> torch.Tensor:
    if task_weights is None:
        task_weights = torch.tensor(DEFAULT_TASK_WEIGHTS, dtype=labels.dtype, device=labels.device)
    rel = labels @ task_weights
    return _group_listwise_loss(scores, rel, request_ids, max_groups=max_groups, temperature=temperature)


def request_single_task_listwise_loss(
    task_scores: torch.Tensor,
    task_labels: torch.Tensor,
    request_ids: torch.Tensor,
    max_groups: int = DEFAULT_LISTWISE_MAX_GROUPS,
    temperature: float = 1.0,
) -> torch.Tensor:
    return _group_listwise_loss(task_scores, task_labels, request_ids, max_groups=max_groups, temperature=temperature)


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
    task_weights = _task_tensor(cfg, "task_weights", "", DEFAULT_TASK_WEIGHTS, labels.dtype, labels.device)
    positive_weights = _task_tensor(cfg, "positive_weights", "positive", (1.0, 1.0, 1.0), labels.dtype, labels.device)
    negative_weights = _task_tensor(cfg, "negative_weights", "negative", (1.0, 1.0, 1.0), labels.dtype, labels.device)
    bce = focal_bce_with_logits(
        logits,
        labels,
        gamma=float(cfg.get("focal_gamma", 1.5)),
        positive_weights=positive_weights,
        negative_weights=negative_weights,
    )
    task_loss = (bce * task_weights).sum()

    bpr_weight = float(cfg.get("bpr_weight", 0.1))
    task_bpr_weight = float(cfg.get("task_bpr_weight", 0.0))
    bpr_max_groups = int(cfg.get("bpr_max_groups", DEFAULT_BPR_MAX_GROUPS))
    bpr_max_pairs_per_group = int(cfg.get("bpr_max_pairs_per_group", DEFAULT_BPR_MAX_PAIRS_PER_GROUP))
    listwise_weight = float(cfg.get("listwise_weight", 0.0))
    collect_listwise_weight = float(cfg.get("collect_listwise_weight", 0.0))
    share_listwise_weight = float(cfg.get("share_listwise_weight", 0.0))
    listwise_max_groups = int(cfg.get("listwise_max_groups", DEFAULT_LISTWISE_MAX_GROUPS))
    listwise_temperature = float(cfg.get("listwise_temperature", 1.0))
    legacy_transition_weight = float(cfg.get("transition_weight", 0.0))
    type_transition_weight = float(cfg.get("type_transition_weight", legacy_transition_weight))
    taxonomy_transition_weight = float(cfg.get("taxonomy_transition_weight", 0.0))
    contrastive_weight = float(cfg.get("contrastive_weight", 0.05))
    aux_like_weight = float(cfg.get("aux_like_weight", 0.0))
    aux_comment_weight = float(cfg.get("aux_comment_weight", 0.0))
    aux_page_time_weight = float(cfg.get("aux_page_time_weight", 0.0))

    bpr = (
        request_bpr_loss(
            outputs["final_score"],
            labels,
            batch["request_id"],
            task_weights,
            max_groups=bpr_max_groups,
            max_pairs_per_group=bpr_max_pairs_per_group,
        )
        if bpr_weight > 0
        else logits.new_tensor(0.0)
    )
    task_bpr = (
        request_task_bpr_loss(
            logits,
            labels,
            batch["request_id"],
            task_weights,
            max_groups=bpr_max_groups,
            max_pairs_per_group=bpr_max_pairs_per_group,
        )
        if task_bpr_weight > 0
        else logits.new_tensor(0.0)
    )
    listwise = (
        request_listwise_loss(
            outputs["final_score"],
            labels,
            batch["request_id"],
            task_weights,
            max_groups=listwise_max_groups,
            temperature=listwise_temperature,
        )
        if listwise_weight > 0
        else logits.new_tensor(0.0)
    )
    collect_listwise = (
        request_single_task_listwise_loss(
            logits[:, 1],
            labels[:, 1],
            batch["request_id"],
            max_groups=listwise_max_groups,
            temperature=listwise_temperature,
        )
        if collect_listwise_weight > 0
        else logits.new_tensor(0.0)
    )
    share_listwise = (
        request_single_task_listwise_loss(
            logits[:, 2],
            labels[:, 2],
            batch["request_id"],
            max_groups=listwise_max_groups,
            temperature=listwise_temperature,
        )
        if share_listwise_weight > 0
        else logits.new_tensor(0.0)
    )
    type_transition = (
        masked_cross_entropy(outputs["type_transition_logits"], batch.get("target_item_type", batch["next_item_type"]))
        if type_transition_weight > 0 and "type_transition_logits" in outputs
        else logits.new_tensor(0.0)
    )
    taxonomy_transition = (
        masked_cross_entropy(outputs["taxonomy_transition_logits"], batch["target_taxonomy_id"])
        if taxonomy_transition_weight > 0 and "taxonomy_transition_logits" in outputs
        else logits.new_tensor(0.0)
    )
    contrastive = (
        info_nce_loss(outputs["text_repr"], outputs["image_repr"])
        if contrastive_weight > 0 and "text_repr" in outputs and "image_repr" in outputs
        else logits.new_tensor(0.0)
    )
    aux_like = (
        focal_bce_with_logits(
            outputs["aux_like_logit"].unsqueeze(-1),
            batch["aux_labels"][:, :1],
            gamma=float(cfg.get("focal_gamma", 1.5)),
            positive_weights=torch.tensor([float(cfg.get("aux_like_pos_weight", 4.0))], dtype=logits.dtype, device=logits.device),
        ).mean()
        if aux_like_weight > 0 and "aux_like_logit" in outputs and "aux_labels" in batch
        else logits.new_tensor(0.0)
    )
    aux_comment = (
        focal_bce_with_logits(
            outputs["aux_comment_logit"].unsqueeze(-1),
            batch["aux_labels"][:, 1:2],
            gamma=float(cfg.get("focal_gamma", 1.5)),
            positive_weights=torch.tensor([float(cfg.get("aux_comment_pos_weight", 6.0))], dtype=logits.dtype, device=logits.device),
        ).mean()
        if aux_comment_weight > 0 and "aux_comment_logit" in outputs and "aux_labels" in batch
        else logits.new_tensor(0.0)
    )
    page_time_mask = None
    if "page_time_log" in batch:
        page_time_mask = (batch["labels"][:, 0] > 0) | (batch["page_time_log"] > 0)
    aux_page_time = (
        F.smooth_l1_loss(outputs["aux_page_time"][page_time_mask], batch["page_time_log"][page_time_mask])
        if aux_page_time_weight > 0 and "aux_page_time" in outputs and page_time_mask is not None and bool(page_time_mask.any())
        else logits.new_tensor(0.0)
    )
    total = (
        task_loss
        + bpr_weight * bpr
        + task_bpr_weight * task_bpr
        + listwise_weight * listwise
        + collect_listwise_weight * collect_listwise
        + share_listwise_weight * share_listwise
        + type_transition_weight * type_transition
        + taxonomy_transition_weight * taxonomy_transition
        + contrastive_weight * contrastive
        + aux_like_weight * aux_like
        + aux_comment_weight * aux_comment
        + aux_page_time_weight * aux_page_time
    )
    logs = {
        "loss": total.detach(),
        "task_loss": task_loss.detach(),
        "bce_click_loss": bce[0].detach(),
        "bce_collect_loss": bce[1].detach(),
        "bce_share_loss": bce[2].detach(),
        "bpr_loss": bpr.detach(),
        "task_bpr_loss": task_bpr.detach(),
        "listwise_loss": listwise.detach(),
        "collect_listwise_loss": collect_listwise.detach(),
        "share_listwise_loss": share_listwise.detach(),
        "transition_loss": type_transition.detach(),
        "type_transition_loss": type_transition.detach(),
        "taxonomy_transition_loss": taxonomy_transition.detach(),
        "contrastive_loss": contrastive.detach(),
        "aux_like_loss": aux_like.detach(),
        "aux_comment_loss": aux_comment.detach(),
        "aux_page_time_loss": aux_page_time.detach(),
    }
    return total, logs
