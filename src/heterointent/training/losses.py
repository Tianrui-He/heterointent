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
    min_group_size: int = 1,
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
    min_group_size = max(int(min_group_size), 1)
    for start, count in zip(starts_list, counts_list):
        if count <= min_group_size:
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
    min_group_size: int = 1,
) -> torch.Tensor:
    if task_weights is None:
        task_weights = torch.tensor(DEFAULT_TASK_WEIGHTS, dtype=labels.dtype, device=labels.device)
    rel = labels @ task_weights
    return _group_bpr_loss(
        scores,
        rel,
        request_ids,
        max_groups=max_groups,
        max_pairs_per_group=max_pairs_per_group,
        min_group_size=min_group_size,
    )


def request_task_bpr_loss(
    task_scores: torch.Tensor,
    labels: torch.Tensor,
    request_ids: torch.Tensor,
    task_weights: torch.Tensor,
    max_groups: int = DEFAULT_BPR_MAX_GROUPS,
    max_pairs_per_group: int = DEFAULT_BPR_MAX_PAIRS_PER_GROUP,
    min_group_size: int = 1,
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
                min_group_size=min_group_size,
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
    min_group_size: int = 1,
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
    min_group_size = max(int(min_group_size), 1)
    for start, count in zip(starts_list, counts_list):
        if count <= min_group_size:
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
    min_group_size: int = 1,
) -> torch.Tensor:
    if task_weights is None:
        task_weights = torch.tensor(DEFAULT_TASK_WEIGHTS, dtype=labels.dtype, device=labels.device)
    rel = labels @ task_weights
    return _group_listwise_loss(
        scores,
        rel,
        request_ids,
        max_groups=max_groups,
        temperature=temperature,
        min_group_size=min_group_size,
    )


def request_single_task_listwise_loss(
    task_scores: torch.Tensor,
    task_labels: torch.Tensor,
    request_ids: torch.Tensor,
    max_groups: int = DEFAULT_LISTWISE_MAX_GROUPS,
    temperature: float = 1.0,
    min_group_size: int = 1,
) -> torch.Tensor:
    return _group_listwise_loss(
        task_scores,
        task_labels,
        request_ids,
        max_groups=max_groups,
        temperature=temperature,
        min_group_size=min_group_size,
    )


def request_topk_coverage_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    request_ids: torch.Tensor,
    task_weights: torch.Tensor | None = None,
    topk: int = 20,
    max_groups: int = DEFAULT_LISTWISE_MAX_GROUPS,
    temperature: float = 0.05,
    margin: float = 0.0,
    min_group_size: int | None = None,
) -> torch.Tensor:
    if scores.numel() == 0:
        return scores.new_tensor(0.0)
    max_groups = max(int(max_groups), 0)
    if max_groups == 0:
        return scores.new_tensor(0.0)
    if task_weights is None:
        task_weights = torch.tensor(DEFAULT_TASK_WEIGHTS, dtype=labels.dtype, device=labels.device)
    topk = max(int(topk), 1)
    temperature = max(float(temperature), 1e-6)
    margin = float(margin)
    min_group_size = topk if min_group_size is None else max(int(min_group_size), topk)

    order = torch.argsort(request_ids)
    sorted_request_ids = request_ids[order]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    _, counts = torch.unique_consecutive(sorted_request_ids, return_counts=True)
    starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    starts_list = starts.detach().cpu().tolist()
    counts_list = counts.detach().cpu().tolist()

    weighted_losses = []
    used_weights = []
    used_groups = 0
    for start, count in zip(starts_list, counts_list):
        if count <= min_group_size:
            continue
        end = start + count
        group_scores = sorted_scores[start:end]
        group_labels = sorted_labels[start:end]
        threshold = torch.topk(group_scores.detach(), k=topk).values[-1]
        group_used = False
        for task_idx in range(min(labels.size(-1), len(TASK_NAMES))):
            weight = task_weights[task_idx].clamp_min(0)
            if weight <= 0:
                continue
            pos_scores = group_scores[group_labels[:, task_idx] > 0]
            if pos_scores.numel() == 0:
                continue
            if pos_scores.numel() == 1:
                smooth_pos = pos_scores.squeeze(0)
            else:
                smooth_pos = temperature * torch.logsumexp(pos_scores / temperature, dim=0)
            weighted_losses.append(weight * F.softplus((threshold + margin - smooth_pos) / temperature))
            used_weights.append(weight)
            group_used = True
        if group_used:
            used_groups += 1
            if used_groups >= max_groups:
                break
    if not weighted_losses:
        return scores.new_tensor(0.0)
    return torch.stack(weighted_losses).sum() / torch.stack(used_weights).sum().clamp_min(1e-12)


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = target.gt(0)
    if not bool(mask.any()):
        return logits.new_tensor(0.0)
    target = target.clamp(min=0, max=logits.size(-1) - 1)
    return F.cross_entropy(logits[mask], target[mask])


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: dict,
    topk: int = 20,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
    hard_bpr_weight = float(cfg.get("hard_bpr_weight", cfg.get("hard_rank_loss_weight", 0.0)))
    hard_task_bpr_weight = float(cfg.get("hard_task_bpr_weight", 0.0))
    bpr_max_groups = int(cfg.get("bpr_max_groups", DEFAULT_BPR_MAX_GROUPS))
    bpr_max_pairs_per_group = int(cfg.get("bpr_max_pairs_per_group", DEFAULT_BPR_MAX_PAIRS_PER_GROUP))
    listwise_weight = float(cfg.get("listwise_weight", 0.0))
    hard_listwise_weight = float(cfg.get("hard_listwise_weight", cfg.get("hard_rank_loss_weight", 0.0)))
    collect_listwise_weight = float(cfg.get("collect_listwise_weight", 0.0))
    share_listwise_weight = float(cfg.get("share_listwise_weight", 0.0))
    hard_collect_listwise_weight = float(cfg.get("hard_collect_listwise_weight", 0.0))
    hard_share_listwise_weight = float(cfg.get("hard_share_listwise_weight", 0.0))
    topk_coverage_weight = float(cfg.get("topk_coverage_weight", 0.0))
    topk_coverage_max_groups = int(cfg.get("topk_coverage_max_groups", DEFAULT_LISTWISE_MAX_GROUPS))
    topk_coverage_temperature = float(cfg.get("topk_coverage_temperature", 0.05))
    topk_coverage_margin = float(cfg.get("topk_coverage_margin", 0.0))
    listwise_max_groups = int(cfg.get("listwise_max_groups", DEFAULT_LISTWISE_MAX_GROUPS))
    listwise_temperature = float(cfg.get("listwise_temperature", 1.0))
    rank_min_group = int(cfg.get("rank_loss_min_group_size", 1))
    hard_rank_min_group = int(cfg.get("hard_rank_loss_min_group_size", topk))
    type_transition_weight = float(cfg.get("type_transition_weight", 0.0))
    taxonomy_transition_weight = float(cfg.get("taxonomy_transition_weight", 0.0))
    aux_like_weight = float(cfg.get("aux_like_weight", 0.0))
    aux_comment_weight = float(cfg.get("aux_comment_weight", 0.0))
    aux_page_time_weight = float(cfg.get("aux_page_time_weight", 0.0))
    rank_scores = outputs["weighted_prob_score"] if "weighted_prob_score" in outputs else outputs["final_score"]

    bpr = (
        request_bpr_loss(
            rank_scores,
            labels,
            batch["request_id"],
            task_weights,
            max_groups=bpr_max_groups,
            max_pairs_per_group=bpr_max_pairs_per_group,
            min_group_size=rank_min_group,
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
            min_group_size=rank_min_group,
        )
        if task_bpr_weight > 0
        else logits.new_tensor(0.0)
    )
    hard_bpr = (
        request_bpr_loss(
            rank_scores,
            labels,
            batch["request_id"],
            task_weights,
            max_groups=bpr_max_groups,
            max_pairs_per_group=bpr_max_pairs_per_group,
            min_group_size=hard_rank_min_group,
        )
        if hard_bpr_weight > 0
        else logits.new_tensor(0.0)
    )
    hard_task_bpr = (
        request_task_bpr_loss(
            logits,
            labels,
            batch["request_id"],
            task_weights,
            max_groups=bpr_max_groups,
            max_pairs_per_group=bpr_max_pairs_per_group,
            min_group_size=hard_rank_min_group,
        )
        if hard_task_bpr_weight > 0
        else logits.new_tensor(0.0)
    )
    listwise = (
        request_listwise_loss(
            rank_scores,
            labels,
            batch["request_id"],
            task_weights,
            max_groups=listwise_max_groups,
            temperature=listwise_temperature,
            min_group_size=rank_min_group,
        )
        if listwise_weight > 0
        else logits.new_tensor(0.0)
    )
    hard_listwise = (
        request_listwise_loss(
            rank_scores,
            labels,
            batch["request_id"],
            task_weights,
            max_groups=listwise_max_groups,
            temperature=listwise_temperature,
            min_group_size=hard_rank_min_group,
        )
        if hard_listwise_weight > 0
        else logits.new_tensor(0.0)
    )
    collect_listwise = (
        request_single_task_listwise_loss(
            logits[:, 1],
            labels[:, 1],
            batch["request_id"],
            max_groups=listwise_max_groups,
            temperature=listwise_temperature,
            min_group_size=rank_min_group,
        )
        if collect_listwise_weight > 0
        else logits.new_tensor(0.0)
    )
    hard_collect_listwise = (
        request_single_task_listwise_loss(
            logits[:, 1],
            labels[:, 1],
            batch["request_id"],
            max_groups=listwise_max_groups,
            temperature=listwise_temperature,
            min_group_size=hard_rank_min_group,
        )
        if hard_collect_listwise_weight > 0
        else logits.new_tensor(0.0)
    )
    share_listwise = (
        request_single_task_listwise_loss(
            logits[:, 2],
            labels[:, 2],
            batch["request_id"],
            max_groups=listwise_max_groups,
            temperature=listwise_temperature,
            min_group_size=rank_min_group,
        )
        if share_listwise_weight > 0
        else logits.new_tensor(0.0)
    )
    hard_share_listwise = (
        request_single_task_listwise_loss(
            logits[:, 2],
            labels[:, 2],
            batch["request_id"],
            max_groups=listwise_max_groups,
            temperature=listwise_temperature,
            min_group_size=hard_rank_min_group,
        )
        if hard_share_listwise_weight > 0
        else logits.new_tensor(0.0)
    )
    topk_coverage = (
        request_topk_coverage_loss(
            rank_scores,
            labels,
            batch["request_id"],
            task_weights=task_weights,
            topk=topk,
            max_groups=topk_coverage_max_groups,
            temperature=topk_coverage_temperature,
            margin=topk_coverage_margin,
            min_group_size=hard_rank_min_group,
        )
        if topk_coverage_weight > 0
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
        + hard_bpr_weight * hard_bpr
        + hard_task_bpr_weight * hard_task_bpr
        + listwise_weight * listwise
        + hard_listwise_weight * hard_listwise
        + collect_listwise_weight * collect_listwise
        + share_listwise_weight * share_listwise
        + hard_collect_listwise_weight * hard_collect_listwise
        + hard_share_listwise_weight * hard_share_listwise
        + topk_coverage_weight * topk_coverage
        + type_transition_weight * type_transition
        + taxonomy_transition_weight * taxonomy_transition
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
        "hard_bpr_loss": hard_bpr.detach(),
        "hard_task_bpr_loss": hard_task_bpr.detach(),
        "listwise_loss": listwise.detach(),
        "hard_listwise_loss": hard_listwise.detach(),
        "collect_listwise_loss": collect_listwise.detach(),
        "share_listwise_loss": share_listwise.detach(),
        "hard_collect_listwise_loss": hard_collect_listwise.detach(),
        "hard_share_listwise_loss": hard_share_listwise.detach(),
        "topk_coverage_loss": topk_coverage.detach(),
        "transition_loss": type_transition.detach(),
        "type_transition_loss": type_transition.detach(),
        "taxonomy_transition_loss": taxonomy_transition.detach(),
        "aux_like_loss": aux_like.detach(),
        "aux_comment_loss": aux_comment.detach(),
        "aux_page_time_loss": aux_page_time.detach(),
    }
    return total, logs
