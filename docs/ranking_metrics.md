# 排序评估指标口径

当前默认评估只保留能直接反映排序能力的核心指标。完整诊断指标仍可通过
`compute_ranking_metrics(..., include_diagnostics=True)` 临时打开。

## 核心指标

- `weighted_hit@20`：比赛近似主分，按 `0.3*click + 0.4*collect + 0.3*share` 汇总 Top-20 命中。它用于和比赛口径对齐，但在候选数小于等于 20 的请求上容易饱和。
- `ndcg@20`：考虑排序位置的加权相关性。越喜欢的物品越靠前，分数越高。
- `preference_auc`：请求内成对排序一致性。比较同一请求下任意两个候选，如果真实偏好更高的物品得分也更高，则记为排序正确。
- `hard_weighted_hit@20`：只在候选数大于 TopK 的请求上计算 `weighted_hit@20`，用于过滤 Top-20 天然覆盖全部候选的简单请求。
- `hard_ndcg@20`：只在候选数大于 TopK 的请求上计算 `ndcg@20`。
- `hard_preference_auc`：只在候选数大于 TopK 的请求上计算 `preference_auc`。
- `request_auc_click`：请求内点击正负样本的区分能力。
- `request_auc_collect`：请求内收藏正负样本的区分能力。
- `request_auc_share`：请求内分享正负样本的区分能力。
- `request_ap_collect`：收藏目标的请求内平均精度，更适合观察稀疏正样本是否被排到前面。
- `request_ap_share`：分享目标的请求内平均精度，更适合观察稀疏正样本是否被排到前面。

## 覆盖率指标

- `candidate_count`：每个请求的平均候选数。
- `candidate_count_gt_topk_rate`：候选数大于 TopK 的请求比例。这个比例越低，Top-20 命中越容易饱和。
- `hard_topk_request_rate`：实际进入 hard 指标统计的请求比例，通常等于 `candidate_count_gt_topk_rate`。
- `request_auc_request_rate_click`：点击 AUC 有效请求比例。没有正负样本同时存在的请求会被跳过。
- `request_auc_request_rate_collect`：收藏 AUC 有效请求比例。
- `request_auc_request_rate_share`：分享 AUC 有效请求比例。
- `num_requests` / `num_rows`：参与评估的请求数和候选行数。

## 使用建议

模型选择仍保留 `weighted_hit@20`，因为它最接近比赛提交分；但判断真实排序能力时，应优先看
`ndcg@20`、`preference_auc`、`hard_ndcg@20`、`hard_preference_auc`，以及
`request_auc_collect/share` 和 `request_ap_collect/share`。
