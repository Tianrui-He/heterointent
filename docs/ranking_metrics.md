# Native 评估指标口径

当前数据集的核心问题是：多数请求候选数不超过 Top-20，click 覆盖占主导，collect/share 正样本极稀疏。因此主线不再使用旧 `quality_score` / `stable_selection_score` 选模，改用面向 hard request 和稀疏行为的 `native_selection_score`。

## 核心指标

- `native_selection_score`：主选模指标。
- `official_weighted_hit@20`：官方公式近似口径，按 `0.3*click + 0.4*collect + 0.3*share` 汇总 Top-20 命中；仅用于对齐比赛，不单独决定模型好坏。
- `hard_weighted_hit@20`：只在 `candidate_count > topk` 的请求上计算官方 hit。
- `hard_official_capture`：`hard_weighted_hit@20 / oracle_score_eligible`，表示 hard 请求上捕获了多少理论可得分。
- `hard_ndcg@20`：hard 请求上的位置敏感排序质量。
- `hard_preference_auc`：hard 请求上的请求内加权偏好 AUC。
- `sparse_ap`：hard 请求上 collect/share AP 的加权平均。
- `sparse_recall`：hard 请求上 collect/share recall@20 的加权平均。
- `topk_boundary_success_*`：对应任务正样本是否越过 Top-K 边界。
- `topk_boundary_margin_*`：对应任务最高正样本分数减去第 K 名分数。

## 主选模公式

```text
native_selection_score =
  0.30 * hard_official_capture
+ 0.25 * hard_ndcg@20
+ 0.20 * hard_preference_auc
+ 0.15 * sparse_ap
+ 0.10 * sparse_recall
```

接受新模型时还要看：

```text
hard_weighted_hit@20 不明显下降
sparse_ap / sparse_recall 不明显下降
paired bootstrap 的 P(delta > 0) >= 0.8
```

## 稳定性评估

使用 `scripts/compare_predictions_bootstrap.py` 对两个预测文件做 request-level paired bootstrap。它输出：

- `mean_delta`
- `95% CI`
- `P(delta > 0)`
- candidate 分桶 delta
- hard / rare hard 分桶 delta

这个脚本用于判断窄分数区间内的提升是否可信，而不是只看单次 valid/test 均值。
