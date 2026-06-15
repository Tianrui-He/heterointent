# HeteroIntent-PLE：赛题二多源异构多目标推荐精排系统

本项目面向“2026 年中国研究生人工智能创新大赛华为赛题二：面向多源异构内容的用户意图感知与多目标推荐排序优化”。当前版本已经接入全量 Qilin 数据，完成数据转换、文本 embedding 合并、item graph 构建、训练、验证、测试和 Top-20 推理导出。

模型目标是对同一推荐请求下的每个候选 item 预测：

```text
p_click, p_collect, p_share
```

最终排序分数固定为：

```text
score = 0.3 * p_click + 0.4 * p_collect + 0.3 * p_share
```

项目内部训练表是“一行 = 一个 request 下的一个候选 item”。推理时会按 `request_id` 分组排序，输出每个请求下得分最高的 Top-20 item。当前导出逻辑已经对同一 `request_id + item_id` 去重，同一请求不会重复推荐同一个物品。

## 当前最终结果

最终 checkpoint：

```text
outputs/qilin_full/best.pt
```

最终提交/推理文件：

```text
outputs/qilin_full/submission_top20_dedup.csv
```

最终指标记录：

```text
outputs/qilin_full/final_eval_metrics.json
```

训练概况：

| 项目 | 数值 |
| --- | ---: |
| 最后训练轮数 | 10 |
| 最佳验证轮数 | 5 |
| 模型参数量 | 122,767,046 |
| 训练设备 | cuda |
| 最佳验证 WeightedHit@20 | 0.242119 |

验证集结果：

> Recall 指标当前采用标准口径：只在该行为有正样本的请求上取平均；旧的全请求平均值保存在 `final_eval_metrics.json` 的 `*_overall@20` 字段中。

| 指标 | 数值 |
| --- | ---: |
| WeightedHit@20 | 0.242119 |
| NDCG@20 | 0.505449 |
| HitClick@20 | 0.756172 |
| RecallClick@20 | 0.932421 |
| HitCollect@20 | 0.028540 |
| RecallCollect@20 | 0.904946 |
| HitShare@20 | 0.012838 |
| RecallShare@20 | 0.899767 |
| AUC Click | 0.658068 |
| AUC Collect | 0.845413 |
| AUC Share | 0.646035 |
| 请求数 | 10,126 |
| 去重后样本行数 | 122,123 |
| 移除重复候选行数 | 3,464 |

测试集结果：

| 指标 | 数值 |
| --- | ---: |
| WeightedHit@20 | 0.321278 |
| NDCG@20 | 0.654834 |
| HitClick@20 | 0.993612 |
| RecallClick@20 | 0.916099 |
| HitCollect@20 | 0.045974 |
| RecallCollect@20 | 0.879879 |
| HitShare@20 | 0.016014 |
| RecallShare@20 | 0.816407 |
| AUC Click | 0.658965 |
| AUC Collect | 0.815774 |
| AUC Share | 0.661394 |
| 请求数 | 11,115 |
| 去重后样本行数 | 179,401 |
| 移除重复候选行数 | 3,166 |

提交文件检查：

| 项目 | 数值 |
| --- | ---: |
| 输出行数 | 123,474 |
| 请求数 | 11,115 |
| 重复 request-item pair | 0 |
| rank 不连续请求数 | 0 |
| 输出 20 个 item 的请求数 | 2,963 |
| 少于 20 个 item 的请求数 | 8,152 |

说明：有不少请求少于 20 个 item，是因为候选池去重后本身不足 20 个唯一 item。当前项目按“给定候选池精排”处理，不额外召回热门 item 补齐。

## 环境

推荐使用已经验证可用的 Conda CUDA 环境：

```text
D:\adaconda3\envs\MiniOneRec-pre\python.exe
```

该环境已验证可识别：

```text
NVIDIA GeForce RTX 5070
PyTorch CUDA 12.8
```

检查 GPU：

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

不要使用项目 `.venv` 训练全量模型，因为 `.venv` 中可能是 CPU 版 PyTorch。

## 目录结构

```text
configs/
  qilin_full.yaml

data/raw/Qilin/
  recommendation_train/
  recommendation_test/
  notes/
  user_feat/

data/processed/qilin_full/
  train.parquet
  valid.parquet
  test.parquet
  metadata.json
  item_id_map.parquet
  user_id_map.parquet
  taxonomy_id_map.parquet
  text_embeddings.npy
  text_embedding_item_ids.npy
  text_embedding_items.parquet
  graph_embedding.npy

outputs/qilin_full/
  best.pt
  last.pt
  metrics.csv
  summary.json
  valid_predictions.parquet
  final_eval_metrics.json
  submission_top20_dedup.csv

scripts/
  prepare_qilin.py
  build_text_embeddings.py
  build_visual_embeddings.py
  merge_embeddings.py
  build_item_graph.py
  train.py
  evaluate.py
  infer.py

src/heterointent/
  data/
  models/
  training/
  evaluation/
  inference/
```

## 完整操作流程

以下命令均在项目根目录运行：

```powershell
cd C:\Users\31278\Desktop\heterointent
```

### 1. 准备 Qilin 原始数据

原始数据目录应为：

```text
data/raw/Qilin/
  recommendation_train/*.parquet
  recommendation_test/*.parquet
  notes/*.parquet
  user_feat/*.parquet
```

### 2. 转换 Qilin 数据

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\prepare_qilin.py --qilin-dir data\raw\Qilin --output-dir data\processed\qilin_full --max-history 20 --text-hash-dim 0
```

这一步会展开 Qilin 的候选列表，生成一行一个候选 item 的训练表，并创建连续 ID 映射。

### 3. 生成文本 embedding

当前项目使用本地 Qwen2-0.5B 生成文本向量，路径示例：

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\build_text_embeddings.py --qilin-dir data\raw\Qilin --processed-dir data\processed\qilin_full --model-name D:\models\Qwen2-0.5B --batch-size 64 --max-length 256 --device cuda
```

也可以替换为更适合检索的 embedding 模型，例如 BGE：

```powershell
--model-name D:\models\bge-small-zh-v1.5
```

### 4. 合并文本 embedding

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\merge_embeddings.py --processed-dir data\processed\qilin_full --text
```

### 5. 构建 item graph

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\build_item_graph.py --processed-dir data\processed\qilin_full --embed-dim 64
```

### 6. 训练

从头训练：

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\train.py --config configs\qilin_full.yaml
```

从断点继续训练：

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\train.py --config configs\qilin_full.yaml --resume outputs\qilin_full\last.pt
```

当前配置使用：

```yaml
device: cuda
batch_size: 2048
amp: true
fast_loader: true
output_dir: outputs/qilin_full
```

### 7. 验证集评估

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\evaluate.py --checkpoint outputs\qilin_full\best.pt --samples data\processed\qilin_full\valid.parquet --device cuda --batch-size 2048
```

### 8. 测试集评估

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\evaluate.py --checkpoint outputs\qilin_full\best.pt --samples data\processed\qilin_full\test.parquet --device cuda --batch-size 2048
```

### 9. 导出 Top-20

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\infer.py --checkpoint outputs\qilin_full\best.pt --samples data\processed\qilin_full\test.parquet --output outputs\qilin_full\submission_top20_dedup.csv --device cuda --batch-size 2048
```

输出字段：

```text
request_id, rank, item_id, score, p_click, p_collect, p_share
```

## 指标含义

- `p_click`：模型预测候选 item 被点击的概率。
- `p_collect`：模型预测候选 item 被收藏的概率。
- `p_share`：模型预测候选 item 被分享的概率。
- `score`：最终排序分数，等于 `0.3*p_click + 0.4*p_collect + 0.3*p_share`。
- `HitClick@20`：每个请求的 Top-20 中是否命中至少一个点击正样本，然后对请求取平均。
- `HitCollect@20`：每个请求的 Top-20 中是否命中至少一个收藏正样本，然后对请求取平均。
- `HitShare@20`：每个请求的 Top-20 中是否命中至少一个分享正样本，然后对请求取平均。
- `RecallClick@20`：只在有点击正样本的请求上计算，等于 Top-20 命中的点击正样本数 / 该请求全部点击正样本数。
- `RecallCollect@20`：只在有收藏正样本的请求上计算；没有收藏行为的请求不参与该平均值。
- `RecallShare@20`：只在有分享正样本的请求上计算；没有分享行为的请求不参与该平均值。
- `RecallClick/Collect/Share_overall@20`：旧口径，把没有对应正样本的请求按 0 计入平均，用于观察行为稀疏性对整体数值的稀释。
- `WeightedHit@20`：主指标近似值，等于 `0.3*HitClick@20 + 0.4*HitCollect@20 + 0.3*HitShare@20`。
- `NDCG@20`：考虑排序位置的加权相关性指标，越靠前命中高价值行为得分越高。
- `AUC Click/Collect/Share`：三类二分类任务的排序区分能力。
- `num_rows_after_dedup`：按 `request_id + item_id` 去重后的评估行数。
- `num_duplicate_rows_removed`：评估前移除的重复候选行数。

## 文件含义

### 配置

- `configs/qilin_full.yaml`：唯一保留的最终训练配置，控制数据路径、模型结构、训练参数、输出目录。

### 数据

- `data/raw/Qilin/`：官方 Qilin 原始 parquet 数据，不应修改。
- `data/processed/qilin_full/train.parquet`：训练集，一行一个候选 item。
- `data/processed/qilin_full/valid.parquet`：验证集。
- `data/processed/qilin_full/test.parquet`：测试集。
- `data/processed/qilin_full/metadata.json`：模型需要的数据规模和特征维度。
- `data/processed/qilin_full/item_id_map.parquet`：原始 `note_idx` 到连续 `item_id` 的映射。
- `data/processed/qilin_full/user_id_map.parquet`：原始 `user_idx` 到连续 `user_id` 的映射。
- `data/processed/qilin_full/taxonomy_id_map.parquet`：类目到连续 `taxonomy_id` 的映射。
- `data/processed/qilin_full/text_embeddings.npy`：离线文本 embedding 矩阵。
- `data/processed/qilin_full/text_embedding_item_ids.npy`：文本 embedding 对应的 `item_id`。
- `data/processed/qilin_full/graph_embedding.npy`：item 共现图生成的图增强 embedding。

### 输出

- `outputs/qilin_full/best.pt`：验证集 WeightedHit@20 最优 checkpoint，用于最终评估和推理。
- `outputs/qilin_full/last.pt`：最后一轮 checkpoint，用于断点续训。
- `outputs/qilin_full/metrics.csv`：每轮训练 loss 和验证指标。
- `outputs/qilin_full/summary.json`：训练摘要，包括最佳指标、参数量、训练设备和最后轮数。
- `outputs/qilin_full/valid_predictions.parquet`：最佳模型在验证集上的逐候选预测结果。
- `outputs/qilin_full/final_eval_metrics.json`：本次最终验证集和测试集指标汇总。
- `outputs/qilin_full/submission_top20_dedup.csv`：最终 Top-20 推荐结果，已去重。

### 脚本

- `scripts/prepare_qilin.py`：官方 Qilin 数据转换。
- `scripts/build_text_embeddings.py`：生成文本 embedding。
- `scripts/build_visual_embeddings.py`：可选，生成视觉 embedding。
- `scripts/merge_embeddings.py`：把离线 embedding 合并进训练表。
- `scripts/build_item_graph.py`：构建 item graph embedding。
- `scripts/train.py`：训练和断点续训入口。
- `scripts/evaluate.py`：验证/测试指标评估入口。
- `scripts/infer.py`：Top-20 推理导出入口。

## 当前模型表现解读

当前模型对点击目标排序较强，测试集 `HitClick@20` 达到 0.9936，`RecallClick@20` 达到 0.9161。收藏和分享在有正样本请求上的召回并不低：测试集 `RecallCollect@20` 为 0.8799，`RecallShare@20` 为 0.8164；但由于只有 5.11% 的测试请求存在收藏行为、1.90% 的测试请求存在分享行为，整体 Hit 指标仍然较低。

后续如果继续追求效果，可以优先尝试：

1. 使用更适合 embedding 的中文模型，如 `bge-small-zh-v1.5` 或 `bge-base-zh-v1.5`。
2. 对 collect/share 做更强的样本重加权或 focal 参数调优。
3. 恢复部分辅助损失并观察速度与效果的折中。
4. 增加视觉特征或更强的 item graph 去噪。

## ?????????

????????????????? + ???????????? `outputs/qilin_full` checkpoint ????? baseline?????????? `type_transition_head` ? `taxonomy_transition_head`????????????????

```text
outputs/qilin_full_dynamic/
```

### 1. ????????

????? `data/processed/qilin_full`?????

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\annotate_dynamic_intent.py --processed-dir data\processed\qilin_full --qilin-dir data\raw\Qilin --max-history 20
```

?????? processed ???????? `data/raw/Qilin`???? `train.parquet`?`valid.parquet`?`test.parquet` ???

```text
target_item_type, target_taxonomy_id,
hist_dominant_item_type, hist_dominant_taxonomy_id,
is_type_shift, is_taxonomy_shift, has_intent_target,
hist_item_type_0..19, hist_taxonomy_id_0..19
```

??????? `scripts/prepare_qilin.py`?????????????????????

### 2. ??????????

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\train.py --config configs\qilin_full.yaml
```

??????????

```yaml
loss:
  type_transition_weight: 0.03
  taxonomy_transition_weight: 0.03

train:
  output_dir: outputs/qilin_full_dynamic
```

???????? `outputs/qilin_full/last.pt` ????????????????????????????????????????? checkpoint?

### 3. ????????

- `intent_type_acc@1`?????? Top-1 ????????????????? item ? `item_type`?
- `intent_type_acc@2`?Top-2 ?????????????
- `intent_taxonomy_acc@1`?Top-1 ????????????? item ? `taxonomy_id`?
- `intent_taxonomy_acc@5`?Top-5 ?????????????
- `intent_taxonomy_mrr`??????????????????????????
- `shift_type_hit@1`??????????????? `intent_type_acc@1`?
- `shift_taxonomy_hit@5`??????????????? `intent_taxonomy_acc@5`?
- `ranking_weighted_hit@20_shift`?????????????????????
- `ranking_weighted_hit@20_stable`??????????????????
- `attention_target_mass`?DIN attention ?????????/????????????????

????????????????????????????????????????? `WeightedHit@20` ????????? `ranking_weighted_hit@20_shift` ????

