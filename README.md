# HeteroIntent-PLE

面向 Qilin 多源异构内容推荐的精排系统。模型对同一 `request_id` 下的候选 item 同时预测：

```text
p_click, p_collect, p_share
```

最终排序分数为：

```text
score = 0.3 * p_click + 0.4 * p_collect + 0.3 * p_share
```

当前主线版本是 **大 PLE 多目标模型 + parquet 元信息多模态统一表征**。由于本地没有真实图片/视频文件，图像和视频模态暂时来自 Qilin `notes` 表中的结构化元信息，例如 `image_path`、`image_num`、`note_type`、`video_duration`、`video_height`、`video_width`。

## 当前版本

| 项目 | 路径或数值 |
| --- | --- |
| 配置文件 | `configs/qilin_full.yaml` |
| 处理后数据 | `data/processed/qilin_full_multimodal_meta` |
| 训练输出 | `outputs/qilin_full_multimodal_meta` |
| 最优 checkpoint | `outputs/qilin_full_multimodal_meta/best.pt` |
| Top-20 文件 | `outputs/qilin_full_multimodal_meta/submission_top20_dedup.csv` |
| 参数量 | 155,110,672 |
| 最佳验证轮次 | epoch 4 |
| 完成训练轮数 | 12 |

当前配置要点：

```yaml
model:
  hidden_dim: 512
  shared_experts: 8
  task_experts: 4
  ple_layers: 3

loss:
  bpr_weight: 0.01
  task_bpr_weight: 0.005
  contrastive_weight: 0.0

data:
  processed_dir: data/processed/qilin_full_multimodal_meta

train:
  output_dir: outputs/qilin_full_multimodal_meta
```

## 最新指标

验证集最佳结果来自 `outputs/qilin_full_multimodal_meta/metrics.csv` 的第 4 轮。

| 指标 | Valid | Test |
| --- | ---: | ---: |
| WeightedHit@20 | 0.242119 | 0.321044 |
| NDCG@20 | 0.520923 | 0.674518 |
| HitClick@20 | 0.755382 | 0.992803 |
| HitCollect@20 | 0.029133 | 0.046064 |
| HitShare@20 | 0.012838 | 0.015924 |
| AUC Click | 0.680019 | 0.696142 |
| AUC Collect | 0.817309 | 0.800443 |
| AUC Share | 0.682769 | 0.675038 |

与原始 `outputs/qilin_full` baseline 相比，当前多模态元信息版在测试集上：

| 指标 | 变化 |
| --- | ---: |
| WeightedHit@20 | -0.000234 |
| NDCG@20 | +0.019684 |
| HitCollect@20 | +0.000090 |
| AUC Click | +0.037177 |
| AUC Share | +0.013644 |

结论：当前多模态元信息主要改善排序位置质量和部分 AUC，尚未带来主指标 `WeightedHit@20` 的提升。真实图片/视频 embedding 缺失是主要限制。

## 模型设计

### 1. 多模态统一表征

`scripts/prepare_qilin.py` 会在 Qilin 转换阶段生成：

- `image_feat_*`：由 `image_num`、`image_path` 数量、是否多图、路径 hash bucket 等元信息构成。
- `video_feat_*`：由 `note_type`、视频时长、分辨率、面积、宽高比、是否横屏/竖屏等元信息构成。
- `dense_feat_*`：保留原始统计特征，不破坏 baseline。

`ItemEncoder` 会把 item ID、类目、位置、text、image-meta、video-meta、dense、graph 分别投影到统一向量空间，再通过 gate fusion 得到 item 表征。缺失模态会被 mask 掉，特征全 0 的 image/video 不参与 softmax 融合。

评估输出中会统计 `mean_gate_*` 和 `top20_mean_gate_*`。最近一次测试集中 Top-20 平均 gate 大致为：

| 模态 | Top-20 gate |
| --- | ---: |
| graph | 0.504662 |
| dense | 0.238981 |
| text | 0.076525 |
| video-meta | 0.061091 |
| image-meta | 0.036860 |

### 2. 动态意图理解

数据转换阶段会基于历史 item 类型和类目生成动态意图字段，例如：

```text
target_item_type, target_taxonomy_id,
hist_dominant_item_type, hist_dominant_taxonomy_id,
is_type_shift, is_taxonomy_shift, has_intent_target,
hist_item_type_*, hist_taxonomy_id_*
```

模型通过 `type_transition_head` 和 `taxonomy_transition_head` 预测用户意图转移。当前配置中：

```yaml
type_transition_weight: 0.03
taxonomy_transition_weight: 0.03
```

### 3. 多目标联合优化

训练目标由三任务 BCE/Focal BCE、加权排序 BPR、分任务 BPR 和动态意图辅助损失组成：

```text
L = task BCE/Focal
  + bpr_weight * request-level weighted BPR
  + task_bpr_weight * per-task BPR
  + type_transition_weight * CE(type)
  + taxonomy_transition_weight * CE(taxonomy)
```

当前 collect/share 是稀疏高价值行为，因此配置中保留正样本加权：

```yaml
positive_weights:
  click: 1.0
  collect: 4.0
  share: 3.0
```

## 环境

推荐使用已经验证可用的 Conda CUDA 环境：

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe
```

检查 GPU：

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

不要使用项目 `.venv` 跑全量训练，因为其中可能是 CPU 版 PyTorch。

## 数据目录

原始 Qilin 数据应放在：

```text
data/raw/Qilin/
  recommendation_train/*.parquet
  recommendation_test/*.parquet
  notes/*.parquet
  user_feat/*.parquet
```

当前主线 processed 目录：

```text
data/processed/qilin_full_multimodal_meta/
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
```

## 运行流程

以下命令均在项目根目录执行：

```powershell
cd C:\Users\31278\Desktop\heterointent
```

### 1. 转换 Qilin 数据

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\prepare_qilin.py --qilin-dir data\raw\Qilin --output-dir data\processed\qilin_full_multimodal_meta --max-history 20 --text-hash-dim 0
```

### 2. 生成文本 embedding

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\build_text_embeddings.py --qilin-dir data\raw\Qilin --processed-dir data\processed\qilin_full_multimodal_meta --model-name D:\models\bge-small-zh-v1.5 --batch-size 256 --max-length 256 --pooling cls --device cuda
```
这一阶段需要为约 94.6 万个 item 生成文本向量，耗时明显长于单轮训练是正常现象。BGE 系列建议使用 `--pooling cls`；如果换回 Qwen 等通用 Transformer，可以使用默认的 `--pooling mean`。

```powershell
--pooling cls
```

### 3. 合并文本 embedding

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\merge_embeddings.py --processed-dir data\processed\qilin_full_multimodal_meta --text
```

### 4. 构建 item graph

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\build_item_graph.py --processed-dir data\processed\qilin_full_multimodal_meta --embed-dim 64
```

### 5. 训练

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\train.py --config configs\qilin_full.yaml
```

断点续训：

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\train.py --config configs\qilin_full.yaml --resume outputs\qilin_full_multimodal_meta\last.pt
```

### 6. 验证和测试

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\evaluate.py --checkpoint outputs\qilin_full_multimodal_meta\best.pt --samples data\processed\qilin_full_multimodal_meta\valid.parquet --device cuda --batch-size 2048

D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\evaluate.py --checkpoint outputs\qilin_full_multimodal_meta\best.pt --samples data\processed\qilin_full_multimodal_meta\test.parquet --device cuda --batch-size 2048
```

### 7. 导出 Top-20

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\infer.py --checkpoint outputs\qilin_full_multimodal_meta\best.pt --samples data\processed\qilin_full_multimodal_meta\test.parquet --output outputs\qilin_full_multimodal_meta\submission_top20_dedup.csv --device cuda --batch-size 2048
```

输出字段：

```text
request_id, rank, item_id, score, p_click, p_collect, p_share
```

## 常用脚本

| 脚本 | 作用 |
| --- | --- |
| `scripts/prepare_qilin.py` | 转换 Qilin parquet，生成训练/验证/测试表和 metadata |
| `scripts/build_text_embeddings.py` | 生成离线文本 embedding |
| `scripts/merge_embeddings.py` | 合并文本或视觉 embedding 到样本表 |
| `scripts/build_item_graph.py` | 构建 item graph embedding |
| `scripts/build_visual_embeddings.py` | 预留真实图片 embedding 入口 |
| `scripts/train.py` | 训练或断点续训 |
| `scripts/evaluate.py` | 验证/测试评估 |
| `scripts/infer.py` | 导出 Top-20 推荐结果 |

## 指标说明

- `WeightedHit@20`：主指标近似值，等于 `0.3*HitClick@20 + 0.4*HitCollect@20 + 0.3*HitShare@20`。
- `NDCG@20`：考虑排序位置的加权相关性指标。
- `HitClick/Collect/Share@20`：Top-20 中是否命中对应行为正样本，并对请求取平均。
- `RecallClick/Collect/Share@20`：只在有对应正样本的请求上计算召回。
- `AUC Click/Collect/Share`：三类二分类任务的区分能力。
- `mean_gate_*`：所有候选样本上的平均模态 gate 权重。
- `top20_mean_gate_*`：Top-20 候选上的平均模态 gate 权重。

## Visual Thumbnail Pipeline

The visual version keeps CLIP/SigLIP outside the deployed recommender. Image/video thumbnail embeddings are generated offline, compressed to 128 dimensions, then appended after the existing image/video metadata features.

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\build_visual_embeddings.py --modality both --qilin-dir data\raw\Qilin --processed-dir data\processed\qilin_full_multimodal_meta --image-root data\raw\Qilin\images --video-root data\raw\Qilin\video_thumbnails --model-name openai/clip-vit-base-patch32 --output-dim 128 --compression auto --batch-size 64 --device cuda

D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\merge_embeddings.py --processed-dir data\processed\qilin_full_multimodal_meta --output-dir data\processed\qilin_full_multimodal_visual --image --video --merge-mode auto

D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\check_parameter_budget.py --config configs\qilin_score_opt_mild_visual.yaml --budget-mb 800 --fail-over-budget

D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\train.py --config configs\qilin_score_opt_mild_visual.yaml
```

For a fast no-download smoke test, add `--mock-encoder --max-items 1000` to `build_visual_embeddings.py`. The mock vectors are deterministic and only validate the data path; they are not meant for final metrics.

## 后续优化方向

1. 接入真实图片文件后，用 `scripts/build_visual_embeddings.py` 生成 CLIP/SigLIP 图像 embedding，再与当前 image metadata 拼接或替换。
2. 将文本 embedding 模型从 Qwen2-0.5B 替换为更适合检索的 BGE/E5 类模型，比较 NDCG 和 AUC。
3. 对 collect/share 做更细的采样和重加权实验，重点观察 HitCollect@20、HitShare@20 与 WeightedHit@20 的权衡。
4. 对 item graph 做去噪或按行为类型加权，避免 graph gate 过强时压制 image/video/text 的增量贡献。
