# HeteroIntent-PLE

面向 Qilin 多源异构内容推荐的精排系统。模型对同一 `request_id` 下的候选 item 同时预测 `p_click`、`p_collect`、`p_share`，并按任务权重融合为最终排序分数：

```text
score = 0.3 * p_click + 0.4 * p_collect + 0.3 * p_share
```

当前主线：**Feature-Opt v2 + History Semantic + Compact sidecar + SigLIP 视觉向量 + PLE 多目标训练**。推理分数严格使用官方公式，不再混合独立 rank head。

---

## 当前版本一览

| 项目 | 路径 / 数值 |
| --- | --- |
| 主配置 | `configs/qilin_feature_opt_v2_history_compact.yaml` |
| 处理后数据 | `data/processed/qilin_v2` |
| 已完成训练输出 | `outputs/qilin_v2_visual_first_image` |
| 最优 checkpoint | `outputs/qilin_v2_visual_first_image/best.pt` |
| Top-20 提交 | `outputs/qilin_v2_visual_first_image/submission_top20_dedup.csv` |
| Gate 诊断 | `outputs/qilin_v2_visual_first_image/test_gate_metrics.json` |
| 展示台数据 | `outputs/showcase_qilin_v2_visual_first_image_posttrain` |
| 参数量 | 155,081,092 |
| 选模指标 | `native_selection_score` |
| 最佳 epoch | 3（早停于 epoch 6） |
| 本轮耗时 | 6 轮约 17.3 min，均值约 2.88 min/epoch |
| 推理公式 | `0.3 * p_click + 0.4 * p_collect + 0.3 * p_share` |

> 注意：当前 YAML 中 `train.output_dir` 指向 `outputs\qilin_intend_transition`，用于动态意图转移主线实验。下表指标来自已经训练完成的 `outputs/qilin_v2_visual_first_image/best.pt`。

---

## 最新指标

本轮使用 `data/processed/qilin_v2`，其中图片视觉向量由 `build_visual_embeddings.py` 从本地图片目录重新生成：`visual_embedding_source = build_visual_embeddings`，`visual_sidecar_source = None`，`image_emb_dim = 128`。首图版 SigLIP/PCA sidecar 覆盖 `155,534 / 945,683` 个 item。

| 指标 | Valid best epoch 3 | Test best.pt | 说明 |
| --- | ---: | ---: | --- |
| Native selection | **0.7009** | **0.6988** | 当前选模指标 |
| OfficialWeightedHit@20 | 0.2435 | **0.3241** | 官方权重 Top-20 命中 |
| HardWeightedHit@20 | 0.3238 | **0.3305** | 候选数 >20 的可区分请求 |
| HardOfficialCapture | 0.9751 | **0.9757** | hard 请求相对 oracle 捕获率 |
| HardNDCG@20 | 0.5242 | **0.5354** | hard 请求排序质量 |
| HardPreference AUC | **0.6938** | 0.6856 | hard 请求 pairwise 区分 |
| Sparse AP | 0.3407 | **0.3425** | collect/share 稀疏正样本 AP |
| Sparse Recall | **0.8747** | 0.8373 | collect/share 稀疏召回 |
| Rare score | **0.5036** | 0.4828 | rare 请求稳定性 |

解读：这次不再只是 official hit 的窄幅波动，而是 hard NDCG、hard AUC、collect/share 稀疏 AP 与 sparse recall 同时改善。`best.pt` 明显优于 `last.pt`，因此后续评估、提交和展示均使用 `outputs/qilin_v2_visual_first_image/best.pt`。

### Best 与 Last 对比

| 指标 | best epoch 3 | last epoch 6 | last - best |
| --- | ---: | ---: | ---: |
| Native selection | 0.7009 | 0.6772 | -0.0237 |
| HardNDCG@20 | 0.5242 | 0.4914 | -0.0328 |
| Sparse AP | 0.3407 | 0.2897 | -0.0510 |
| Hard request AP share | 0.3189 | 0.2438 | -0.0751 |

### 数据与提交核查

| 项目 | 数值 |
| --- | ---: |
| Valid 请求 / 去重后评估行数 | 10,126 / 122,123 |
| Test 请求 / 去重后评估行数 | 11,115 / 179,401 |
| Test 平均候选数 | 16.14 |
| Test hard 请求占比 | 24.97% |
| Test click / collect / share 正样本行数 | 44,685 / 801 / 250 |
| 提交行数 / 请求数 | 123,474 / 11,115 |
| 提交重复 request-item | 0 |

### Top-20 模态 gate（测试集）

| 模态 | Top-20 平均 gate |
| --- | ---: |
| graph | 0.3624 |
| dense | 0.3397 |
| image-emb | 0.0928 |
| text | 0.0566 |
| video-meta | 0.0165 |
| image-meta | 0.0127 |

Gate 诊断说明这次 `image_emb` 已经实际参与排序，不再是接近 0 的空挂载。当前模型仍主要依赖 graph 与 dense 信号，但视觉向量已经成为可解释的第三层信号。

---



## 环境准备

```powershell
$PY = "D:\adaconda3\envs\MiniOneRec-pre\python.exe"
cd C:\Users\31278\Desktop\heterointent
& $PY -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
& $PY -m pip install -r requirements.txt
```

依赖本地模型与图片（视觉重编码时需要）：

```powershell
Test-Path data\raw\Qilin
Test-Path D:\models\bge-small-zh-v1.5
Test-Path D:\models\siglip-base-patch16-224
Test-Path "E:\qilin\mnt\ali-sh-1\usr\lihaitao\process_0106\image\part_0"
```

---



## 目录结构

```text
data/raw/Qilin/                          # 官方原始 parquet
data/processed/
  qilin_base/                            # prepare_qilin 输出
  qilin_v2/                              # upgrade v2 历史特征 + text/image sidecar
  visual_path_cache_siglip/              # path-level SigLIP cache，可断点续跑
outputs/qilin_v2_visual_first_image/
  best.pt / last.pt / metrics.csv
  summary.json
  valid_predictions.parquet
  submission_top20_dedup.csv
  test_gate_metrics.json
```

---



## 完整流程（实测可跑通）

以下均在项目根目录执行。**实测耗时**（RTX 5070，全量 Qilin）供参考。

### 0. 判定是否需要重建 processed

```powershell
& $PY -c "import json, pathlib; p=pathlib.Path('data/processed/qilin_v2/metadata.json'); print('metadata_exists =', p.exists()); m=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}; fs=m.get('feature_sidecars', {}); print('visual_embedding_source =', m.get('visual_embedding_source')); print('visual_sidecar_source =', m.get('visual_sidecar_source')); print('image_emb =', fs.get('image_emb')); print('image_emb_dim =', m.get('image_emb_dim'))"
```

若输出为 `visual_embedding_source = build_visual_embeddings`、`visual_sidecar_source = None`，且 `image_emb_dim = 128`，说明 `data\processed\qilin_v2` 已经使用当前视觉重编码流程，可以从步骤 6 开始检查并训练。

若 `metadata_exists = False`，从步骤 0.1 开始跑完整流程；若仍能看到 `visual_sidecar_source` 或缺少 `image_emb`，说明当前训练会缺少新的视觉 sidecar，需要先完成步骤 5，再训练。

### 0.1 创建输出目录

```powershell
New-Item -ItemType Directory -Force -Path `
  data\processed\qilin_base,`
  data\processed\qilin_v2,`
  outputs\qilin_v2_visual_first_image
```



### 1. 原始数据 → processed base（~2 min）

```powershell
& $PY scripts\prepare_qilin.py `
  --qilin-dir data\raw\Qilin `
  --output-dir data\processed\qilin_base `
  --max-history 20 `
  --text-hash-dim 0
```

### 2. 构图 embedding（~30 s）

```powershell
& $PY scripts\build_item_graph.py `
  --processed-dir data\processed\qilin_base `
  --embed-dim 64
```



### 3. 文本 / 查询 sidecar（GPU，~25 min）

脚本默认一次编码 **joint + title + content + query**（见 `build_text_embeddings.py` 中 `--item-texts` 默认 `joint title content`，且 `--query` 默认开启）。**一条命令即可**，无需分两次：

```powershell
& $PY scripts\build_text_embeddings.py `
  --processed-dir data\processed\qilin_base `
  --qilin-dir data\raw\Qilin `
  --model-name D:\models\bge-small-zh-v1.5 `
  --batch-size 512 --pooling cls --device cuda
```

若只需补跑部分视图（例如已有 joint，缺 query），可显式指定，并加上 `--no-query` 跳过 query：

```powershell
# 仅补 title / content（不重复 joint、不编 query）
& $PY scripts\build_text_embeddings.py `
  --processed-dir data\processed\qilin_base `
  --qilin-dir data\raw\Qilin `
  --model-name D:\models\bge-small-zh-v1.5 `
  --batch-size 512 --pooling cls --device cuda `
  --item-texts title content --no-query
```

> **注意**：步骤 3 **全部跑完**后再执行 `upgrade`（不要与编码并行），否则 `qilin_v2` 复制 sidecar 时会缺文件。



### 4. 升级 v2 历史语义特征（~1 min）

```powershell
& $PY scripts\upgrade_feature_opt_v2_columns.py `
  --processed-dir data\processed\qilin_base `
  --output-dir data\processed\qilin_v2 `
  --qilin-dir data\raw\Qilin `
  --max-history 20
```



### 5. 视觉 sidecar（GPU，耗时取决于图片覆盖）

主线下一轮不再读取旧视觉 sidecar。当前训练配置直接读取 `data\processed\qilin_v2`，不要再使用旧实验数据目录。

当前视觉脚本会先按 `part_N/path` 排序读取图片，再写 path-level cache。这样可以把百万级小图片的跨目录随机跳读降到最低；中断后重跑会复用 `data\processed\visual_path_cache_siglip`。

本轮推荐先做**首图/封面版**视觉向量：`--max-images-per-item 1`。它把唯一图片数从约 117.6 万降到约 46.2 万，足够验证视觉是否对主线有收益；如果后续确认视觉有效，再考虑 `--max-images-per-item 4` 的多图均值版本。

#### 5.1 生成图片 embedding sidecar

先单独跑图片向量。当前推荐 `batch_size=512`、`image_workers=8`、首图模式；若显存不足，把 `--batch-size 512` 改为 `256` 或 `128`。

这一步不是一次性直接写出 `image_embeddings.npy`，而是分三层推进：

| 阶段 | 会生成/更新的文件 | 含义 |
| --- | --- | --- |
| 1. path-level cache | `data\processed\visual_path_cache_siglip\path_embeddings.npy`、`path_embedding_index.parquet` | 图片路径到 SigLIP 原始 768 维向量的缓存，可跨实验复用 |
| 2. item-level sidecar | `data\processed\qilin_v2\image_embeddings.npy`、`image_embedding_item_ids.npy`、`image_embedding_items.parquet` | 当前 `item_id_map` 下供训练读取的 128 维 item 视觉向量 |
| 3. metadata 注册 | `data\processed\qilin_v2\metadata.json` | 写入 `feature_sidecars.image_emb`、`image_emb_dim=128` |

因此运行过程中只看到 `visual_path_cache_siglip` 增长、还没有 `image_embeddings.npy` 是正常的；只有所有缺失图片编码完成、聚合并 PCA 压缩后，最终 sidecar 才会一次性写出。

```powershell
& $PY scripts\build_visual_embeddings.py `
  --processed-dir data\processed\qilin_v2 `
  --qilin-dir data\raw\Qilin `
  --modality image `
  --image-root "E:\qilin\mnt\ali-sh-1\usr\lihaitao\process_0106\image" `
  --model-name D:\models\siglip-base-patch16-224 `
  --device cuda `
  --fp16 `
  --cache-dir data\processed\visual_path_cache_siglip `
  --batch-size 512 `
  --image-workers 8 `
  --prefetch-batches 8 `
  --fast-preprocess `
  --max-images-per-item 1 `
  --output-dim 128 `
  --compression pca
```

成功后应出现：

```powershell
Test-Path data\processed\qilin_v2\image_embeddings.npy
Test-Path data\processed\qilin_v2\image_embedding_item_ids.npy
Test-Path data\processed\qilin_v2\image_embedding_items.parquet
```

运行中可用下面的命令查看当前阶段：

```powershell
& $PY -c "import pathlib, numpy as np, pandas as pd; cache=pathlib.Path('data/processed/visual_path_cache_siglip'); idx=cache/'path_embedding_index.parquet'; val=cache/'path_embeddings.npy'; print('cache rows =', len(pd.read_parquet(idx)) if idx.exists() else 0); print('cache shape =', np.load(val, mmap_mode='r').shape if val.exists() else None); proc=pathlib.Path('data/processed/qilin_v2'); print('image sidecar exists =', (proc/'image_embeddings.npy').exists()); print('image ids exists =', (proc/'image_embedding_item_ids.npy').exists()); print('image items exists =', (proc/'image_embedding_items.parquet').exists())"
```

这一步跑完后，后续大部分实验**不需要再读 E 盘图片**。可复用层级如下：

| 场景 | 是否需要重跑视觉编码 |
| --- | --- |
| 只改训练配置、loss、指标或重新训练 | 不需要；直接复用 `data\processed\qilin_v2` |
| 输出目录被清理，但 `data\processed\qilin_v2\image_embeddings.npy` 还在 | 不需要；直接重新训练 |
| `image_embeddings.npy` 缺失，但 `visual_path_cache_siglip` 还在 | 需要跑步骤 5.1；脚本会复用已缓存路径，只编码缺失图片 |
| 从 `--max-images-per-item 1` 改成 `4` | 需要跑步骤 5.1；首图 cache 会复用，只补额外图片 |
| 改 SigLIP 模型、图片预处理尺寸或重建了 `item_id_map` | 不要直接复用最终 `image_embeddings.npy`；应重新生成视觉 sidecar |

`data\processed\visual_path_cache_siglip` 是 path-level 原始 SigLIP cache，跟训练实验无关，建议长期保留；`data\processed\qilin_v2\image_embeddings.npy` 是 item-level 训练 sidecar，要求 `qilin_v2` 的 `item_id_map` 不变。

#### 5.2 核查训练数据状态

```powershell
& $PY -c "import json, pathlib; p=pathlib.Path('data/processed/qilin_v2/metadata.json'); m=json.loads(p.read_text(encoding='utf-8')); fs=m.get('feature_sidecars', {}); print('visual_embedding_source =', m.get('visual_embedding_source')); print('visual_sidecar_source =', m.get('visual_sidecar_source')); print('image_emb =', fs.get('image_emb')); print('image_emb_dim =', m.get('image_emb_dim'))"
```

期望看到 `visual_embedding_source = build_visual_embeddings`、`visual_sidecar_source = None`，且 `image_emb` 存在、`image_emb_dim = 128`。如果 `image_emb` 缺失，说明步骤 5.1 没有成功写入 sidecar，不要直接训练。

### 6. 训练前检查

```powershell
& $PY -m pytest tests\test_metrics.py tests\test_losses.py tests\test_rank.py
& $PY scripts\check_parameter_budget.py `
  --config configs\qilin_feature_opt_v2_history_compact.yaml --budget-mb 800
```



### 7. 正式训练（本轮实测约 2.88 min/epoch，6 轮约 17.3 min）

```powershell
& $PY scripts\train.py --config configs\qilin_feature_opt_v2_history_compact.yaml
```

当前配置的 `train.output_dir` 是 `outputs\qilin_intend_transition`，用于动态意图转移主线实验。

**训练加速要点**（实测）：


| 配置                         | 推荐值              | 错误示例                             |
| -------------------------- | ---------------- | -------------------------------- |
| `fast_loader`              | `true`           | —                                |
| `pin_memory`               | `true`           | `false`                          |
| `tensor_device`            | **不设置**（数据留 CPU） | `cuda`（占满 12GB 显存，~20 min/epoch） |
| `batch_size`               | `3072`           | —                                |
| `request_preserving_train` | `true`           | —                                |


断点续训：

```powershell
& $PY scripts\train.py `
  --config configs\qilin_feature_opt_v2_history_compact.yaml `
  --resume outputs\qilin_v2_visual_first_image\last.pt
```

断点续训会按当前配置写入新的 `train.output_dir`；如果只需要使用本轮结果，不要从 `last.pt` 继续训练，直接使用 `best.pt`。



### 8. 评估

```powershell
& $PY scripts\evaluate.py `
  --checkpoint outputs\qilin_v2_visual_first_image\best.pt `
  --samples data\processed\qilin_v2\test.parquet `
  --batch-size 8192 --fast-loader --topk 20
```

模型对比使用 paired bootstrap。当前仓库不再保留旧 `baseline/` 目录；只有当你手里已有上一轮模型的 `valid_predictions.parquet` 时才需要跑这一步：

```powershell
& $PY scripts\compare_predictions_bootstrap.py `
  --old outputs\<old_run>\valid_predictions.parquet `
  --new outputs\qilin_v2_visual_first_image\valid_predictions.parquet `
  --output outputs\qilin_v2_visual_first_image\bootstrap_compare.json
```

如果没有旧预测文件，先跳过 bootstrap；本轮训练完成后保留 `outputs\qilin_v2_visual_first_image\valid_predictions.parquet`，后续新实验再把它作为 `--old`。

### 9. 导出 Top-20

```powershell
& $PY scripts\infer.py `
  --checkpoint outputs\qilin_v2_visual_first_image\best.pt `
  --samples data\processed\qilin_v2\test.parquet `
  --output outputs\qilin_v2_visual_first_image\submission_top20_dedup.csv `
  --batch-size 8192 --topk 20
```

输出字段：`request_id, rank, item_id, score, p_click, p_collect, p_share`

### 10. Gate 诊断

```powershell
& $PY scripts\report_gate_metrics.py `
  --checkpoint outputs\qilin_v2_visual_first_image\best.pt `
  --samples data\processed\qilin_v2\test.parquet `
  --batch-size 8192 --topk 20 `
  --output outputs\qilin_v2_visual_first_image\test_gate_metrics.json
```



### 11. 展示台（可选）

```powershell
& $PY scripts\export_showcase_data.py `
  --processed-dir data\processed\qilin_v2 `
  --run-dir outputs\qilin_v2_visual_first_image `
  --checkpoint outputs\qilin_v2_visual_first_image\best.pt `
  --config configs\qilin_feature_opt_v2_history_compact.yaml `
  --output-dir outputs\showcase_qilin_v2_visual_first_image_posttrain `
  --max-cases 120

& $PY demo\showcase_app.py --data-dir outputs\showcase_qilin_v2_visual_first_image_posttrain --host 127.0.0.1 --port 7860
```

---



## 模型与训练摘要

- **ItemEncoder**：多模态 gate fusion（text/title/content、graph、dense、视觉等）
- **UserInterestEncoder**：Transformer 历史序列 + history semantic 摘要（v2）
- **PLERanker**：8 shared × 4 task experts × 3 层
- **选模**：验证集 `native_selection_score` 最大化保存 `best.pt`
- **损失**：加权 Focal BCE + 全请求 BPR/listwise + hard 请求额外 BPR/listwise + collect/share 单任务 listwise + Top-K 覆盖边界 loss + 辅助头（like/comment/page_time）

更完整指标定义见 `docs/ranking_metrics.md`。

---



## 常用脚本


| 脚本                                  | 作用                           |
| ----------------------------------- | ---------------------------- |
| `prepare_qilin.py`                  | 原始 Qilin → processed parquet |
| `build_item_graph.py`               | 构建 `graph_embedding.npy`     |
| `build_text_embeddings.py`          | BGE 文本/查询 sidecar            |
| `build_visual_embeddings.py`        | SigLIP 图片 sidecar |
| `upgrade_feature_opt_v2_columns.py` | 追加 v2 历史语义列                  |
| `compact_processed_features.py`     | compact parquet + mmap sidecar |
| `build_processed_compact.py`        | 一体化入口；全量视觉不建议一把梭            |
| `train.py`                          | 训练 / 断点续训                    |
| `evaluate.py`                       | 单 split 指标                   |
| `infer.py`                          | 导出 Top-20 CSV                |
| `compare_predictions_bootstrap.py`  | paired bootstrap 模型对比        |
| `report_gate_metrics.py`            | Top-20 模态 gate 统计            |


---



## 常见问题

**Q: 训练很慢 / 显存满但 GPU 功率低？**
A: 检查是否误设 `tensor_device: cuda`。应使用 `fast_loader: true` + `pin_memory: true`，数据留在 CPU；本轮实测约 2.88 分钟/epoch。

**Q: compact 与全量 processed 区别？**
A: 特征语义相同；compact 将 512 维 text/query 移出 parquet，训练时 mmap sidecar 加载。

**Q: 视觉向量从哪来？**
A: 主线用 `build_visual_embeddings.py` 从 `E:\qilin\mnt\ali-sh-1\usr\lihaitao\process_0106\image` 读取图片并用本地 SigLIP 在 GPU 上生成 `data\processed\qilin_v2` 下的 `image_emb` sidecar。脚本会按 `part_N/path` 排序读取，并使用 `data\processed\visual_path_cache_siglip` 断点续跑。

**Q: 视觉编码跑一次后能复用吗？**
A: 可以。改训练、loss、指标或重新 compact 时，复用 `data\processed\qilin_v2\image_embeddings.npy` 即可，不再读图片；如果只剩 `visual_path_cache_siglip`，重跑 `build_visual_embeddings.py` 会跳过已缓存路径。只有改 SigLIP 模型、图片预处理尺寸、`max-images-per-item` 聚合策略，或重建 `item_id_map` 时，才需要重新生成 item-level sidecar。

**Q: 为什么已经有 `visual_path_cache_siglip`，但没有 `image_embeddings.npy`？**
A: `visual_path_cache_siglip` 只是图片路径级中间缓存，会在每个大 chunk 后落盘；`image_embeddings.npy` 要等所有缺失路径编码完、按 item 聚合、PCA 压缩到 128 维后才写出。看到 cache 增长但最终 sidecar 还不存在，通常说明视觉脚本仍在步骤 5.1 中运行。

**Q: 为什么不直接用 `build_processed_compact.py` 一步跑视觉？**
A: 全量图片约百万级唯一路径，瓶颈主要是小文件 I/O 和图片解码。一体化命令失败后不易判断进度，也容易在 compact 前耗很久；分成“视觉 sidecar → compact”后可以单独观察 cache、覆盖率和 GPU 利用率。

**Q: upgrade 与 text 编码顺序？**
A: 必须先完成全部 text sidecar，再跑 `upgrade_feature_opt_v2_columns.py`（或之后手动补拷 sidecar）。
