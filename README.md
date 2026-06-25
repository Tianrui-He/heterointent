# HeteroIntent-PLE

面向 Qilin 多源异构内容推荐的精排系统。模型对同一 `request_id` 下的候选 item 同时预测 `p_click`、`p_collect`、`p_share`，并按任务权重融合为最终排序分数：

```text
score = 0.3 * p_click + 0.4 * p_collect + 0.3 * p_share
```

当前**推荐主线**为 **Feature-Opt v2 + Compact 数据布局 + SigLIP 视觉 sidecar + 大 PLE 多目标训练**。文本/查询向量以 mmap sidecar 方式加载，避免 parquet 内嵌 512 维列导致 IO 与磁盘膨胀；视觉向量从离线 SigLIP 编码结果挂载，不重复跑文本 embedding。

---

## 当前版本一览


| 项目            | 路径或数值                                                                     |
| ------------- | ------------------------------------------------------------------------- |
| 主配置           | `configs/qilin_feature_opt_v2_history_compact.yaml`                       |
| 处理后数据         | `data/processed/qilin_full_feature_opt_v2_compact`                        |
| 训练输出          | `outputs/qilin_feature_opt_v2_history_compact`                            |
| 最优 checkpoint | `outputs/qilin_feature_opt_v2_history_compact/best.pt`                    |
| 测试指标          | `outputs/qilin_feature_opt_v2_history_compact/test_metrics.json`          |
| Gate 诊断       | `outputs/qilin_feature_opt_v2_history_compact/test_gate_metrics.json`     |
| Top-20 提交     | `outputs/qilin_feature_opt_v2_history_compact/submission_top20_dedup.csv` |
| 展示台数据         | `outputs/showcase_feature_opt_v2_compact`                                 |
| 参数量           | 155,091,782                                                               |
| 最佳验证轮次        | epoch 5（按 `quality_score` 选模）                                             |
| 完成训练轮数        | 8（早停）                                                                     |


### 最新测试集指标（best.pt）


| 指标                  | Test   |
| ------------------- | ------ |
| quality_score       | 0.6381 |
| WeightedHit@20      | 0.3228 |
| NDCG@20             | 0.7290 |
| Preference AUC      | 0.7171 |
| Request AUC Collect | 0.7999 |
| Request AUC Share   | 0.7698 |


相对无视觉 embedding 的 `feature_opt_v2` 非 compact 版本，同测试集上 NDCG@20 约 **+2.7%**（0.730 vs 0.703），`quality_score` 约 **+2.8%**。

### Top-20 模态 gate（分组口径）

评估脚本按 `item_dense + ratio` 合并为 dense，`text_fused` 为 text：


| 模态         | Top-20 gate |
| ---------- | ----------- |
| graph      | 0.2909      |
| dense      | 0.3391      |
| text       | 0.1328      |
| video-meta | 0.0524      |
| image-meta | 0.0112      |
| image-emb  | ≈0          |
| video-emb  | 0.0008      |


`image_emb` 权重极低，因当前仅 **91,566 / 945,683（9.7%）** item 有 SigLIP 向量，且 Top-20 候选命中率低。

---

## 环境准备

推荐使用已验证的 Conda CUDA 环境（不要用项目内 CPU 版 `.venv` 跑全量训练）：

```powershell
$PY = "D:\adaconda3\envs\MiniOneRec-pre\python.exe"
cd C:\Users\31278\Desktop\heterointent
```

检查 GPU：

```powershell
& $PY -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

安装依赖（若尚未安装）：

```powershell
& $PY -m pip install -r requirements.txt
```

---

## 目录结构

### 原始数据

```text
data/raw/Qilin/
  recommendation_train/*.parquet
  recommendation_test/*.parquet
  notes/*.parquet
  user_feat/*.parquet
```

### 主线 processed（compact）

```text
data/processed/qilin_full_feature_opt_v2_compact/
  train.parquet / valid.parquet / test.parquet   # 精简列，无内嵌 512 维 text/query
  metadata.json
  item_id_map.parquet / user_id_map.parquet / ...
  text_embeddings.npy + text_embedding_item_ids.npy
  text_title_embeddings.npy / text_content_embeddings.npy
  query_embeddings.npy + query_embedding_request_ids.npy
  image_embeddings.npy / video_embeddings.npy     # 视觉 sidecar（可选）
  graph_embedding.npy
  compact_summary.json
  visual_sidecar_attach.json                      # 记录视觉 sidecar 来源
```

### 本地图片根目录（视觉编码用）

```text
E:\qilin\mnt\ali-sh-1\usr\lihaitao\process_0106\image\
  part_0/ ... part_50/
```

当前已编码路径仅覆盖 `**part_0`–`part_29**`（由 `notes.image_path` 决定，不是扫描全部分片）。

---

## 完整操作流程

以下命令均在**项目根目录**执行。脚本会自动把 `src` 加入 `PYTHONPATH`。

### 流程总览

```text
[一次性] 原始 Qilin → feature_opt_v2 全量 processed
    ↓
[推荐一步] build_processed_compact.py
           = 文本/查询 embedding（可跳过已有）
           + 从本地图片路径 SigLIP 视觉 embedding
           + compact 精简 parquet
    ↓
训练 → 验证/测试评估 → Top-20 导出 → Gate 诊断 → 展示台
```

若你已有 `data/processed/qilin_full_feature_opt_v2_compact` 且 sidecar 齐全，可从 **步骤 3** 直接训练。

视觉向量**直接从 `notes.image_path` + `--image-root` 解析文件并编码**，不复制其他实验目录的 sidecar。

---

### 步骤 1：转换 Qilin 为 processed（首次）

```powershell
& $PY scripts\prepare_qilin.py `
  --qilin-dir data\raw\Qilin `
  --output-dir data\processed\qilin_full_feature_opt `
  --max-history 20 `
  --text-hash-dim 0
```

构建 item graph（全量 processed 上执行一次即可，compact 会继承）：

```powershell
& $PY scripts\build_item_graph.py `
  --processed-dir data\processed\qilin_full_feature_opt `
  --embed-dim 64
```

升级到 Feature-Opt v2 列（历史语义、cold_stage、text_stat、image/video meta 等）：

```powershell
& $PY scripts\upgrade_feature_opt_v2_columns.py `
  --processed-dir data\processed\qilin_full_feature_opt `
  --output-dir data\processed\qilin_full_feature_opt_v2 `
  --qilin-dir data\raw\Qilin `
  --max-history 20
```

---

### 步骤 2：一体化 embedding + compact（推荐）

`scripts/build_processed_compact.py` 在**同一流程**中完成：

1. 文本/查询 BGE 编码（默认跳过已存在的 sidecar）
2. 按 `notes.image_path` 在 `--image-root` 下定位图片，SigLIP 编码视觉向量
3. 更新 `metadata.json` 的 sidecar 配置
4. 输出 compact parquet（剥离内嵌高维列，保留 mmap sidecar）

```powershell
& $PY scripts\build_processed_compact.py `
  --processed-dir data\processed\qilin_full_feature_opt_v2 `
  --output-dir data\processed\qilin_full_feature_opt_v2_compact `
  --qilin-dir data\raw\Qilin `
  --image-root "E:\qilin\mnt\ali-sh-1\usr\lihaitao\process_0106\image" `
  --text-model-name D:\models\bge-small-zh-v1.5 `
  --text-pooling cls `
  --visual-model-name D:\models\siglip-base-patch16-224 `
  --visual-output-dim 128 `
  --visual-compression pca `
  --visual-batch-size 64 `
  --visual-fp16 `
  --visual-image-workers 4 `
  --visual-cache-dir data\processed\visual_path_cache_siglip `
  --text-device cuda `
  --visual-device cuda
```

常用变体：


| 场景                          | 命令追加                                    |
| --------------------------- | --------------------------------------- |
| 文本已编码，只补视觉 + compact        | `--skip-text`                           |
| 视觉已编码，只补文本 + compact        | `--skip-visual`                         |
| 仅 compact（sidecar 已在 v2 目录） | `--skip-text --skip-visual`             |
| 强制全部重编码                     | `--force-reencode`                      |
| 冒烟测试                        | `--mock-visual --visual-max-items 1000` |


产物：

- `compact_summary.json` — parquet 列精简统计
- `pipeline_summary.json` — 本次 text/visual/compact 摘要
- `visual_embedding_summary.json` — 图片路径命中与编码统计
- `image_embeddings.npy` / `video_embeddings.npy` — 由**当前图片路径**编码得到

path 级 cache 写在 `--visual-cache-dir`；重复执行时**自动跳过**已有文本 sidecar 与已缓存图片路径，不会无谓重跑 BGE。

#### 分步执行（可选）

若希望拆开跑，仍可单独调用：

```powershell
# 仅文本
& $PY scripts\build_text_embeddings.py --qilin-dir data\raw\Qilin --processed-dir data\processed\qilin_full_feature_opt_v2 --model-name D:\models\bge-small-zh-v1.5 --item-texts joint title content --query --pooling cls --device cuda

# 仅从图片路径编码视觉
& $PY scripts\build_visual_embeddings.py --modality both --processed-dir data\processed\qilin_full_feature_opt_v2 --qilin-dir data\raw\Qilin --image-root "E:\qilin\mnt\ali-sh-1\usr\lihaitao\process_0106\image" --model-name D:\models\siglip-base-patch16-224 --output-dim 128 --compression pca --batch-size 64 --fp16 --cache-dir data\processed\visual_path_cache_siglip --device cuda

# 仅 compact
& $PY scripts\compact_processed_features.py --processed-dir data\processed\qilin_full_feature_opt_v2 --output-dir data\processed\qilin_full_feature_opt_v2_compact
```

`scripts/merge_embeddings.py` 仅用于把 embedding **写回 parquet 宽表**；compact 主线不需要。

---

### 步骤 3：训练

```powershell
& $PY scripts\train.py --config configs\qilin_feature_opt_v2_history_compact.yaml
```

断点续训：

```powershell
& $PY scripts\train.py `
  --config configs\qilin_feature_opt_v2_history_compact.yaml `
  --resume outputs\qilin_feature_opt_v2_history_compact\last.pt
```

训练产出：


| 文件                          | 说明                     |
| --------------------------- | ---------------------- |
| `best.pt`                   | 验证集 `quality_score` 最优 |
| `last.pt`                   | 最后一轮（可与 best 二选一保留）    |
| `metrics.csv`               | 每轮 train/valid 指标      |
| `summary.json`              | 最优轮次与选模指标              |
| `valid_predictions.parquet` | 验证集预测（展示台用）            |
| `config.yaml`               | 训练配置快照                 |


当前配置要点：`batch_size=3072`，`fast_loader=true`，`selection_metric=quality_score`，`enable_intent_heads=false`，`use_text_fusion_gate=true`，`use_query_interaction=true`，`use_history_semantic=true`。

---

### 步骤 4：验证 / 测试评估

```powershell
& $PY scripts\evaluate.py `
  --checkpoint outputs\qilin_feature_opt_v2_history_compact\best.pt `
  --samples data\processed\qilin_full_feature_opt_v2_compact\valid.parquet `
  --batch-size 8192 `
  --fast-loader `
  --topk 20

& $PY scripts\evaluate.py `
  --checkpoint outputs\qilin_feature_opt_v2_history_compact\best.pt `
  --samples data\processed\qilin_full_feature_opt_v2_compact\test.parquet `
  --batch-size 8192 `
  --fast-loader `
  --topk 20
```

---

### 步骤 5：导出测试集 Top-20

```powershell
& $PY scripts\infer.py `
  --checkpoint outputs\qilin_feature_opt_v2_history_compact\best.pt `
  --samples data\processed\qilin_full_feature_opt_v2_compact\test.parquet `
  --output outputs\qilin_feature_opt_v2_history_compact\submission_top20_dedup.csv `
  --batch-size 8192 `
  --topk 20
```

输出字段：

```text
request_id, rank, item_id, score, p_click, p_collect, p_share
```

---

### 步骤 6：模态 Gate 诊断

```powershell
& $PY scripts\report_gate_metrics.py `
  --checkpoint outputs\qilin_feature_opt_v2_history_compact\best.pt `
  --samples data\processed\qilin_full_feature_opt_v2_compact\test.parquet `
  --output outputs\qilin_feature_opt_v2_history_compact\test_gate_metrics.json
```

输出包含 `grouped_top20`（graph / dense / text 等分组）与 `raw_top20`（`ItemEncoder.part_names` 全部分量）。

---

### 步骤 7：生成并启动展示台

```powershell
& $PY scripts\export_showcase_data.py `
  --processed-dir data\processed\qilin_full_feature_opt_v2_compact `
  --run-dir outputs\qilin_feature_opt_v2_history_compact `
  --checkpoint outputs\qilin_feature_opt_v2_history_compact\best.pt `
  --config configs\qilin_feature_opt_v2_history_compact.yaml `
  --output-dir outputs\showcase_feature_opt_v2_compact `
  --thumbnail-index-dir data\processed\qilin_full_feature_opt_v2_compact `
  --max-cases 120
```

自检：

```powershell
& $PY demo\showcase_app.py --data-dir outputs\showcase_feature_opt_v2_compact --smoke
```

启动网页：

```powershell
& $PY demo\showcase_app.py `
  --data-dir outputs\showcase_feature_opt_v2_compact `
  --host 127.0.0.1 `
  --port 7860
```

浏览器打开 `http://127.0.0.1:7860`。

---

## 模型设计摘要

### 多模态 Item 表征

`ItemEncoder` 将 item ID、类目、位置、text（title/content 经 `TextFusionGate` 融合）、image-meta、video-meta、image/video emb、item_dense、ratio、cold_stage、graph 等投影到统一空间，再经 **softmax gate fusion** 得到 item 向量。缺失模态在 gate 中 mask，不参与归一化。

### 用户侧与排序

- `UserInterestEncoder`：Transformer 编码历史序列，可融合 history semantic 摘要。
- `QueryInteractionModule`：query 与候选 item 交互。
- `PLERanker`：8 shared experts × 4 task experts × 3 层，输出三任务 logit。
- `rank_score_head`：与任务概率按 `rank_score_blend=0.2` 融合为 `final_score`。
- 辅助头：like / comment / page_time（`enable_aux_heads=true`）。

### 训练目标

```text
L = 加权 Focal BCE(click/collect/share)
  + bpr_weight * request BPR
  + task_bpr_weight * per-task BPR
  + listwise_weight * listwise
  + aux_* 辅助回归/分类
  + l2_weight * L2
```

当前配置中 `transition` / `contrastive` / `collect&share listwise` 权重均为 0；`enable_intent_heads=false`，不训练意图转移头。

### 选模与指标

- 验证集按 `**quality_score**` 保存 `best.pt`（非单独 WH@20）。
- `quality_score` 由 `ranking_quality_score`（60%）与 `recommendation_quality_score`（40%）组成。

---

## 常用脚本


| 脚本                                          | 作用                                      |
| ------------------------------------------- | --------------------------------------- |
| `scripts/prepare_qilin.py`                  | 原始 Qilin → processed parquet + metadata |
| `scripts/upgrade_feature_opt_v2_columns.py` | 在已有 processed 上追加 v2 特征列                |
| `scripts/build_processed_compact.py`        | **推荐** 文本 + 图片路径视觉 + compact 一步完成       |
| `scripts/build_text_embeddings.py`          | 单独离线文本/查询 embedding                     |
| `scripts/build_item_graph.py`               | 构建 `graph_embedding.npy`                |
| `scripts/compact_processed_features.py`     | 单独剥离 parquet 内嵌高维列                      |
| `scripts/build_visual_embeddings.py`        | 单独从本地图片路径编码 SigLIP/CLIP                 |
| `scripts/attach_visual_sidecars.py`         | （已弃用）从其他 processed 目录复制视觉 sidecar       |
| `scripts/merge_embeddings.py`               | 将 sidecar 合并回 parquet（非 compact 主线）     |
| `scripts/train.py`                          | 训练 / 断点续训                               |
| `scripts/evaluate.py`                       | 单 split 排序指标评估                          |
| `scripts/infer.py`                          | 导出 Top-20 CSV                           |
| `scripts/report_gate_metrics.py`            | Top-20 模态 gate 分组统计                     |
| `scripts/export_showcase_data.py`           | 生成展示台离线数据                               |
| `scripts/check_parameter_budget.py`         | 估算参数量与 checkpoint 体积                    |
| `scripts/audit_reliability.py`              | 数据可靠性审计                                 |


---

## 指标说明


| 指标                               | 含义                                                    |
| -------------------------------- | ----------------------------------------------------- |
| `WeightedHit@20`                 | `0.3·HitClick + 0.4·HitCollect + 0.3·HitShare`（请求级平均） |
| `NDCG@20`                        | 按任务权重加权的排序 NDCG                                       |
| `quality_score`                  | 综合排序与推荐质量（**当前选模指标**）                                 |
| `preference_auc`                 | 请求内加权相关性 pair AUC                                     |
| `request_auc_`* / `request_ap_`* | 分任务请求级 AUC / AP                                       |
| `mean_gate_*`                    | 全候选平均模态 gate                                          |
| `top20_mean_gate_*`              | 各请求 Top-20 内平均 gate，再对请求均值                            |


更完整的指标定义见 `docs/ranking_metrics.md`。

---

## 常见问题

**Q: compact 与全量 `qilin_full_feature_opt_v2` 有何区别？**  
A: 特征语义相同；compact 把 text/query 512 维列移出 parquet，训练时经 mmap sidecar 加载，磁盘与 IO 更省。全量目录可在确认 compact 无误后删除以释放约 17GB。

**Q: 视觉 embedding 从哪来？**  
A: 由 `build_processed_compact.py`（或 `build_visual_embeddings.py`）根据 `notes.image_path` 在 `--image-root` 下解析真实文件并 SigLIP 编码。不要从其他实验目录复制 sidecar。

**Q: 挂载视觉 sidecar 后需要重跑文本 embedding 吗？**  
A: 一体化脚本默认 `--skip-existing`：已有 `text_*.npy` / `query_*.npy` 会跳过。用 `--force-reencode` 可强制全量重跑。

**Q: 为什么 image_emb gate 接近 0？**  
A: 仅 9.7% item 有向量，且 gate 还有 image_meta 等竞争分量；扩大 `build_visual_embeddings` 覆盖后需重新训练。

**Q: `part_30`–`part_50` 磁盘有图为何没用上？**  
A: 路径解析依赖 `notes.image_path`，当前 metadata 只引用 `part_0`–`part_29`。要覆盖更多分片需扩展路径规则或补全 notes 映射。

**Q: 训练很慢 / GPU 利用率低？**  
A: 确认 `fast_loader: true`、`batch_size` 足够大（如 3072）、`num_workers` 在 Windows 上常为 0；评估用 `--fast-loader --batch-size 8192`。

---

## 后续优化方向

1. 扩大 SigLIP 视觉覆盖（`part_30+` 路径映射 + 增量 `build_visual_embeddings`），再训一版 compact 模型。
2. 在视觉覆盖率提升后，观察 `image_emb` gate 与 NDCG 是否同步上升。
3. 对 collect/share 稀疏任务继续做 listwise / 重加权实验（当前 collect&share listwise 权重为 0）。
4. 若 graph gate 长期偏高，可对 graph embedding 去噪或降低 graph 在 gate 中的竞争强度。

