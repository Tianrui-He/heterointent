# Technical Report Template

## 1. Task Understanding

Describe the double-waterfall recommendation setting, heterogeneous cards,
dynamic user intent and the Click/Collect/Share objective.

## 2. Data

- Primary public dataset: Qilin.
- Main table: Rec requests with exposed candidates and feedback labels.
- Auxiliary context: Search sessions, user features, note features and 20 recent clicked note IDs.
- Split: chronological 80/10/10, no future behavior in history features.

## 3. Model

HeteroIntent-PLE contains:

- Item Encoder: item/domain/taxonomy/statistical/text/image/video/graph features.
- User Encoder: SASRec-style session encoder + DIN-style candidate attention.
- Cross-domain Alignment: gated multimodal fusion + denoised co-occurrence item graph.
- Multi-task Ranker: PLE with click, collect and share towers.

When raw image/video files are unavailable, image and video modalities are
represented by structured metadata from Qilin parquet fields, such as image
count, image path buckets, video duration and resolution. Missing modality
masks prevent all-zero modality features from receiving fusion gate mass.

## 4. Objective

```text
L = 0.3 * BalancedFocalBCE_click
  + 0.4 * BalancedFocalBCE_collect
  + 0.3 * BalancedFocalBCE_share
  + lambda_rank * BPR_weighted_score
  + lambda_task_rank * BPR_click_collect_share
  + lambda_intent * L_transition
  + lambda_contrast * L_contrastive
```

The balanced focal terms use stronger positive weights for sparse high-value
behaviors such as collect and share. `BPR_weighted_score` aligns the final
0.3/0.4/0.3 score with request-level ranking, while `BPR_click_collect_share`
keeps each task tower independently rank-aware. The implementation uses bounded
sampled BPR to preserve request-level ranking signal without exhaustive pair
enumeration.

Final ranking:

```text
score = 0.3 * p_click + 0.4 * p_collect + 0.3 * p_share
```

## 5. Experiments

Report:

- DeepFM/DCN V2/Shared-bottom baseline.
- DIN/SASRec/DSIN user sequence ablation.
- Shared-bottom vs MMoE vs PLE.
- No multimodal vs gated multimodal vs graph-enhanced.
- Model size and CPU/GPU latency.

## 6. Explainability

Show:

- History attention over recently clicked items.
- Modality gate weights.
- PLE gate weights per task.
- Per-task scores and final weighted score.
- Case studies for cross-category intent transfer.
