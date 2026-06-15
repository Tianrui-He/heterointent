# Ablation Plan

1. Baseline ranker:
   - Shared-bottom
   - MMoE
   - PLE

2. User intent:
   - No history
   - Mean history pooling
   - SASRec + DIN candidate attention

3. Item representation:
   - ID/category/dense only
   - + text features
   - + image/video metadata features
   - + image/video features
   - + graph embedding

4. Loss:
   - Multi-task BCE only
   - Balanced focal BCE with collect/share positive weights
   - + request-level BPR
   - + per-task request-level BPR
   - + transition loss
   - + contrastive loss
   - collect/share positive weight grid: 2/4/6/8
   - BPR weight grid: 0/0.01/0.03/0.05

5. Deployment:
   - embedding dim 32/64/128
   - PLE expert count 2/4/8
   - CPU latency and checkpoint size
