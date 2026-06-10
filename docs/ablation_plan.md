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
   - + image/video features
   - + graph embedding

4. Loss:
   - Multi-task BCE only
   - + request-level BPR
   - + transition loss
   - + contrastive loss

5. Deployment:
   - embedding dim 32/64/128
   - PLE expert count 2/4/8
   - CPU latency and checkpoint size
