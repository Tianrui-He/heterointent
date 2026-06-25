# HeteroIntent Showcase App

本目录提供本机离线推荐可解释展示系统。它读取 `outputs/showcase` 中的轻量数据，不需要现场跑模型或占用 GPU。

## 1. 生成展示数据

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe scripts\export_showcase_data.py --processed-dir data\processed\qilin_full_multimodal_meta --run-dir outputs\qilin_score_opt_mild --output-dir outputs\showcase
```

生成内容：

- `outputs/showcase/overview.json`
- `outputs/showcase/valid_top20.parquet`
- `outputs/showcase/test_top20.parquet`
- `outputs/showcase/showcase_cases.parquet`
- `outputs/showcase/defense_report.md`
- `outputs/showcase/charts/*.png`

## 2. 自检

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe demo\showcase_app.py --data-dir outputs\showcase --smoke
```

看到 `showcase app smoke ok` 表示首页和核心 API 可访问。

## 3. 启动网页

```powershell
D:\adaconda3\envs\MiniOneRec-pre\python.exe demo\showcase_app.py --data-dir outputs\showcase --host 127.0.0.1 --port 7860
```

然后打开：

```text
http://127.0.0.1:7860
```

## 展示建议

1. 先讲 Overview 的样本不均衡和核心指标。
2. 切到 Showcase Cases，选择 `share` 或 `collect` 案例。
3. 在 Request Explorer 中点击 Top-20 item。
4. 用 Why This Rank 展示三任务概率、`rank_score_head`、模态 gate 和用户意图 attention。
