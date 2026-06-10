from __future__ import annotations

from pathlib import Path

from heterointent.config import load_config
from heterointent.data.synthetic import make_synthetic_dataset
from heterointent.training import train


def test_synthetic_training_smoke() -> None:
    processed = Path("outputs/test_smoke_pytest/processed")
    output = Path("outputs/test_smoke_pytest/run")
    make_synthetic_dataset(
        processed,
        num_users=12,
        num_items=40,
        num_requests=30,
        candidates_per_request=8,
        text_dim=8,
        image_dim=8,
        video_dim=4,
        dense_dim=6,
        seed=7,
    )
    cfg = load_config("configs/smoke.yaml")
    cfg["data"]["processed_dir"] = str(processed)
    cfg["data"]["batch_size"] = 16
    cfg["train"]["output_dir"] = str(output)
    cfg["train"]["epochs"] = 1
    cfg["model"]["embed_dim"] = 16
    cfg["model"]["hidden_dim"] = 32
    cfg["model"]["transformer_heads"] = 2
    result = train(cfg)
    assert Path(result["output_dir"], "best.pt").exists()
    assert Path(result["output_dir"], "metrics.csv").exists()
