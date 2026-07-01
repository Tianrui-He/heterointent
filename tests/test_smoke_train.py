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
        image_emb_dim=8,
        item_dense_dim=6,
        ratio_dim=4,
        seed=7,
    )
    cfg = load_config("configs/qilin_feature_opt_v2_history_compact.yaml")
    cfg["device"] = "cpu"
    cfg["data"]["processed_dir"] = str(processed)
    cfg["data"]["batch_size"] = 16
    cfg["data"]["pin_memory"] = False
    cfg["data"]["fast_loader"] = True
    cfg["train"]["output_dir"] = str(output)
    cfg["train"]["epochs"] = 1
    cfg["train"]["amp"] = False
    cfg["model"]["embed_dim"] = 16
    cfg["model"]["hidden_dim"] = 32
    cfg["model"]["transformer_heads"] = 2
    cfg["model"]["use_graph_embedding"] = False
    cfg["loss"]["type_transition_weight"] = 0.03
    cfg["loss"]["taxonomy_transition_weight"] = 0.03
    result = train(cfg)
    assert Path(result["output_dir"], "best.pt").exists()
    assert Path(result["output_dir"], "metrics.csv").exists()


def test_score_optimized_training_smoke() -> None:
    processed = Path("outputs/test_score_opt_smoke_pytest/processed")
    output = Path("outputs/test_score_opt_smoke_pytest/run")
    make_synthetic_dataset(
        processed,
        num_users=12,
        num_items=40,
        num_requests=30,
        candidates_per_request=8,
        text_dim=8,
        image_emb_dim=8,
        item_dense_dim=6,
        ratio_dim=4,
        seed=11,
    )
    cfg = load_config("configs/qilin_feature_opt_v2_history_compact.yaml")
    cfg["device"] = "cpu"
    cfg["data"]["processed_dir"] = str(processed)
    cfg["data"]["batch_size"] = 24
    cfg["data"]["pin_memory"] = False
    cfg["data"]["fast_loader"] = True
    cfg["data"]["request_preserving_train"] = True
    cfg["train"]["output_dir"] = str(output)
    cfg["train"]["epochs"] = 1
    cfg["train"]["amp"] = False
    cfg["model"]["embed_dim"] = 16
    cfg["model"]["hidden_dim"] = 32
    cfg["model"]["transformer_heads"] = 2
    cfg["model"]["use_graph_embedding"] = False
    cfg["model"]["enable_intent_heads"] = False

    result = train(cfg)

    assert Path(result["output_dir"], "best.pt").exists()
    assert Path(result["output_dir"], "metrics.csv").exists()
