from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.build_visual_embeddings import build_embeddings_for_modality, load_visual_index
from scripts.check_parameter_budget import inspect_budget
from scripts.export_showcase_data import _attach_thumbnails, _load_thumbnail_lookup


def _write_minimal_processed(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"raw_item_id": [101, 202], "item_id": [1, 2]}).to_parquet(path / "item_id_map.parquet", index=False)
    base = pd.DataFrame(
        {
            "request_id": [1, 1],
            "item_id": [1, 2],
            "user_id": [1, 1],
            "item_type": [1, 2],
            "taxonomy_id": [1, 1],
            "position": [1, 2],
            "click": [1, 0],
            "collect": [0, 1],
            "share": [0, 0],
            "image_meta_feat_0": [0.1, 0.2],
            "video_meta_feat_0": [0.5, 0.6],
        }
    )
    for split in ["train", "valid", "test"]:
        base.to_parquet(path / f"{split}.parquet", index=False)
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "num_users": 2,
                "num_items": 3,
                "num_item_types": 3,
                "num_taxonomies": 2,
                "max_history": 20,
                "text_dim": 0,
                "image_meta_dim": 1,
                "video_meta_dim": 1,
            }
        ),
        encoding="utf-8",
    )


def test_visual_index_resolves_images(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_minimal_processed(processed)
    qilin_dir = tmp_path / "qilin"
    notes_dir = qilin_dir / "notes"
    notes_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "note_idx": [101, 202],
            "image_path": [["image/part/101.jpg"], ["image/part/202.jpg"]],
            "note_type": [1, 2],
        }
    ).to_parquet(notes_dir / "notes.parquet", index=False)
    image_root = tmp_path / "images"
    (image_root / "part").mkdir(parents=True)
    (image_root / "part" / "101.jpg").write_bytes(b"not a real image")
    (image_root / "part" / "202.jpg").write_bytes(b"not a real image")

    image_index = load_visual_index(processed, qilin_dir, image_root, max_images_per_item=4)

    assert image_index["path_count"].tolist() == [1, 1]
    assert image_index["source"].tolist() == ["image_path", "image_path"]


def test_visual_index_can_filter_qilin_image_parts(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_minimal_processed(processed)
    qilin_dir = tmp_path / "qilin"
    notes_dir = qilin_dir / "notes"
    notes_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "note_idx": [101, 202],
            "image_path": [["image/part_50/101.jpg"], ["image/part_51/202.jpg"]],
            "note_type": [1, 1],
        }
    ).to_parquet(notes_dir / "notes.parquet", index=False)
    image_root = tmp_path / "images"
    (image_root / "part_50").mkdir(parents=True)
    (image_root / "part_51").mkdir(parents=True)
    (image_root / "part_50" / "101.jpg").write_bytes(b"not a real image")
    (image_root / "part_51" / "202.jpg").write_bytes(b"not a real image")

    image_index = load_visual_index(
        processed,
        qilin_dir,
        image_root,
        max_images_per_item=4,
        image_part_min=0,
        image_part_max=50,
    )

    assert image_index["path_count"].tolist() == [1, 0]
    assert "part_50" in image_index.loc[0, "path"]
    assert image_index.loc[1, "status"] == "missing"


def test_mock_visual_embedding_export_writes_all_items_and_summary(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_minimal_processed(processed)
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "101.jpg").write_bytes(b"mock")

    args = argparse.Namespace(
        processed_dir=str(processed),
        qilin_dir=None,
        image_root=str(image_root),
        max_images_per_item=4,
        max_items=0,
        mock_encoder=True,
        mock_dim=16,
        model_name="mock",
        batch_size=8,
        device="cpu",
        output_dim=4,
        compression="random",
        seed=2026,
        save_dtype="float32",
    )
    summary = build_embeddings_for_modality(args, "image")
    values = np.load(processed / "image_embeddings.npy")
    item_ids = np.load(processed / "image_embedding_item_ids.npy")

    assert summary["items"] == 2
    assert summary["encoded_items"] == 1
    assert values.shape == (2, 4)
    assert item_ids.tolist() == [1, 2]
    assert np.linalg.norm(values[0]) > 0
    assert np.linalg.norm(values[1]) == 0


def test_thumbnail_lookup_and_attach_uses_exported_embedding_items(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    image_path = tmp_path / "thumb.jpg"
    image_path.write_bytes(b"fake")
    pd.DataFrame(
        {
            "item_id": [1],
            "path": [str(image_path)],
            "paths_json": [json.dumps([str(image_path)])],
            "status": ["encoded"],
        }
    ).to_parquet(processed / "image_embedding_items.parquet", index=False)
    lookup = _load_thumbnail_lookup(processed)
    rows = _attach_thumbnails(pd.DataFrame({"item_id": [1, 2]}), lookup)

    assert rows.loc[0, "thumbnail_path"] == str(image_path)
    assert rows.loc[0, "thumbnail_source"] == "image"
    assert rows.loc[1, "thumbnail_path"] == ""


def test_parameter_budget_report_for_visual_config_shape(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_minimal_processed(processed)
    config_path = tmp_path / "config.yaml"
    config = {
        "data": {"processed_dir": str(processed), "metadata_file": "metadata.json"},
        "model": {
            "embed_dim": 8,
            "hidden_dim": 16,
            "max_position": 20,
            "transformer_layers": 1,
            "transformer_heads": 2,
            "shared_experts": 2,
            "task_experts": 1,
            "ple_layers": 1,
            "ranker": "ple",
            "use_graph_embedding": True,
            "graph_embedding_trainable": False,
            "enable_intent_heads": False,
        },
        "loss": {"task_weights": {"click": 0.3, "collect": 0.4, "share": 0.3}},
        "evaluation": {"score_weights": {"click": 0.3, "collect": 0.4, "share": 0.3}},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    report = inspect_budget(config_path, budget_mb=800.0)

    assert report["under_budget"] is True
    assert report["metadata_dims"]["image_meta_dim"] == 1
    assert report["metadata_dims"]["video_meta_dim"] == 1
    assert report["total_params"] >= report["trainable_params"]
