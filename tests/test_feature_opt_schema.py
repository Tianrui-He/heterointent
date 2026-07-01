from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from heterointent.data.dataset import RankingDataset
from heterointent.data.qilin import convert_qilin_directory
from scripts.compact_processed_features import compact_processed_dir


def _write_minimal_qilin_raw(root: Path) -> None:
    (root / "recommendation_train").mkdir(parents=True)
    (root / "recommendation_test").mkdir(parents=True)
    (root / "notes").mkdir(parents=True)
    (root / "user_feat").mkdir(parents=True)

    rec = pd.DataFrame(
        {
            "request_idx": [10, 11],
            "session_idx": [100, 100],
            "user_idx": [1, 2],
            "query": ["long request context one", "long request context two"],
            "recent_clicked_note_idxs": [[101, 102], [102, 103]],
            "rec_result_details_with_idx": [
                [
                    {
                        "note_idx": 101,
                        "click": 1,
                        "collect": 0,
                        "share": 0,
                        "like": 1,
                        "comment": 0,
                        "page_time": 12.0,
                        "position": 1,
                        "request_timestamp": 1000.0,
                    },
                    {
                        "note_idx": 102,
                        "click": 0,
                        "collect": 0,
                        "share": 0,
                        "like": 0,
                        "comment": 0,
                        "page_time": 0.0,
                        "position": 2,
                        "request_timestamp": 1000.0,
                    },
                ],
                [
                    {
                        "note_idx": 103,
                        "click": 1,
                        "collect": 1,
                        "share": 0,
                        "like": 0,
                        "comment": 1,
                        "page_time": 30.0,
                        "position": 1,
                        "request_timestamp": 1010.0,
                    }
                ],
            ],
        }
    )
    rec.to_parquet(root / "recommendation_train" / "part.parquet", index=False)
    rec.iloc[[0]].assign(request_idx=[20]).to_parquet(root / "recommendation_test" / "part.parquet", index=False)

    notes = pd.DataFrame(
        {
            "note_idx": [101, 102, 103],
            "note_title": ["t1", "t2", "t3"],
            "note_content": ["c1", "c2", "c3"],
            "note_type": [1, 2, 1],
            "taxonomy1_id": ["tax1_a", "tax1_a", "tax1_b"],
            "taxonomy2_id": ["tax2_a", "tax2_b", "tax2_b"],
            "taxonomy3_id": ["tax3_a", "nan", "tax3_c"],
            "image_path": [["image/a.jpg"], [], ["image/c.jpg"]],
            "image_num": [1.0, 0.0, 1.0],
            "video_duration": [0.0, 20.0, 0.0],
            "video_height": [0.0, 720.0, 0.0],
            "video_width": [0.0, 1280.0, 0.0],
            "imp_num": [100.0, 5.0, 1000.0],
            "imp_rec_num": [80.0, 3.0, 600.0],
            "imp_search_num": [20.0, 2.0, 400.0],
            "click_num": [20.0, 1.0, 100.0],
            "click_rec_num": [18.0, 1.0, 70.0],
            "click_search_num": [2.0, 0.0, 30.0],
            "like_num": [3.0, 0.0, 10.0],
            "collect_num": [1.0, 0.0, 5.0],
            "comment_num": [0.0, 0.0, 2.0],
            "share_num": [1.0, 0.0, 4.0],
            "view_time": [120.0, 0.0, 1000.0],
            "rec_view_time": [100.0, 0.0, 600.0],
            "search_view_time": [20.0, 0.0, 400.0],
            "valid_view_times": [10.0, 0.0, 50.0],
            "full_view_times": [2.0, 0.0, 20.0],
        }
    )
    notes.to_parquet(root / "notes" / "part.parquet", index=False)

    users = pd.DataFrame(
        {
            "user_idx": [1, 2],
            "gender": ["female", "male"],
            "platform": ["iOS", "Android"],
            "age": ["19-22", "23-25"],
            "location": ["", "中国  广东  深圳"],
            "dense_feat1": [0.1, 0.2],
            "fans_num": [3.0, 4.0],
            "follows_num": [5.0, 6.0],
        }
    )
    users.to_parquet(root / "user_feat" / "part.parquet", index=False)


def test_qilin_conversion_emits_feature_opt_schema(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    _write_minimal_qilin_raw(raw)

    metadata = convert_qilin_directory(raw, processed, max_history=3, text_hash_dim=0)
    train = pd.read_parquet(processed / "train.parquet")
    valid = pd.read_parquet(processed / "valid.parquet")
    test = pd.read_parquet(processed / "test.parquet")
    disk_metadata = json.loads((processed / "metadata.json").read_text(encoding="utf-8"))

    required = {
        "query",
        "like",
        "comment",
        "page_time_log",
        "gender_id",
        "platform_id",
        "age_id",
        "location_id",
        "taxonomy1_id",
        "taxonomy2_id",
        "hist_taxonomy1_id_0",
        "hist_taxonomy2_id_0",
        "item_dense_feat_0",
        "user_dense_feat_0",
        "ratio_feat_0",
        "cross_feat_0",
        "image_meta_feat_0",
        "video_meta_feat_0",
        "text_stat_feat_0",
        "cold_stage_id",
        "has_query",
        "has_image_emb",
        "history_text_feat_0",
        "history_ratio_feat_0",
    }
    assert required.issubset(train.columns)
    assert metadata == disk_metadata
    assert metadata["num_taxonomy1"] > 1
    assert metadata["num_taxonomy2"] > 1
    assert metadata["image_meta_dim"] > 0
    assert metadata["video_meta_dim"] > 0
    assert metadata["item_dense_dim"] > 0
    assert metadata["user_dense_dim"] > 0
    assert metadata["ratio_dim"] > 0
    assert metadata["cross_dim"] > 0
    assert metadata["text_stat_dim"] == 3
    assert metadata["history_text_dim"] == 64
    assert metadata["history_ratio_dim"] == 16
    assert metadata["num_cold_stages"] == 5

    feature_cols = [
        c
        for c in train.columns
        if c.startswith(("item_dense_feat_", "user_dense_feat_", "ratio_feat_", "cross_feat_"))
    ]
    values = pd.concat([train[feature_cols], valid[feature_cols], test[feature_cols]], ignore_index=True).to_numpy()
    assert np.isfinite(values).all()
    assert not any(col.startswith(("like_feat_", "comment_feat_", "page_time_log_feat_")) for col in train.columns)
    assert (processed / "feature_standardization.json").exists()


def test_compact_processed_features_uses_sidecar_lookup(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    compact = tmp_path / "compact"
    processed.mkdir()
    base = pd.DataFrame(
        {
            "request_id": [10, 10, 11],
            "item_id": [1, 2, 2],
            "user_id": [1, 1, 1],
            "item_type": [1, 2, 2],
            "taxonomy_id": [1, 1, 1],
            "position": [1, 2, 1],
            "click": [1, 0, 0],
            "collect": [0, 1, 0],
            "share": [0, 0, 1],
            "text_feat_0": [0.1, 0.3, 0.3],
            "text_feat_1": [0.2, 0.4, 0.4],
            "query_feat_0": [1.0, 1.0, 4.0],
            "query_feat_1": [2.0, 2.0, 5.0],
            "query_feat_2": [3.0, 3.0, 6.0],
        }
    )
    for split in ("train", "valid", "test"):
        base.to_parquet(processed / f"{split}.parquet", index=False)
    metadata = {
        "num_users": 2,
        "num_items": 3,
        "num_item_types": 3,
        "num_taxonomies": 2,
        "max_history": 2,
        "text_dim": 2,
        "query_dim": 3,
    }
    (processed / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.save(processed / "text_embeddings.npy", np.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32"))
    np.save(processed / "text_embedding_item_ids.npy", np.array([1, 2], dtype="int64"))
    np.save(processed / "query_embeddings.npy", np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"))
    np.save(processed / "query_embedding_request_ids.npy", np.array([10, 11], dtype="int64"))

    summary = compact_processed_dir(processed, compact, groups=["text", "query"])
    compact_train = pd.read_parquet(compact / "train.parquet")
    compact_metadata = json.loads((compact / "metadata.json").read_text(encoding="utf-8"))
    dataset = RankingDataset(compact / "train.parquet", compact_metadata)

    assert summary["splits"]["train"]["dropped_columns"] == 5
    assert "text_feat_0" not in compact_train.columns
    assert "query_feat_0" not in compact_train.columns
    assert np.allclose(dataset.tensors["text_feat"].numpy(), base[["text_feat_0", "text_feat_1"]].to_numpy())
    assert np.allclose(
        dataset.tensors["query_feat"].numpy(),
        base[["query_feat_0", "query_feat_1", "query_feat_2"]].to_numpy(),
    )


def test_build_processed_compact_mock_visual(tmp_path: Path) -> None:
    from scripts.build_processed_compact import build_processed_compact

    processed = tmp_path / "processed"
    compact = tmp_path / "compact"
    qilin_dir = tmp_path / "qilin"
    notes_dir = qilin_dir / "notes"
    notes_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "note_idx": [101, 202],
            "image_path": [["image/part/101.jpg"], ["image/part/202.jpg"]],
            "note_type": [1, 2],
            "note_title": ["a", "b"],
            "note_content": ["c", "d"],
        }
    ).to_parquet(notes_dir / "notes.parquet", index=False)
    image_root = tmp_path / "images"
    (image_root / "part").mkdir(parents=True)
    (image_root / "part" / "101.jpg").write_bytes(b"x")
    (image_root / "part" / "202.jpg").write_bytes(b"y")

    processed.mkdir()
    base = pd.DataFrame(
        {
            "request_id": [10, 10],
            "item_id": [1, 2],
            "user_id": [1, 1],
            "item_type": [1, 2],
            "taxonomy_id": [1, 1],
            "position": [1, 2],
            "click": [1, 0],
            "collect": [0, 1],
            "share": [0, 0],
            "query": ["q1", "q1"],
            "text_feat_0": [0.1, 0.3],
            "text_feat_1": [0.2, 0.4],
            "query_feat_0": [1.0, 1.0],
            "image_emb_feat_0": [0.5, 0.6],
        }
    )
    for split in ("train", "valid", "test"):
        base.to_parquet(processed / f"{split}.parquet", index=False)
    pd.DataFrame({"raw_item_id": [101, 202], "item_id": [1, 2]}).to_parquet(processed / "item_id_map.parquet", index=False)
    (processed / "metadata.json").write_text(
        json.dumps(
            {
                "num_users": 2,
                "num_items": 3,
                "num_item_types": 3,
                "num_taxonomies": 2,
                "max_history": 2,
                "text_dim": 2,
                "query_dim": 1,
                "image_emb_dim": 1,
            }
        ),
        encoding="utf-8",
    )

    summary = build_processed_compact(
        processed,
        compact,
        qilin_dir=qilin_dir,
        groups=["text", "query", "image_emb"],
        build_text=False,
        build_visual=True,
        image_root=image_root,
        visual_mock_encoder=True,
        visual_output_dim=4,
        skip_existing_embeddings=False,
    )

    metadata = json.loads((compact / "metadata.json").read_text(encoding="utf-8"))
    compact_train = pd.read_parquet(compact / "train.parquet")
    assert summary["compact"]["compacted_groups"]
    assert metadata["image_emb_dim"] == 4
    assert "image_emb_feat_0" not in compact_train.columns
    assert (compact / "image_embeddings.npy").exists()
    assert (compact / "pipeline_summary.json").exists()
