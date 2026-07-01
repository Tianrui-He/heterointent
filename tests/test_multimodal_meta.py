from __future__ import annotations

import pandas as pd
import torch

from heterointent.data.qilin import build_note_features
from heterointent.data.preprocess import infer_metadata
from heterointent.models.encoders import ItemEncoder


def test_build_note_features_adds_image_and_video_meta_features() -> None:
    notes = pd.DataFrame(
        [
            {
                "note_idx": 101,
                "note_title": "title",
                "note_content": "content",
                "note_type": 1,
                "taxonomy3_id": "cat_a",
                "image_num": 2,
                "image_path": ["image/part_001/1.jpg", "image/part_002/2.jpg"],
                "video_duration": 0,
                "video_height": 0,
                "video_width": 0,
            },
            {
                "note_idx": 202,
                "note_title": "video",
                "note_content": "content",
                "note_type": 2,
                "taxonomy3_id": "cat_b",
                "image_num": 0,
                "image_path": [],
                "video_duration": 36,
                "video_height": 720,
                "video_width": 1280,
            },
        ]
    )

    features = build_note_features(notes, item_map={101: 1, 202: 2}, text_hash_dim=0)
    image_cols = [col for col in features.columns if col.startswith("image_meta_feat_")]
    video_cols = [col for col in features.columns if col.startswith("video_meta_feat_")]

    assert len(image_cols) == 13
    assert len(video_cols) == 12
    assert features.loc[0, "image_meta_feat_2"] == 1.0
    assert features.loc[0, image_cols].sum() > 0
    assert features.loc[1, "video_meta_feat_0"] == 1.0
    assert features.loc[1, "video_meta_feat_1"] == 1.0
    assert features.loc[1, video_cols].sum() > 0

    metadata = infer_metadata(features, max_history=20)
    assert metadata["image_meta_dim"] == 13
    assert metadata["video_meta_dim"] == 12


def test_item_encoder_masks_absent_modalities_in_gate() -> None:
    metadata = {
        "num_items": 4,
        "num_item_types": 3,
        "num_taxonomies": 5,
        "image_emb_dim": 2,
        "video_meta_dim": 2,
        "text_dim": 0,
    }
    encoder = ItemEncoder(metadata=metadata, embed_dim=8, max_position=20, dropout=0.0, use_graph_embedding=False)
    batch = {
        "item_id": torch.tensor([1, 2]),
        "item_type": torch.tensor([1, 2]),
        "taxonomy_id": torch.tensor([1, 2]),
        "position": torch.tensor([1, 2]),
        "has_image_emb": torch.tensor([0, 1]),
        "image_emb_feat": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        "video_meta_feat": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
    }

    _, extras = encoder(batch)
    image_idx = encoder.part_names.index("image_emb")
    video_idx = encoder.part_names.index("video_meta")

    assert extras["modality_gate_mask"][0, image_idx] == 0.0
    assert extras["modality_gate"][0, image_idx] < 1e-6
    assert extras["modality_gate_mask"][1, video_idx] == 0.0
    assert extras["modality_gate"][1, video_idx] < 1e-6


def test_item_encoder_forward_without_image_or_video_features() -> None:
    metadata = {
        "num_items": 4,
        "num_item_types": 3,
        "num_taxonomies": 5,
        "text_dim": 0,
    }
    encoder = ItemEncoder(metadata=metadata, embed_dim=8, max_position=20, dropout=0.0, use_graph_embedding=False)
    batch = {
        "item_id": torch.tensor([1, 2]),
        "item_type": torch.tensor([1, 2]),
        "taxonomy_id": torch.tensor([1, 2]),
        "position": torch.tensor([1, 2]),
    }

    fused, extras = encoder(batch)

    assert fused.shape == (2, 8)
    assert extras["modality_gate"].shape == (2, 4)
