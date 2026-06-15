from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from heterointent.data.dynamic_intent import annotate_dynamic_intents, build_item_intent_lookups
from heterointent.data.io import write_table
from heterointent.utils import write_json


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_synthetic_dataset(
    output_dir: str | Path,
    num_users: int = 120,
    num_items: int = 600,
    num_requests: int = 900,
    candidates_per_request: int = 30,
    num_item_types: int = 8,
    num_taxonomies: int = 32,
    text_dim: int = 16,
    image_dim: int = 16,
    video_dim: int = 8,
    dense_dim: int = 12,
    max_history: int = 20,
    seed: int = 2026,
) -> dict:
    """Create a small Qilin-like processed dataset for smoke tests."""

    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    user_pref = rng.normal(size=(num_users + 1, num_item_types))
    item_type = rng.integers(1, num_item_types + 1, size=num_items + 1)
    taxonomy = rng.integers(1, num_taxonomies + 1, size=num_items + 1)
    text_feat = rng.normal(size=(num_items + 1, text_dim)).astype("float32")
    image_feat = rng.normal(size=(num_items + 1, image_dim)).astype("float32")
    video_feat = rng.normal(size=(num_items + 1, video_dim)).astype("float32")
    dense_feat = rng.normal(size=(num_items + 1, dense_dim)).astype("float32")

    rows: list[dict] = []
    histories = {u: list(rng.integers(1, num_items + 1, size=max_history)) for u in range(1, num_users + 1)}

    for request_id in range(1, num_requests + 1):
        user_id = int(rng.integers(1, num_users + 1))
        session_id = int(user_id * 1000 + request_id // 4)
        timestamp = request_id
        intent_shift = rng.normal(scale=0.7, size=num_item_types)
        pref = user_pref[user_id] + intent_shift
        candidates = rng.choice(np.arange(1, num_items + 1), size=candidates_per_request, replace=False)
        hist = histories[user_id][-max_history:]
        hist_types = item_type[np.array(hist)]
        next_type = int(np.bincount(hist_types, minlength=num_item_types + 1)[1:].argmax() + 1)

        for pos, item_id in enumerate(candidates, start=1):
            t = int(item_type[item_id])
            type_affinity = pref[t - 1]
            content_affinity = 0.2 * text_feat[item_id, : min(8, text_dim)].sum()
            position_bias = -0.03 * pos
            click_p = _sigmoid(np.array(type_affinity + content_affinity + position_bias - 0.7))
            collect_p = _sigmoid(np.array(type_affinity + 0.4 * dense_feat[item_id, 0] - 1.6))
            share_p = _sigmoid(np.array(type_affinity + 0.3 * image_feat[item_id, 0] - 1.8))
            click = int(rng.random() < click_p)
            collect = int(click and rng.random() < collect_p)
            share = int(click and rng.random() < share_p)

            row = {
                "request_id": request_id,
                "session_id": session_id,
                "user_id": user_id,
                "item_id": int(item_id),
                "item_type": t,
                "taxonomy_id": int(taxonomy[item_id]),
                "timestamp": timestamp,
                "position": pos,
                "click": click,
                "collect": collect,
                "share": share,
                "next_item_type": next_type,
            }
            for i, h in enumerate(hist):
                row[f"hist_item_{i}"] = int(h)
            for i, value in enumerate(text_feat[item_id]):
                row[f"text_feat_{i}"] = float(value)
            for i, value in enumerate(image_feat[item_id]):
                row[f"image_feat_{i}"] = float(value)
            for i, value in enumerate(video_feat[item_id]):
                row[f"video_feat_{i}"] = float(value)
            for i, value in enumerate(dense_feat[item_id]):
                row[f"dense_feat_{i}"] = float(value)
            rows.append(row)

        positives = [int(i) for i in candidates if rng.random() < 0.08 or item_type[i] == next_type]
        histories[user_id].extend(positives[:3] or [int(candidates[0])])

    df = pd.DataFrame(rows).sort_values(["timestamp", "request_id", "position"]).reset_index(drop=True)
    item_features = pd.DataFrame(
        {
            "item_id": np.arange(num_items + 1),
            "item_type": item_type,
            "taxonomy_id": taxonomy,
        }
    )
    type_lookup, taxonomy_lookup = build_item_intent_lookups(item_features)
    df = annotate_dynamic_intents(df, type_lookup, taxonomy_lookup, max_history=max_history)

    n = len(df)
    train = df.iloc[: int(n * 0.8)]
    valid = df.iloc[int(n * 0.8) : int(n * 0.9)]
    test = df.iloc[int(n * 0.9) :]

    write_table(train, output_dir / "train.parquet")
    write_table(valid, output_dir / "valid.parquet")
    write_table(test, output_dir / "test.parquet")

    metadata = {
        "num_users": num_users + 1,
        "num_items": num_items + 1,
        "num_item_types": num_item_types + 1,
        "num_taxonomies": num_taxonomies + 1,
        "text_dim": text_dim,
        "image_dim": image_dim,
        "video_dim": video_dim,
        "dense_dim": dense_dim,
        "max_history": max_history,
        "dynamic_intent": True,
    }
    write_json(metadata, output_dir / "metadata.json")
    return metadata
