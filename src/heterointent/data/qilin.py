from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from heterointent.data.io import write_table
from heterointent.data.dynamic_intent import annotate_dynamic_intents, build_item_intent_lookups
from heterointent.data.preprocess import infer_metadata
from heterointent.utils import write_json

NOTE_DENSE_COLUMNS = [
    "video_duration", "video_height", "video_width", "image_num", "content_length", "commercial_flag",
    "imp_num", "imp_rec_num", "imp_search_num", "click_num", "click_rec_num", "click_search_num",
    "like_num", "collect_num", "comment_num", "share_num", "screenshot_num", "hide_num",
    "rec_like_num", "rec_collect_num", "rec_comment_num", "rec_share_num", "rec_follow_num",
    "search_like_num", "search_collect_num", "search_comment_num", "search_share_num", "search_follow_num",
    "accum_like_num", "accum_collect_num", "accum_comment_num", "view_time", "rec_view_time",
    "search_view_time", "valid_view_times", "full_view_times",
]

USER_DENSE_COLUMNS = [f"dense_feat{i}" for i in range(1, 41)] + ["fans_num", "follows_num"]
NOTE_COLUMNS = ["note_idx", "note_title", "note_content", "note_type", "taxonomy3_id", "image_path", *NOTE_DENSE_COLUMNS]
USER_COLUMNS = ["user_idx", *USER_DENSE_COLUMNS]
IMAGE_PATH_BUCKETS = 8


def _read_parquet_dir(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {path}")
    frames = []
    for file in files:
        if columns is None:
            frames.append(pd.read_parquet(file))
            continue
        schema_cols = pq.ParquetFile(file).schema.names
        selected = [c for c in columns if c in schema_cols]
        frames.append(pd.read_parquet(file, columns=selected))
    return pd.concat(frames, ignore_index=True)


def _stable_bucket(value: object, modulo: int = 1_000_000) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo + 1


def _hash_text(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype="float32")
    if not text:
        return vec
    for token in str(text).split():
        vec[_stable_bucket(token, dim) - 1] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _as_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return []
        if text.startswith("[") and text.endswith("]"):
            return [part.strip().strip("'\"") for part in text[1:-1].split(",") if part.strip()]
        return [text]
    return []


def _as_details(details: object) -> Iterable[dict]:
    return _as_list(details)


def _image_bucket_key(path: str) -> str:
    parts = Path(path).parts
    if len(parts) >= 2:
        return parts[1]
    return path


def _resolution_features(note: pd.DataFrame) -> pd.DataFrame:
    width = note["video_width"].fillna(0.0).clip(lower=0.0).astype("float32") if "video_width" in note else pd.Series(0.0, index=note.index)
    height = note["video_height"].fillna(0.0).clip(lower=0.0).astype("float32") if "video_height" in note else pd.Series(0.0, index=note.index)
    area = width * height
    aspect = width / height.replace(0.0, np.nan)
    aspect = aspect.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(0.0, 10.0)
    return pd.DataFrame(
        {
            "video_width_log": np.log1p(width),
            "video_height_log": np.log1p(height),
            "video_area_log": np.log1p(area),
            "video_aspect": aspect.astype("float32"),
            "video_is_landscape": width.gt(height).astype("float32"),
            "video_is_portrait": height.gt(width).astype("float32"),
            "video_has_resolution": area.gt(0).astype("float32"),
        },
        index=note.index,
    )


def _image_path_lists(note: pd.DataFrame) -> list[list[str]]:
    if "image_path" not in note.columns:
        return [[] for _ in range(len(note))]
    return [[str(path) for path in _as_list(value) if str(path)] for value in note["image_path"].tolist()]


def build_image_meta_features(note: pd.DataFrame) -> pd.DataFrame:
    """Build lightweight image metadata features from Qilin parquet columns."""

    image_num = note["image_num"].fillna(0.0).clip(lower=0.0).astype("float32") if "image_num" in note else pd.Series(0.0, index=note.index)
    path_lists = _image_path_lists(note)
    path_count = pd.Series([len(paths) for paths in path_lists], index=note.index, dtype="float32")
    bucket_values = np.zeros((len(note), IMAGE_PATH_BUCKETS), dtype="float32")
    for row_idx, paths in enumerate(path_lists):
        for path in paths:
            bucket_values[row_idx, _stable_bucket(_image_bucket_key(path), IMAGE_PATH_BUCKETS) - 1] += 1.0
    bucket_sums = bucket_values.sum(axis=1, keepdims=True)
    bucket_values = np.divide(bucket_values, np.maximum(bucket_sums, 1.0), out=np.zeros_like(bucket_values), where=bucket_sums > 0)

    base = pd.DataFrame(
        {
            "image_feat_0": np.log1p(image_num),
            "image_feat_1": np.log1p(path_count),
            "image_feat_2": image_num.gt(0).astype("float32"),
            "image_feat_3": path_count.gt(1).astype("float32"),
            "image_feat_4": (path_count - image_num).abs().astype("float32"),
        },
        index=note.index,
    )
    buckets = pd.DataFrame(bucket_values, columns=[f"image_feat_{5 + i}" for i in range(IMAGE_PATH_BUCKETS)], index=note.index)
    return pd.concat([base, buckets], axis=1).astype("float32")


def build_video_meta_features(note: pd.DataFrame) -> pd.DataFrame:
    """Build lightweight video metadata features from Qilin parquet columns."""

    note_type = note["note_type"].fillna(0.0).astype("float32") if "note_type" in note else pd.Series(0.0, index=note.index)
    duration = note["video_duration"].fillna(0.0).clip(lower=0.0).astype("float32") if "video_duration" in note else pd.Series(0.0, index=note.index)
    image_num = note["image_num"].fillna(0.0).clip(lower=0.0).astype("float32") if "image_num" in note else pd.Series(0.0, index=note.index)
    resolution = _resolution_features(note)
    values = pd.DataFrame(
        {
            "video_feat_0": note_type.eq(2).astype("float32"),
            "video_feat_1": duration.gt(0).astype("float32"),
            "video_feat_2": np.log1p(duration),
            "video_feat_3": np.log1p(image_num),
            "video_feat_4": duration.clip(0.0, 600.0) / 600.0,
            "video_feat_5": resolution["video_width_log"],
            "video_feat_6": resolution["video_height_log"],
            "video_feat_7": resolution["video_area_log"],
            "video_feat_8": resolution["video_aspect"],
            "video_feat_9": resolution["video_is_landscape"],
            "video_feat_10": resolution["video_is_portrait"],
            "video_feat_11": resolution["video_has_resolution"],
        },
        index=note.index,
    )
    return values.astype("float32")


def flatten_recommendation_frame(rec_df: pd.DataFrame, max_history: int = 20) -> pd.DataFrame:
    rows: list[dict] = []
    for row in rec_df.itertuples(index=False):
        request_id = int(getattr(row, "request_idx"))
        session_id = int(getattr(row, "session_idx"))
        raw_user_id = int(getattr(row, "user_idx"))
        recent = _as_list(getattr(row, "recent_clicked_note_idxs", []))[:max_history]
        for detail in _as_details(getattr(row, "rec_result_details_with_idx")):
            if not isinstance(detail, dict):
                continue
            raw_item_id = int(detail.get("note_idx", 0) or 0)
            flat = {
                "request_id": request_id,
                "session_id": session_id,
                "raw_user_id": raw_user_id,
                "raw_item_id": raw_item_id,
                "timestamp": int(float(detail.get("request_timestamp", 0) or 0)),
                "position": int(detail.get("position", 0) or 0),
                "click": int(detail.get("click", 0) or 0),
                "collect": int(detail.get("collect", 0) or 0),
                "share": int(detail.get("share", 0) or 0),
                "next_item_type": 0,
            }
            for i in range(max_history):
                flat[f"hist_raw_item_{i}"] = int(recent[i]) if i < len(recent) else 0
            rows.append(flat)
    return pd.DataFrame(rows)


def _compact_map(values: Iterable[int]) -> dict[int, int]:
    unique = sorted({int(v) for v in values if int(v) > 0})
    return {raw: idx + 1 for idx, raw in enumerate(unique)}


def _apply_compact_ids(df: pd.DataFrame, item_map: dict[int, int], user_map: dict[int, int], max_history: int) -> pd.DataFrame:
    df = df.copy()
    df["user_id"] = df["raw_user_id"].astype(int).map(user_map).fillna(0).astype(int)
    df["item_id"] = df["raw_item_id"].astype(int).map(item_map).fillna(0).astype(int)
    for i in range(max_history):
        raw_col = f"hist_raw_item_{i}"
        col = f"hist_item_{i}"
        df[col] = df[raw_col].astype(int).map(item_map).fillna(0).astype(int)
    return df.drop(columns=[f"hist_raw_item_{i}" for i in range(max_history)])


def build_note_features(notes_df: pd.DataFrame, item_map: dict[int, int], text_hash_dim: int = 0) -> pd.DataFrame:
    note = notes_df[notes_df["note_idx"].astype(int).isin(item_map)].copy()
    note["raw_item_id"] = note["note_idx"].astype(int)
    note["item_id"] = note["raw_item_id"].map(item_map).fillna(0).astype(int)
    note["item_type"] = note["note_type"].fillna(0).astype(int) if "note_type" in note.columns else 0
    note["taxonomy_key"] = note["taxonomy3_id"].fillna("missing").astype(str) if "taxonomy3_id" in note.columns else "missing"

    features = note[["item_id", "raw_item_id", "item_type", "taxonomy_key"]].copy().reset_index(drop=True)
    dense_cols = [c for c in NOTE_DENSE_COLUMNS if c in note.columns]
    if dense_cols:
        dense = note[dense_cols].fillna(0.0).astype("float32").reset_index(drop=True)
        dense.columns = [f"dense_feat_{i}" for i in range(dense.shape[1])]
        features = pd.concat([features, dense], axis=1)

    image_meta = build_image_meta_features(note).reset_index(drop=True)
    video_meta = build_video_meta_features(note).reset_index(drop=True)
    features = pd.concat([features, image_meta, video_meta], axis=1)

    if text_hash_dim > 0:
        title = note["note_title"].fillna("") if "note_title" in note.columns else pd.Series("", index=note.index)
        content = note["note_content"].fillna("") if "note_content" in note.columns else pd.Series("", index=note.index)
        texts = (title.astype(str) + " " + content.astype(str)).to_numpy()
        text_values = np.stack([_hash_text(text, text_hash_dim) for text in texts]) if len(texts) else np.zeros((0, text_hash_dim), dtype="float32")
        text_df = pd.DataFrame(text_values, columns=[f"text_feat_{i}" for i in range(text_hash_dim)])
        features = pd.concat([features, text_df], axis=1)
    return features


def build_user_features(user_df: pd.DataFrame, user_map: dict[int, int], dense_offset: int) -> pd.DataFrame:
    users = user_df[user_df["user_idx"].astype(int).isin(user_map)].copy()
    users["raw_user_id"] = users["user_idx"].astype(int)
    users["user_id"] = users["raw_user_id"].map(user_map).fillna(0).astype(int)
    features = users[["user_id", "raw_user_id"]].copy().reset_index(drop=True)
    dense_cols = [c for c in USER_DENSE_COLUMNS if c in users.columns]
    if dense_cols:
        dense = users[dense_cols].fillna(0.0).astype("float32").reset_index(drop=True)
        dense.columns = [f"dense_feat_{dense_offset + i}" for i in range(dense.shape[1])]
        features = pd.concat([features, dense], axis=1)
    return features


def _save_mappings(output_dir: Path, item_map: dict[int, int], user_map: dict[int, int], taxonomy_map: dict[str, int]) -> None:
    pd.DataFrame({"raw_item_id": list(item_map.keys()), "item_id": list(item_map.values())}).to_parquet(output_dir / "item_id_map.parquet", index=False)
    pd.DataFrame({"raw_user_id": list(user_map.keys()), "user_id": list(user_map.values())}).to_parquet(output_dir / "user_id_map.parquet", index=False)
    pd.DataFrame({"taxonomy_key": list(taxonomy_map.keys()), "taxonomy_id": list(taxonomy_map.values())}).to_parquet(output_dir / "taxonomy_id_map.parquet", index=False)


def convert_qilin_directory(qilin_dir: str | Path, output_dir: str | Path, max_history: int = 20, text_hash_dim: int = 0) -> dict:
    """Convert official THUIR/Qilin parquet folders into compact-ID training tables.

    Raw Qilin files are read-only. Outputs are written under output_dir.
    """
    qilin_dir = Path(qilin_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rec_train = _read_parquet_dir(qilin_dir / "recommendation_train")
    rec_test = _read_parquet_dir(qilin_dir / "recommendation_test")
    train_flat = flatten_recommendation_frame(rec_train, max_history=max_history)
    test_flat = flatten_recommendation_frame(rec_test, max_history=max_history)

    hist_cols = [f"hist_raw_item_{i}" for i in range(max_history)]
    item_values = pd.concat([train_flat["raw_item_id"], test_flat["raw_item_id"], train_flat[hist_cols].stack(), test_flat[hist_cols].stack()])
    user_values = pd.concat([train_flat["raw_user_id"], test_flat["raw_user_id"]])
    item_map = _compact_map(item_values.astype(int).tolist())
    user_map = _compact_map(user_values.astype(int).tolist())

    train_flat = _apply_compact_ids(train_flat, item_map, user_map, max_history=max_history)
    test_flat = _apply_compact_ids(test_flat, item_map, user_map, max_history=max_history)

    notes_raw = _read_parquet_dir(qilin_dir / "notes", columns=NOTE_COLUMNS)
    notes = build_note_features(notes_raw, item_map=item_map, text_hash_dim=text_hash_dim)
    note_dense_dim = len([c for c in notes.columns if c.startswith("dense_feat_")])
    users_raw = _read_parquet_dir(qilin_dir / "user_feat", columns=USER_COLUMNS)
    users = build_user_features(users_raw, user_map=user_map, dense_offset=note_dense_dim)

    def enrich(df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(notes, on=["item_id", "raw_item_id"], how="left")
        df = df.merge(users, on=["user_id", "raw_user_id"], how="left")
        df["item_type"] = df["item_type"].fillna(0).astype(int)
        df["taxonomy_key"] = df["taxonomy_key"].fillna("missing").astype(str)
        feature_cols = [c for c in df.columns if c.startswith("dense_feat_") or c.startswith("text_feat_") or c.startswith("image_feat_") or c.startswith("video_feat_")]
        if feature_cols:
            df[feature_cols] = df[feature_cols].fillna(0.0).astype("float32")
        return df

    train_flat = enrich(train_flat)
    test_flat = enrich(test_flat)
    taxonomy_values = pd.concat([train_flat["taxonomy_key"], test_flat["taxonomy_key"]]).fillna("missing").astype(str)
    taxonomy_map = {value: idx + 1 for idx, value in enumerate(sorted(taxonomy_values.unique()))}
    train_flat["taxonomy_id"] = train_flat["taxonomy_key"].map(taxonomy_map).fillna(0).astype(int)
    test_flat["taxonomy_id"] = test_flat["taxonomy_key"].map(taxonomy_map).fillna(0).astype(int)
    train_flat = train_flat.drop(columns=["taxonomy_key"])
    test_flat = test_flat.drop(columns=["taxonomy_key"])

    notes["taxonomy_id"] = notes["taxonomy_key"].map(taxonomy_map).fillna(0).astype(int)
    item_type_lookup, item_taxonomy_lookup = build_item_intent_lookups(notes[["item_id", "item_type", "taxonomy_id"]])
    train_flat = annotate_dynamic_intents(train_flat, item_type_lookup, item_taxonomy_lookup, max_history=max_history)
    test_flat = annotate_dynamic_intents(test_flat, item_type_lookup, item_taxonomy_lookup, max_history=max_history)

    train_flat = train_flat.sort_values(["timestamp", "request_id", "position"]).reset_index(drop=True)
    split = int(len(train_flat) * 0.9)
    train, valid = train_flat.iloc[:split].reset_index(drop=True), train_flat.iloc[split:].reset_index(drop=True)

    write_table(train, output_dir / "train.parquet")
    write_table(valid, output_dir / "valid.parquet")
    write_table(test_flat, output_dir / "test.parquet")
    metadata = infer_metadata(pd.concat([train, valid, test_flat], ignore_index=True), max_history=max_history)
    write_json(metadata, output_dir / "metadata.json")
    _save_mappings(output_dir, item_map, user_map, taxonomy_map)
    write_json({"num_raw_items": len(item_map), "num_raw_users": len(user_map), "num_taxonomies": len(taxonomy_map), "train_rows": len(train), "valid_rows": len(valid), "test_rows": len(test_flat)}, output_dir / "id_mapping_summary.json")
    return metadata
