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
NOTE_COLUMNS = ["note_idx", "note_title", "note_content", "note_type", "taxonomy3_id", *NOTE_DENSE_COLUMNS]
USER_COLUMNS = ["user_idx", *USER_DENSE_COLUMNS]


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
    return []


def _as_details(details: object) -> Iterable[dict]:
    return _as_list(details)


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

