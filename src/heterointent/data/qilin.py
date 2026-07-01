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
from heterointent.data.schema import (
    FEATURE_PREFIXES,
    NUM_COLD_STAGES,
    history_columns,
    history_taxonomy1_columns,
    history_taxonomy2_columns,
    history_taxonomy_columns,
    history_type_columns,
)
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
USER_CATEGORICAL_COLUMNS = ["gender", "platform", "age", "location"]
NOTE_COLUMNS = [
    "note_idx",
    "note_title",
    "note_content",
    "note_type",
    "taxonomy1_id",
    "taxonomy2_id",
    "taxonomy3_id",
    "image_path",
    *NOTE_DENSE_COLUMNS,
]
USER_COLUMNS = ["user_idx", *USER_CATEGORICAL_COLUMNS, *USER_DENSE_COLUMNS]
IMAGE_PATH_BUCKETS = 8
LOCATION_MIN_COUNT = 10
STANDARDIZED_PREFIXES = ("item_dense_feat_", "user_dense_feat_", "ratio_feat_")
HISTORY_TEXT_SUMMARY_DIM = 64
HISTORY_RATIO_SUMMARY_DIM = 16
TEXT_STAT_DIM = 3


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


def _feature_name(prefix: str, idx: int) -> str:
    return f"{prefix}{idx}"


def _feature_columns(columns: Iterable[str], prefix: str) -> list[str]:
    def suffix_value(name: str) -> int:
        try:
            return int(name.rsplit("_", 1)[-1])
        except ValueError:
            return -1

    return sorted([c for c in columns if c.startswith(prefix)], key=suffix_value)


def _clean_key(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text in {"", "nan", "None", "missing"} else text


def _clean_key_series(values: pd.Series) -> pd.Series:
    return values.map(_clean_key).astype(str)


def _compact_key_map(values: Iterable[object]) -> dict[str, int]:
    unique = sorted({_clean_key(v) for v in values if _clean_key(v)})
    return {raw: idx + 1 for idx, raw in enumerate(unique)}


def _compact_user_category_map(values: Iterable[object], min_count: int = 1, low_freq_unk: bool = False) -> dict[str, int]:
    cleaned = pd.Series([_clean_key(v) for v in values])
    counts = cleaned[cleaned.ne("")].value_counts()
    mapping: dict[str, int] = {}
    next_id = 2 if low_freq_unk else 1
    for key in sorted(counts.index):
        if int(counts[key]) < min_count:
            if low_freq_unk:
                mapping[key] = 1
            continue
        mapping[key] = next_id
        next_id += 1
    return mapping


def _apply_key_map(values: pd.Series, mapping: dict[str, int]) -> pd.Series:
    return _clean_key_series(values).map(mapping).fillna(0).astype(int)


def _safe_float(note: pd.DataFrame, column: str) -> pd.Series:
    if column not in note.columns:
        return pd.Series(0.0, index=note.index, dtype="float32")
    return note[column].fillna(0.0).astype("float32")


def _smooth_rate(num: pd.Series, den: pd.Series, alpha: float = 1.0, beta: float = 10.0) -> pd.Series:
    return ((num.clip(lower=0.0) + alpha) / (den.clip(lower=0.0) + beta)).replace([np.inf, -np.inf], 0.0).fillna(0.0)


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


def build_image_meta_features(note: pd.DataFrame, prefix: str = "image_meta_feat_") -> pd.DataFrame:
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
            _feature_name(prefix, 0): np.log1p(image_num),
            _feature_name(prefix, 1): np.log1p(path_count),
            _feature_name(prefix, 2): image_num.gt(0).astype("float32"),
            _feature_name(prefix, 3): path_count.gt(1).astype("float32"),
            _feature_name(prefix, 4): (path_count - image_num).abs().astype("float32"),
        },
        index=note.index,
    )
    buckets = pd.DataFrame(bucket_values, columns=[_feature_name(prefix, 5 + i) for i in range(IMAGE_PATH_BUCKETS)], index=note.index)
    return pd.concat([base, buckets], axis=1).astype("float32")


def build_video_meta_features(note: pd.DataFrame, prefix: str = "video_meta_feat_") -> pd.DataFrame:
    """Build lightweight video metadata features from Qilin parquet columns."""

    note_type = note["note_type"].fillna(0.0).astype("float32") if "note_type" in note else pd.Series(0.0, index=note.index)
    duration = note["video_duration"].fillna(0.0).clip(lower=0.0).astype("float32") if "video_duration" in note else pd.Series(0.0, index=note.index)
    image_num = note["image_num"].fillna(0.0).clip(lower=0.0).astype("float32") if "image_num" in note else pd.Series(0.0, index=note.index)
    resolution = _resolution_features(note)
    values = pd.DataFrame(
        {
            _feature_name(prefix, 0): note_type.eq(2).astype("float32"),
            _feature_name(prefix, 1): duration.gt(0).astype("float32"),
            _feature_name(prefix, 2): np.log1p(duration),
            _feature_name(prefix, 3): np.log1p(image_num),
            _feature_name(prefix, 4): duration.clip(0.0, 600.0) / 600.0,
            _feature_name(prefix, 5): resolution["video_width_log"],
            _feature_name(prefix, 6): resolution["video_height_log"],
            _feature_name(prefix, 7): resolution["video_area_log"],
            _feature_name(prefix, 8): resolution["video_aspect"],
            _feature_name(prefix, 9): resolution["video_is_landscape"],
            _feature_name(prefix, 10): resolution["video_is_portrait"],
            _feature_name(prefix, 11): resolution["video_has_resolution"],
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
        query = str(getattr(row, "query", "") or "")
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
                "query": query,
                "timestamp": int(float(detail.get("request_timestamp", 0) or 0)),
                "position": int(detail.get("position", 0) or 0),
                "click": int(detail.get("click", 0) or 0),
                "collect": int(detail.get("collect", 0) or 0),
                "share": int(detail.get("share", 0) or 0),
                "like": int(detail.get("like", 0) or 0),
                "comment": int(detail.get("comment", 0) or 0),
                "page_time_log": float(np.log1p(max(float(detail.get("page_time", 0.0) or 0.0), 0.0))),
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


def build_item_dense_features(note: pd.DataFrame) -> pd.DataFrame:
    dense_cols = [c for c in NOTE_DENSE_COLUMNS if c in note.columns]
    if not dense_cols:
        return pd.DataFrame(index=note.index)
    values = note[dense_cols].fillna(0.0).astype("float32").clip(lower=0.0)
    values = np.log1p(values.clip(upper=1_000_000.0))
    values.columns = [f"item_dense_feat_{i}" for i in range(values.shape[1])]
    return values.astype("float32")


def build_ratio_features(note: pd.DataFrame) -> pd.DataFrame:
    imp = _safe_float(note, "imp_num")
    click = _safe_float(note, "click_num")
    like = _safe_float(note, "like_num")
    collect = _safe_float(note, "collect_num")
    comment = _safe_float(note, "comment_num")
    share = _safe_float(note, "share_num")
    rec_imp = _safe_float(note, "imp_rec_num")
    search_imp = _safe_float(note, "imp_search_num")
    rec_click = _safe_float(note, "click_rec_num")
    search_click = _safe_float(note, "click_search_num")
    view_time = _safe_float(note, "view_time")
    rec_view_time = _safe_float(note, "rec_view_time")
    search_view_time = _safe_float(note, "search_view_time")
    valid_view_times = _safe_float(note, "valid_view_times")
    full_view_times = _safe_float(note, "full_view_times")

    imp_bucket = pd.cut(
        imp,
        bins=[-np.inf, 9.0, 99.0, 999.0, np.inf],
        labels=[0, 1, 2, 3],
    ).astype("int8")
    click_bucket = pd.cut(
        click,
        bins=[-np.inf, 0.0, 9.0, 99.0, np.inf],
        labels=[0, 1, 2, 3],
    ).astype("int8")
    popularity = np.log1p(imp) + 0.5 * np.log1p(click) + np.log1p(collect + share)
    popularity_bucket = (
        pd.qcut(popularity.rank(method="first"), q=4, labels=[0, 1, 2, 3]).astype("int8")
        if len(note) >= 4
        else pd.Series(0, index=note.index, dtype="int8")
    )

    raw = pd.DataFrame(
        {
            "ctr_smooth": _smooth_rate(click, imp),
            "like_per_click_smooth": _smooth_rate(like, click),
            "collect_per_click_smooth": _smooth_rate(collect, click),
            "comment_per_click_smooth": _smooth_rate(comment, click),
            "share_per_click_smooth": _smooth_rate(share, click),
            "rec_imp_ratio": _smooth_rate(rec_imp, rec_imp + search_imp, alpha=0.5, beta=1.0),
            "search_imp_ratio": _smooth_rate(search_imp, rec_imp + search_imp, alpha=0.5, beta=1.0),
            "rec_ctr_smooth": _smooth_rate(rec_click, rec_imp),
            "search_ctr_smooth": _smooth_rate(search_click, search_imp),
            "view_time_per_imp": _smooth_rate(view_time, imp, alpha=0.0, beta=10.0),
            "view_time_per_click": _smooth_rate(view_time, click, alpha=0.0, beta=10.0),
            "rec_view_time_per_imp": _smooth_rate(rec_view_time, rec_imp, alpha=0.0, beta=10.0),
            "search_view_time_per_imp": _smooth_rate(search_view_time, search_imp, alpha=0.0, beta=10.0),
            "full_view_rate": _smooth_rate(full_view_times, valid_view_times, alpha=0.5, beta=1.0),
            "item_is_cold": imp.lt(100.0).astype("float32"),
            "imp_confidence": (np.log1p(imp.clip(lower=0.0)) / np.log1p(max(float(imp.max()), 1.0))).astype("float32"),
            "click_confidence": (np.log1p(click.clip(lower=0.0)) / np.log1p(max(float(click.max()), 1.0))).astype("float32"),
            "ratio_reliability": np.where(imp.ge(100.0), 1.0, np.clip(np.log1p(imp.clip(lower=0.0)) / np.log1p(100.0), 0.0, 1.0)).astype("float32"),
            "imp_bucket_norm": imp_bucket.astype("float32") / 3.0,
            "click_bucket_norm": click_bucket.astype("float32") / 3.0,
            "popularity_bucket_norm": popularity_bucket.astype("float32") / 3.0,
        },
        index=note.index,
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    raw.columns = [f"ratio_feat_{i}" for i in range(raw.shape[1])]
    return raw.astype("float32")


def _cold_stage_from_imp(imp: pd.Series) -> pd.Series:
    imp = imp.fillna(0.0)
    stage = np.where(
        imp.lt(10.0),
        1,
        np.where(imp.lt(100.0), 2, np.where(imp.lt(1000.0), 3, 4)),
    )
    return pd.Series(stage, index=imp.index, dtype="int8")


def build_text_stat_features(note: pd.DataFrame) -> pd.DataFrame:
    title = note["note_title"].fillna("").astype(str) if "note_title" in note.columns else pd.Series("", index=note.index)
    content = note["note_content"].fillna("").astype(str) if "note_content" in note.columns else pd.Series("", index=note.index)
    title_len = title.str.len().clip(0, 512).astype("float32") / 512.0
    content_len = content.str.len().clip(0, 4096).astype("float32") / 4096.0

    def _char_overlap(a: str, b: str) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 0.0
        return float(len(sa & sb) / max(len(sa | sb), 1))

    similarity = pd.Series([_char_overlap(a, b) for a, b in zip(title.tolist(), content.tolist())], index=note.index, dtype="float32")
    return pd.DataFrame(
        {
            "text_stat_feat_0": title_len,
            "text_stat_feat_1": content_len,
            "text_stat_feat_2": similarity,
        },
        index=note.index,
    ).astype("float32")


def _build_item_feature_lookup(features: pd.DataFrame, prefix: str, out_dim: int) -> np.ndarray:
    if features.empty or "item_id" not in features.columns:
        return np.zeros((1, out_dim), dtype=np.float32)
    cols = _feature_columns(features.columns, prefix)
    max_id = int(features["item_id"].max()) + 1
    lookup = np.zeros((max_id, out_dim), dtype=np.float32)
    if not cols:
        return lookup
    use_dim = min(len(cols), out_dim)
    item_ids = features["item_id"].to_numpy(dtype=np.int64, copy=False)
    values = features[cols[:use_dim]].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    lookup[item_ids] = values
    return lookup


def _aggregate_history_feature(
    hist_items: np.ndarray,
    lookup: np.ndarray,
    out_dim: int,
    mode: str = "mean",
) -> np.ndarray:
    batch, hist_len = hist_items.shape
    out = np.zeros((batch, out_dim), dtype=np.float32)
    if batch == 0 or hist_len == 0:
        return out
    if mode == "last":
        filled = np.zeros(batch, dtype=bool)
        for pos in range(hist_len - 1, -1, -1):
            ids = hist_items[:, pos]
            valid = (ids > 0) & (ids < lookup.shape[0]) & ~filled
            if not bool(valid.any()):
                continue
            out[valid] = lookup[ids[valid]]
            filled[valid] = True
            if bool(filled.all()):
                break
        return out
    counts = np.zeros(batch, dtype=np.float32)
    for pos in range(hist_len):
        ids = hist_items[:, pos]
        valid = (ids > 0) & (ids < lookup.shape[0])
        if not bool(valid.any()):
            continue
        out[valid] += lookup[ids[valid]]
        counts[valid] += 1.0
    nonzero = counts > 0
    out[nonzero] /= counts[nonzero, None]
    return out


def build_text_lookup_from_sidecar(
    processed_dir: Path,
    *,
    values_name: str = "text_embeddings.npy",
    ids_name: str = "text_embedding_item_ids.npy",
    out_dim: int = HISTORY_TEXT_SUMMARY_DIM,
) -> np.ndarray:
    values_path = processed_dir / values_name
    ids_path = processed_dir / ids_name
    if not values_path.exists() or not ids_path.exists():
        return np.zeros((1, out_dim), dtype=np.float32)
    item_ids = np.load(ids_path).astype(np.int64, copy=False)
    text_values = np.load(values_path, mmap_mode="r")
    max_id = int(item_ids.max(initial=0)) + 1
    lookup = np.zeros((max_id, out_dim), dtype=np.float32)
    use_dim = min(int(text_values.shape[1]), out_dim)
    lookup[item_ids] = np.asarray(text_values[:, :use_dim], dtype=np.float32)
    return lookup


def _add_history_semantic_summary(
    df: pd.DataFrame,
    notes: pd.DataFrame,
    max_history: int,
    *,
    text_lookup: np.ndarray | None = None,
    ratio_lookup: np.ndarray | None = None,
) -> pd.DataFrame:
    hist_cols = history_columns(max_history)
    hist_items = df[hist_cols].to_numpy(dtype=np.int64, copy=False)
    if text_lookup is None:
        text_prefixes = ("text_feat_", "text_title_feat_", "text_content_feat_", "text_stat_feat_")
        text_lookup = np.zeros((1, HISTORY_TEXT_SUMMARY_DIM), dtype=np.float32)
        for prefix in text_prefixes:
            lookup = _build_item_feature_lookup(notes, prefix, HISTORY_TEXT_SUMMARY_DIM)
            if lookup.shape[0] > 1:
                text_lookup = lookup
                break
    if ratio_lookup is None:
        ratio_lookup = _build_item_feature_lookup(notes, "ratio_feat_", HISTORY_RATIO_SUMMARY_DIM)
    text_mean = _aggregate_history_feature(hist_items, text_lookup, HISTORY_TEXT_SUMMARY_DIM, mode="mean")
    text_last = _aggregate_history_feature(hist_items, text_lookup, HISTORY_TEXT_SUMMARY_DIM, mode="last")
    ratio_mean = _aggregate_history_feature(hist_items, ratio_lookup, HISTORY_RATIO_SUMMARY_DIM, mode="mean")
    text_mean_df = pd.DataFrame(
        text_mean,
        columns=[f"history_text_feat_{idx}" for idx in range(HISTORY_TEXT_SUMMARY_DIM)],
        index=df.index,
    )
    text_last_df = pd.DataFrame(
        text_last,
        columns=[f"history_text_last_feat_{idx}" for idx in range(HISTORY_TEXT_SUMMARY_DIM)],
        index=df.index,
    )
    ratio_df = pd.DataFrame(
        ratio_mean,
        columns=[f"history_ratio_feat_{idx}" for idx in range(HISTORY_RATIO_SUMMARY_DIM)],
        index=df.index,
    )
    return pd.concat([df, text_mean_df, text_last_df, ratio_df], axis=1)


def build_note_features(notes_df: pd.DataFrame, item_map: dict[int, int], text_hash_dim: int = 0) -> pd.DataFrame:
    note = notes_df[notes_df["note_idx"].astype(int).isin(item_map)].copy()
    note["raw_item_id"] = note["note_idx"].astype(int)
    note["item_id"] = note["raw_item_id"].map(item_map).fillna(0).astype(int)
    note["item_type"] = note["note_type"].fillna(0).astype(int) if "note_type" in note.columns else 0
    note["taxonomy1_key"] = _clean_key_series(note["taxonomy1_id"]) if "taxonomy1_id" in note.columns else ""
    note["taxonomy2_key"] = _clean_key_series(note["taxonomy2_id"]) if "taxonomy2_id" in note.columns else ""
    note["taxonomy_key"] = _clean_key_series(note["taxonomy3_id"]) if "taxonomy3_id" in note.columns else ""

    features = note[["item_id", "raw_item_id", "item_type", "taxonomy1_key", "taxonomy2_key", "taxonomy_key"]].copy().reset_index(drop=True)
    item_dense = build_item_dense_features(note).reset_index(drop=True)
    if not item_dense.empty:
        features = pd.concat([features, item_dense], axis=1)

    ratio = build_ratio_features(note).reset_index(drop=True)
    if not ratio.empty:
        features = pd.concat([features, ratio], axis=1)

    text_stat = build_text_stat_features(note).reset_index(drop=True)
    features = pd.concat([features, text_stat], axis=1)

    imp = _safe_float(note, "imp_num")
    click = _safe_float(note, "click_num")
    path_lists = _image_path_lists(note)
    path_count = pd.Series([len(paths) for paths in path_lists], index=note.index, dtype="float32")
    features["cold_stage_id"] = _cold_stage_from_imp(imp).reset_index(drop=True)
    features["item_is_cold"] = imp.lt(100.0).astype("int8").reset_index(drop=True)
    features["item_imp_bucket"] = pd.cut(imp, bins=[-np.inf, 9.0, 99.0, 999.0, np.inf], labels=[0, 1, 2, 3]).astype("int8").reset_index(drop=True)
    features["item_click_bucket"] = pd.cut(click, bins=[-np.inf, 0.0, 9.0, 99.0, np.inf], labels=[0, 1, 2, 3]).astype("int8").reset_index(drop=True)
    features["has_image_emb"] = ((note["image_num"].fillna(0.0).gt(0) if "image_num" in note.columns else False) | path_count.gt(0)).astype("int8").reset_index(drop=True)

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


def build_user_category_maps(user_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        "gender": _compact_user_category_map(user_df.get("gender", pd.Series(dtype=object)), min_count=1),
        "platform": _compact_user_category_map(user_df.get("platform", pd.Series(dtype=object)), min_count=1),
        "age": _compact_user_category_map(user_df.get("age", pd.Series(dtype=object)), min_count=1),
        "location": _compact_user_category_map(
            user_df.get("location", pd.Series(dtype=object)),
            min_count=LOCATION_MIN_COUNT,
            low_freq_unk=True,
        ),
    }


def build_user_features(user_df: pd.DataFrame, user_map: dict[int, int], category_maps: dict[str, dict[str, int]]) -> pd.DataFrame:
    users = user_df[user_df["user_idx"].astype(int).isin(user_map)].copy()
    users["raw_user_id"] = users["user_idx"].astype(int)
    users["user_id"] = users["raw_user_id"].map(user_map).fillna(0).astype(int)
    features = users[["user_id", "raw_user_id"]].copy().reset_index(drop=True)
    for name in USER_CATEGORICAL_COLUMNS:
        raw = users[name] if name in users.columns else pd.Series("", index=users.index)
        features[f"{name}_id"] = _apply_key_map(raw, category_maps.get(name, {})).reset_index(drop=True)
    dense_cols = [c for c in USER_DENSE_COLUMNS if c in users.columns]
    if dense_cols:
        dense = users[dense_cols].fillna(0.0).astype("float32").clip(lower=0.0)
        for col in ["fans_num", "follows_num"]:
            if col in dense.columns:
                dense[col] = np.log1p(dense[col].clip(upper=1_000_000.0))
        dense = dense.reset_index(drop=True)
        dense.columns = [f"user_dense_feat_{i}" for i in range(dense.shape[1])]
        features = pd.concat([features, dense], axis=1)
    return features


def _save_key_mapping(output_dir: Path, name: str, key_col: str, id_col: str, mapping: dict[str, int]) -> None:
    pd.DataFrame({key_col: list(mapping.keys()), id_col: list(mapping.values())}).to_parquet(output_dir / f"{name}_map.parquet", index=False)


def _save_user_category_maps(output_dir: Path, category_maps: dict[str, dict[str, int]]) -> None:
    for name, mapping in category_maps.items():
        _save_key_mapping(output_dir, f"{name}_id", name, f"{name}_id", mapping)


def _save_mappings(
    output_dir: Path,
    item_map: dict[int, int],
    user_map: dict[int, int],
    taxonomy1_map: dict[str, int],
    taxonomy2_map: dict[str, int],
    taxonomy_map: dict[str, int],
    user_category_maps: dict[str, dict[str, int]],
) -> None:
    pd.DataFrame({"raw_item_id": list(item_map.keys()), "item_id": list(item_map.values())}).to_parquet(output_dir / "item_id_map.parquet", index=False)
    pd.DataFrame({"raw_user_id": list(user_map.keys()), "user_id": list(user_map.values())}).to_parquet(output_dir / "user_id_map.parquet", index=False)
    _save_key_mapping(output_dir, "taxonomy1_id", "taxonomy1_key", "taxonomy1_id", taxonomy1_map)
    _save_key_mapping(output_dir, "taxonomy2_id", "taxonomy2_key", "taxonomy2_id", taxonomy2_map)
    _save_key_mapping(output_dir, "taxonomy_id", "taxonomy_key", "taxonomy_id", taxonomy_map)
    _save_user_category_maps(output_dir, user_category_maps)


def _lookup_array(features: pd.DataFrame, value_col: str) -> np.ndarray:
    if features.empty:
        return np.zeros(1, dtype=np.int64)
    size = int(features["item_id"].max()) + 1
    lookup = np.zeros(size, dtype=np.int64)
    item_ids = features["item_id"].to_numpy(dtype=np.int64, copy=False)
    values = features[value_col].fillna(0).astype(int).to_numpy(dtype=np.int64, copy=False)
    lookup[item_ids] = values
    return lookup


def _annotate_history_from_lookup(df: pd.DataFrame, lookup: np.ndarray, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for i, col in enumerate(columns):
        hist_col = f"hist_item_{i}"
        if hist_col not in df.columns:
            df[col] = 0
            continue
        ids = df[hist_col].fillna(0).astype(int).to_numpy()
        valid = (ids >= 0) & (ids < len(lookup))
        values = np.zeros(len(df), dtype=np.int64)
        values[valid] = lookup[ids[valid]]
        df[col] = values
    return df


def _add_cross_features(df: pd.DataFrame, max_history: int) -> pd.DataFrame:
    df = df.copy()
    hist_items = df[history_columns(max_history)].to_numpy(dtype=np.int64, copy=False)
    valid = hist_items > 0
    denom = np.maximum(valid.sum(axis=1), 1)
    match_specs = [
        (history_type_columns(max_history), "item_type"),
        (history_taxonomy1_columns(max_history), "taxonomy1_id"),
        (history_taxonomy2_columns(max_history), "taxonomy2_id"),
        (history_taxonomy_columns(max_history), "taxonomy_id"),
    ]
    values = []
    for hist_cols, target_col in match_specs:
        hist = df[hist_cols].to_numpy(dtype=np.int64, copy=False)
        target = df[target_col].fillna(0).astype(int).to_numpy()[:, None]
        match = (hist == target) & (target > 0) & valid
        values.append(match.sum(axis=1) / denom)
    values.append(valid.sum(axis=1) / max(max_history, 1))
    for idx, value in enumerate(values):
        df[f"cross_feat_{idx}"] = value.astype("float32")
    return df


def _standardize_splits(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    prefixes: Iterable[str] = STANDARDIZED_PREFIXES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    stats: dict[str, dict[str, list[float] | list[str]]] = {}
    splits = [train.copy(), valid.copy(), test.copy()]
    for prefix in prefixes:
        cols = _feature_columns(train.columns, prefix)
        if not cols:
            continue
        mean = splits[0][cols].mean(axis=0).astype("float32")
        std = splits[0][cols].std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0).astype("float32")
        for frame in splits:
            frame[cols] = ((frame[cols] - mean) / std).replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32")
        stats[prefix] = {"columns": cols, "mean": mean.tolist(), "std": std.tolist()}
    return splits[0], splits[1], splits[2], stats


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
    users_raw = _read_parquet_dir(qilin_dir / "user_feat", columns=USER_COLUMNS)
    user_category_maps = build_user_category_maps(users_raw)
    users = build_user_features(users_raw, user_map=user_map, category_maps=user_category_maps)

    taxonomy1_map = _compact_key_map(notes["taxonomy1_key"].tolist() if "taxonomy1_key" in notes else [])
    taxonomy2_map = _compact_key_map(notes["taxonomy2_key"].tolist() if "taxonomy2_key" in notes else [])
    taxonomy_map = _compact_key_map(notes["taxonomy_key"].tolist() if "taxonomy_key" in notes else [])
    notes["taxonomy1_id"] = _apply_key_map(notes["taxonomy1_key"], taxonomy1_map) if "taxonomy1_key" in notes else 0
    notes["taxonomy2_id"] = _apply_key_map(notes["taxonomy2_key"], taxonomy2_map) if "taxonomy2_key" in notes else 0
    notes["taxonomy_id"] = _apply_key_map(notes["taxonomy_key"], taxonomy_map) if "taxonomy_key" in notes else 0

    def enrich(df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(notes, on=["item_id", "raw_item_id"], how="left")
        df = df.merge(users, on=["user_id", "raw_user_id"], how="left")
        df["item_type"] = df["item_type"].fillna(0).astype(int)
        for col in ["taxonomy1_id", "taxonomy2_id", "taxonomy_id", "gender_id", "platform_id", "age_id", "location_id"]:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0).astype(int)
        feature_prefixes = tuple(FEATURE_PREFIXES.values())
        feature_cols = [c for c in df.columns if c.startswith(feature_prefixes)]
        if feature_cols:
            df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32")
        if "query" in df.columns:
            df["has_query"] = df["query"].fillna("").astype(str).str.len().gt(0).astype("int8")
        else:
            df["has_query"] = 0
        for col in ["cold_stage_id", "item_imp_bucket", "item_click_bucket", "has_image_emb"]:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype("int8")
        return df

    train_flat = enrich(train_flat)
    test_flat = enrich(test_flat)
    for frame in [train_flat, test_flat]:
        drop_cols = [c for c in ["taxonomy1_key", "taxonomy2_key", "taxonomy_key"] if c in frame.columns]
        if drop_cols:
            frame.drop(columns=drop_cols, inplace=True)

    item_type_lookup, item_taxonomy_lookup = build_item_intent_lookups(notes[["item_id", "item_type", "taxonomy_id"]])
    train_flat = annotate_dynamic_intents(train_flat, item_type_lookup, item_taxonomy_lookup, max_history=max_history)
    test_flat = annotate_dynamic_intents(test_flat, item_type_lookup, item_taxonomy_lookup, max_history=max_history)
    taxonomy1_lookup = _lookup_array(notes[["item_id", "taxonomy1_id"]], "taxonomy1_id")
    taxonomy2_lookup = _lookup_array(notes[["item_id", "taxonomy2_id"]], "taxonomy2_id")
    train_flat = _annotate_history_from_lookup(train_flat, taxonomy1_lookup, history_taxonomy1_columns(max_history))
    train_flat = _annotate_history_from_lookup(train_flat, taxonomy2_lookup, history_taxonomy2_columns(max_history))
    test_flat = _annotate_history_from_lookup(test_flat, taxonomy1_lookup, history_taxonomy1_columns(max_history))
    test_flat = _annotate_history_from_lookup(test_flat, taxonomy2_lookup, history_taxonomy2_columns(max_history))
    train_flat = _add_cross_features(train_flat, max_history=max_history)
    test_flat = _add_cross_features(test_flat, max_history=max_history)
    train_flat = _add_history_semantic_summary(train_flat, notes, max_history=max_history)
    test_flat = _add_history_semantic_summary(test_flat, notes, max_history=max_history)

    train_flat = train_flat.sort_values(["timestamp", "request_id", "position"]).reset_index(drop=True)
    split = int(len(train_flat) * 0.9)
    train, valid = train_flat.iloc[:split].reset_index(drop=True), train_flat.iloc[split:].reset_index(drop=True)
    train, valid, test_flat, dense_stats = _standardize_splits(train, valid, test_flat)

    write_table(train, output_dir / "train.parquet")
    write_table(valid, output_dir / "valid.parquet")
    write_table(test_flat, output_dir / "test.parquet")
    metadata = infer_metadata(pd.concat([train, valid, test_flat], ignore_index=True), max_history=max_history)
    metadata["num_cold_stages"] = NUM_COLD_STAGES
    metadata["history_text_dim"] = HISTORY_TEXT_SUMMARY_DIM
    metadata["history_text_last_dim"] = HISTORY_TEXT_SUMMARY_DIM
    metadata["history_ratio_dim"] = HISTORY_RATIO_SUMMARY_DIM
    metadata["text_stat_dim"] = TEXT_STAT_DIM
    write_json(metadata, output_dir / "metadata.json")
    write_json(dense_stats, output_dir / "feature_standardization.json")
    _save_mappings(output_dir, item_map, user_map, taxonomy1_map, taxonomy2_map, taxonomy_map, user_category_maps)
    summary = {
        "num_raw_items": len(item_map),
        "num_raw_users": len(user_map),
        "num_taxonomy1": len(taxonomy1_map),
        "num_taxonomy2": len(taxonomy2_map),
        "num_taxonomies": len(taxonomy_map),
        "train_rows": len(train),
        "valid_rows": len(valid),
        "test_rows": len(test_flat),
    }
    write_json(summary, output_dir / "id_mapping_summary.json")
    return metadata
