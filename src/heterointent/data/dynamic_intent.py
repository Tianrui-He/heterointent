from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from heterointent.data.io import read_table, write_table
from heterointent.data.schema import (
    INTENT_COLUMNS,
    TASKS,
    history_columns,
    history_taxonomy_columns,
    history_type_columns,
)
from heterointent.utils import read_json, write_json

TASK_WEIGHTS = {"click": 0.3, "collect": 0.4, "share": 0.3}
NOTE_COLUMNS = ["note_idx", "note_type", "taxonomy3_id"]


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


def _ensure_lookup_size(lookup: np.ndarray, max_value: int) -> np.ndarray:
    if max_value < len(lookup):
        return lookup
    expanded = np.zeros(max_value + 1, dtype=lookup.dtype)
    expanded[: len(lookup)] = lookup
    return expanded


def build_item_intent_lookups(item_features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if item_features.empty:
        return np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)
    required = {"item_id", "item_type", "taxonomy_id"}
    missing = required - set(item_features.columns)
    if missing:
        raise ValueError(f"Missing item intent feature columns: {sorted(missing)}")
    features = item_features[list(required)].dropna().copy()
    features["item_id"] = features["item_id"].astype(int)
    features = features[features["item_id"] >= 0].drop_duplicates("item_id", keep="first")
    max_item = int(features["item_id"].max()) if len(features) else 0
    type_lookup = np.zeros(max_item + 1, dtype=np.int64)
    taxonomy_lookup = np.zeros(max_item + 1, dtype=np.int64)
    ids = features["item_id"].to_numpy(dtype=np.int64)
    type_lookup[ids] = features["item_type"].fillna(0).astype(int).to_numpy(dtype=np.int64)
    taxonomy_lookup[ids] = features["taxonomy_id"].fillna(0).astype(int).to_numpy(dtype=np.int64)
    return type_lookup, taxonomy_lookup


def build_item_intent_lookups_from_processed(
    processed_dir: str | Path, qilin_dir: str | Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    processed_dir = Path(processed_dir)
    if qilin_dir is not None:
        qilin_dir = Path(qilin_dir)
        item_map = pd.read_parquet(processed_dir / "item_id_map.parquet")
        taxonomy_map_path = processed_dir / "taxonomy_id_map.parquet"
        taxonomy_map = pd.read_parquet(taxonomy_map_path) if taxonomy_map_path.exists() else pd.DataFrame()
        notes = _read_parquet_dir(qilin_dir / "notes", columns=NOTE_COLUMNS)
        notes = notes.rename(columns={"note_idx": "raw_item_id"})
        notes["raw_item_id"] = notes["raw_item_id"].astype(int)
        features = item_map.merge(notes, on="raw_item_id", how="left")
        features["item_type"] = features["note_type"].fillna(0).astype(int) if "note_type" in features.columns else 0
        if not taxonomy_map.empty and "taxonomy3_id" in features.columns:
            tax = dict(zip(taxonomy_map["taxonomy_key"].astype(str), taxonomy_map["taxonomy_id"].astype(int)))
            features["taxonomy_id"] = (
                features["taxonomy3_id"].fillna("missing").astype(str).map(tax).fillna(0).astype(int)
            )
        else:
            features["taxonomy_id"] = 0
        return build_item_intent_lookups(features[["item_id", "item_type", "taxonomy_id"]])

    frames = []
    for split in ("train", "valid", "test"):
        path = processed_dir / f"{split}.parquet"
        if path.exists():
            frame = read_table(path)
            cols = [c for c in ["item_id", "item_type", "taxonomy_id"] if c in frame.columns]
            if set(cols) == {"item_id", "item_type", "taxonomy_id"}:
                frames.append(frame[cols])
    if not frames:
        return np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)
    return build_item_intent_lookups(pd.concat(frames, ignore_index=True))


def _safe_lookup(values: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    safe = np.where((values >= 0) & (values < len(lookup)), values, 0)
    return lookup[safe]


def _dominant(values: np.ndarray) -> int:
    non_zero = values[values > 0]
    if non_zero.size == 0:
        return 0
    counts = np.bincount(non_zero)
    if counts.size <= 1:
        return 0
    return int(np.argmax(counts[1:]) + 1)


def _dominant_rows(values: np.ndarray) -> np.ndarray:
    return np.array([_dominant(row.astype(np.int64, copy=False)) for row in values], dtype=np.int64)


def _target_intents(df: pd.DataFrame) -> pd.DataFrame:
    strength = sum(TASK_WEIGHTS[task] * df[task].fillna(0).astype(float) for task in TASKS)
    positives = df.loc[strength > 0, ["request_id", "item_type", "taxonomy_id", "position"]].copy()
    positives["_intent_strength"] = strength.loc[positives.index].to_numpy()
    if positives.empty:
        return pd.DataFrame(columns=["request_id", "target_item_type", "target_taxonomy_id"])
    positives = positives.sort_values(
        ["request_id", "_intent_strength", "position"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    targets = positives.drop_duplicates("request_id", keep="first")
    return targets.rename(columns={"item_type": "target_item_type", "taxonomy_id": "target_taxonomy_id"})[
        ["request_id", "target_item_type", "target_taxonomy_id"]
    ]


def annotate_dynamic_intents(
    df: pd.DataFrame,
    item_type_lookup: np.ndarray,
    item_taxonomy_lookup: np.ndarray,
    max_history: int = 20,
) -> pd.DataFrame:
    df = df.copy()
    hist_cols = history_columns(max_history)
    hist_type_cols = history_type_columns(max_history)
    hist_tax_cols = history_taxonomy_columns(max_history)
    for col in hist_cols:
        if col not in df.columns:
            df[col] = 0
    for task in TASKS:
        if task not in df.columns:
            df[task] = 0

    max_hist_item = int(df[hist_cols].max(numeric_only=True).max()) if hist_cols else 0
    item_type_lookup = _ensure_lookup_size(item_type_lookup, max_hist_item)
    item_taxonomy_lookup = _ensure_lookup_size(item_taxonomy_lookup, max_hist_item)

    request_hist = df.drop_duplicates("request_id", keep="first")[["request_id", *hist_cols]].copy()
    hist_items = request_hist[hist_cols].fillna(0).astype(int).to_numpy(dtype=np.int64, copy=True)
    hist_types = _safe_lookup(hist_items, item_type_lookup)
    hist_taxonomies = _safe_lookup(hist_items, item_taxonomy_lookup)

    for i, col in enumerate(hist_type_cols):
        request_hist[col] = hist_types[:, i]
    for i, col in enumerate(hist_tax_cols):
        request_hist[col] = hist_taxonomies[:, i]
    request_hist["hist_dominant_item_type"] = _dominant_rows(hist_types)
    request_hist["hist_dominant_taxonomy_id"] = _dominant_rows(hist_taxonomies)

    targets = _target_intents(df)
    drop_cols = [c for c in [*INTENT_COLUMNS, *hist_type_cols, *hist_tax_cols] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = df.merge(
        request_hist[["request_id", "hist_dominant_item_type", "hist_dominant_taxonomy_id", *hist_type_cols, *hist_tax_cols]],
        on="request_id",
        how="left",
    )
    df = df.merge(targets, on="request_id", how="left")

    fill_int_cols = [
        "target_item_type",
        "target_taxonomy_id",
        "hist_dominant_item_type",
        "hist_dominant_taxonomy_id",
        *hist_type_cols,
        *hist_tax_cols,
    ]
    for col in fill_int_cols:
        df[col] = df[col].fillna(0).astype(int)
    df["has_intent_target"] = df["target_item_type"].gt(0).astype(int)
    df["is_type_shift"] = (
        df["has_intent_target"].eq(1)
        & df["hist_dominant_item_type"].gt(0)
        & df["target_item_type"].ne(df["hist_dominant_item_type"])
    ).astype(int)
    df["is_taxonomy_shift"] = (
        df["has_intent_target"].eq(1)
        & df["hist_dominant_taxonomy_id"].gt(0)
        & df["target_taxonomy_id"].ne(df["hist_dominant_taxonomy_id"])
    ).astype(int)
    df["next_item_type"] = df["target_item_type"].astype(int)
    return df


def annotate_processed_directory(
    processed_dir: str | Path,
    qilin_dir: str | Path | None = None,
    splits: Iterable[str] = ("train", "valid", "test"),
    max_history: int | None = None,
) -> dict[str, dict[str, float]]:
    processed_dir = Path(processed_dir)
    metadata_path = processed_dir / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    max_history = int(max_history or metadata.get("max_history", 20))
    type_lookup, taxonomy_lookup = build_item_intent_lookups_from_processed(processed_dir, qilin_dir=qilin_dir)

    summary: dict[str, dict[str, float]] = {}
    for split in splits:
        path = processed_dir / f"{split}.parquet"
        if not path.exists():
            continue
        df = read_table(path)
        annotated = annotate_dynamic_intents(df, type_lookup, taxonomy_lookup, max_history=max_history)
        write_table(annotated, path)
        request_level = annotated.drop_duplicates("request_id", keep="first")
        summary[split] = {
            "rows": float(len(annotated)),
            "requests": float(request_level["request_id"].nunique()),
            "intent_target_requests": float(request_level["has_intent_target"].sum()),
            "type_shift_requests": float(request_level["is_type_shift"].sum()),
            "taxonomy_shift_requests": float(request_level["is_taxonomy_shift"].sum()),
        }

    metadata["dynamic_intent"] = True
    metadata["max_history"] = max_history
    write_json(metadata, metadata_path)
    write_json(summary, processed_dir / "dynamic_intent_summary.json")
    return summary
