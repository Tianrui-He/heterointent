from __future__ import annotations

from pathlib import Path

import pandas as pd

from heterointent.data.io import read_table, write_table
from heterointent.data.schema import BASE_COLUMNS, FEATURE_PREFIXES, history_columns, prefixed_columns
from heterointent.utils import write_json


def infer_metadata(df: pd.DataFrame, max_history: int = 20) -> dict:
    columns = list(df.columns)
    metadata = {
        "num_users": int(df["user_id"].max()) + 1 if "user_id" in df else 1,
        "num_items": int(df["item_id"].max()) + 1 if "item_id" in df else 1,
        "num_item_types": int(df["item_type"].max()) + 1 if "item_type" in df else 1,
        "num_taxonomy1": int(df["taxonomy1_id"].max()) + 1 if "taxonomy1_id" in df else 1,
        "num_taxonomy2": int(df["taxonomy2_id"].max()) + 1 if "taxonomy2_id" in df else 1,
        "num_taxonomies": int(df["taxonomy_id"].max()) + 1 if "taxonomy_id" in df else 1,
        "num_genders": int(df["gender_id"].max()) + 1 if "gender_id" in df else 1,
        "num_platforms": int(df["platform_id"].max()) + 1 if "platform_id" in df else 1,
        "num_ages": int(df["age_id"].max()) + 1 if "age_id" in df else 1,
        "num_locations": int(df["location_id"].max()) + 1 if "location_id" in df else 1,
        "max_history": max_history,
    }
    for name, prefix in FEATURE_PREFIXES.items():
        metadata[f"{name}_dim"] = len(prefixed_columns(columns, prefix))
    return metadata


def normalize_flat_samples(df: pd.DataFrame, max_history: int = 20) -> pd.DataFrame:
    rename_map = {
        "note_id": "item_id",
        "rec_request_id": "request_id",
        "search_id": "request_id",
        "type": "item_type",
        "cat_id": "taxonomy_id",
        "collect_label": "collect",
        "share_label": "share",
        "click_label": "click",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()
    for col in BASE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    for col in history_columns(max_history):
        if col not in df.columns:
            df[col] = 0
    return df


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sort_cols = [c for c in ["timestamp", "request_id", "position"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    n = len(df)
    return (
        df.iloc[: int(n * 0.8)].reset_index(drop=True),
        df.iloc[int(n * 0.8) : int(n * 0.9)].reset_index(drop=True),
        df.iloc[int(n * 0.9) :].reset_index(drop=True),
    )


def preprocess_flat_samples(input_path: str | Path, output_dir: str | Path, max_history: int = 20) -> dict:
    output_dir = Path(output_dir)
    df = normalize_flat_samples(read_table(input_path), max_history=max_history)
    metadata = infer_metadata(df, max_history=max_history)
    train, valid, test = split_by_time(df)
    write_table(train, output_dir / "train.parquet")
    write_table(valid, output_dir / "valid.parquet")
    write_table(test, output_dir / "test.parquet")
    write_json(metadata, output_dir / "metadata.json")
    return metadata
