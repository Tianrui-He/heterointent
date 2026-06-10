from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd


def merge_one(df: pd.DataFrame, values: np.ndarray, item_ids: np.ndarray, prefix: str) -> pd.DataFrame:
    emb = pd.DataFrame(values, columns=[f"{prefix}_feat_{i}" for i in range(values.shape[1])])
    emb["item_id"] = item_ids.astype(int)
    return df.merge(emb, on="item_id", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge precomputed text/image/video embeddings into processed train/valid/test parquet files.")
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "processed" / "qilin_full"))
    parser.add_argument("--text", action="store_true", help="Merge text_embeddings.npy if present.")
    parser.add_argument("--image", action="store_true", help="Merge image_embeddings.npy if present.")
    parser.add_argument("--video", action="store_true", help="Merge video_embeddings.npy if present.")
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    merges = []
    for name, enabled in [("text", args.text), ("image", args.image), ("video", args.video)]:
        if not enabled:
            continue
        values_path = processed / f"{name}_embeddings.npy"
        ids_path = processed / f"{name}_embedding_item_ids.npy"
        if not values_path.exists() or not ids_path.exists():
            raise FileNotFoundError(f"Missing {values_path} or {ids_path}")
        merges.append((name, np.load(values_path), np.load(ids_path)))

    for split in ["train", "valid", "test"]:
        path = processed / f"{split}.parquet"
        df = pd.read_parquet(path)
        for prefix, values, item_ids in merges:
            old_cols = [c for c in df.columns if c.startswith(f"{prefix}_feat_")]
            if old_cols:
                df = df.drop(columns=old_cols)
            df = merge_one(df, values, item_ids, prefix)
        feat_cols = [c for c in df.columns if c.startswith("text_feat_") or c.startswith("image_feat_") or c.startswith("video_feat_")]
        if feat_cols:
            df[feat_cols] = df[feat_cols].fillna(0.0).astype("float32")
        df.to_parquet(path, index=False)
        print(f"updated {path} with {len(feat_cols)} embedding columns")

    metadata_path = processed / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sample = pd.read_parquet(processed / "train.parquet", columns=None)
    metadata["text_dim"] = len([c for c in sample.columns if c.startswith("text_feat_")])
    metadata["image_dim"] = len([c for c in sample.columns if c.startswith("image_feat_")])
    metadata["video_dim"] = len([c for c in sample.columns if c.startswith("video_feat_")])
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("updated metadata", metadata)


if __name__ == "__main__":
    main()
