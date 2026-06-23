from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd


SPLITS = ("train", "valid", "test")


def _feature_columns(columns: list[str], prefix: str) -> list[str]:
    def suffix_value(name: str) -> int:
        try:
            return int(name.rsplit("_", 1)[-1])
        except ValueError:
            return -1

    return sorted([c for c in columns if c.startswith(f"{prefix}_feat_")], key=suffix_value)


def _next_feature_offset(columns: list[str], prefix: str) -> int:
    values = []
    for col in _feature_columns(columns, prefix):
        try:
            values.append(int(col.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return max(values) + 1 if values else 0


def merge_one(df: pd.DataFrame, values: np.ndarray, item_ids: np.ndarray, prefix: str, mode: str) -> pd.DataFrame:
    values = values.astype("float32", copy=False)
    old_cols = _feature_columns(list(df.columns), prefix)
    if mode == "replace" and old_cols:
        df = df.drop(columns=old_cols)
        offset = 0
    else:
        offset = _next_feature_offset(list(df.columns), prefix)
    emb = pd.DataFrame(values, columns=[f"{prefix}_feat_{offset + i}" for i in range(values.shape[1])])
    emb["item_id"] = item_ids.astype(int)
    return df.merge(emb, on="item_id", how="left")


def _copy_support_files(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.name in {f"{split}.parquet" for split in SPLITS} or path.name == "metadata.json":
            continue
        target = output_dir / path.name
        if path.is_file():
            shutil.copy2(path, target)


def _resolve_mode(name: str, merge_mode: str) -> str:
    if merge_mode == "auto":
        return "replace" if name == "text" else "append"
    return merge_mode


def load_requested_embeddings(processed_dir: Path, enabled: dict[str, bool]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    merges: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, is_enabled in enabled.items():
        if not is_enabled:
            continue
        values_path = processed_dir / f"{name}_embeddings.npy"
        ids_path = processed_dir / f"{name}_embedding_item_ids.npy"
        if not values_path.exists() or not ids_path.exists():
            raise FileNotFoundError(f"Missing {values_path} or {ids_path}")
        merges.append((name, np.load(values_path), np.load(ids_path)))
    return merges


def merge_embeddings(
    processed_dir: Path,
    output_dir: Path,
    enabled: dict[str, bool],
    merge_mode: str = "auto",
) -> dict[str, object]:
    merges = load_requested_embeddings(processed_dir, enabled)
    if not merges:
        raise ValueError("No embeddings selected. Pass --text, --image, or --video.")

    same_dir = processed_dir.resolve() == output_dir.resolve()
    if not same_dir:
        _copy_support_files(processed_dir, output_dir)

    split_shapes: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        path = processed_dir / f"{split}.parquet"
        df = pd.read_parquet(path)
        for prefix, values, item_ids in merges:
            df = merge_one(df, values, item_ids, prefix, mode=_resolve_mode(prefix, merge_mode))
        feat_cols = [
            c
            for c in df.columns
            if c.startswith("text_feat_") or c.startswith("image_feat_") or c.startswith("video_feat_")
        ]
        if feat_cols:
            df[feat_cols] = df[feat_cols].fillna(0.0).astype("float32")
        df.to_parquet(output_dir / f"{split}.parquet", index=False)
        split_shapes[split] = {"rows": int(len(df)), "feature_columns": int(len(feat_cols))}
        print(f"updated {output_dir / f'{split}.parquet'} with {len(feat_cols)} multimodal columns")

    metadata_path = processed_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sample = pd.read_parquet(output_dir / "train.parquet", columns=None)
    metadata["text_dim"] = len(_feature_columns(list(sample.columns), "text"))
    metadata["image_dim"] = len(_feature_columns(list(sample.columns), "image"))
    metadata["video_dim"] = len(_feature_columns(list(sample.columns), "video"))
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "source_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "merge_mode": merge_mode,
        "merged": [name for name, _, _ in merges],
        "metadata": {
            "text_dim": int(metadata.get("text_dim", 0)),
            "image_dim": int(metadata.get("image_dim", 0)),
            "video_dim": int(metadata.get("video_dim", 0)),
        },
        "splits": split_shapes,
    }
    (output_dir / "embedding_merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("updated metadata", summary["metadata"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge precomputed text/image/video embeddings into processed parquet files.")
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "processed" / "qilin_full"))
    parser.add_argument("--output-dir", default=None, help="Optional destination processed directory. Defaults to --processed-dir.")
    parser.add_argument("--text", action="store_true", help="Merge text_embeddings.npy if present.")
    parser.add_argument("--image", action="store_true", help="Merge image_embeddings.npy if present.")
    parser.add_argument("--video", action="store_true", help="Merge video_embeddings.npy if present.")
    parser.add_argument(
        "--merge-mode",
        choices=["auto", "append", "replace"],
        default="auto",
        help="auto replaces text features and appends image/video embeddings after existing metadata features.",
    )
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    output = Path(args.output_dir) if args.output_dir else processed
    merge_embeddings(
        processed_dir=processed,
        output_dir=output,
        enabled={"text": args.text, "image": args.image, "video": args.video},
        merge_mode=args.merge_mode,
    )


if __name__ == "__main__":
    main()
