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

from heterointent.data.schema import FEATURE_PREFIXES


SPLITS = ("train", "valid", "test")
EMBEDDING_ID_COLUMNS = {
    "text": "item_id",
    "text_title": "item_id",
    "text_content": "item_id",
    "image": "item_id",
    "video": "item_id",
    "query": "request_id",
}


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


def merge_one(df: pd.DataFrame, values: np.ndarray, ids: np.ndarray, id_col: str, prefix: str, mode: str) -> pd.DataFrame:
    values = values.astype("float32", copy=False)
    old_cols = _feature_columns(list(df.columns), prefix)
    if mode == "replace" and old_cols:
        df = df.drop(columns=old_cols)
        offset = 0
    else:
        offset = _next_feature_offset(list(df.columns), prefix)
    emb = pd.DataFrame(values, columns=[f"{prefix}_feat_{offset + i}" for i in range(values.shape[1])])
    emb[id_col] = ids.astype(int)
    return df.merge(emb, on=id_col, how="left")


def _resolve_feature_group(name: str, columns: list[str]) -> str:
    if name == "image" and _feature_columns(columns, "image_meta"):
        return "image_emb"
    if name == "video" and _feature_columns(columns, "video_meta"):
        return "video_emb"
    return name


def _append_visual_presence_flag(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    emb_group = f"{modality}_emb"
    meta_group = f"{modality}_meta"
    emb_cols = _feature_columns(list(df.columns), emb_group)
    meta_cols = _feature_columns(list(df.columns), meta_group)
    if not emb_cols or not meta_cols:
        return df
    offset = _next_feature_offset(list(df.columns), meta_group)
    df[f"{meta_group}_feat_{offset}"] = df[emb_cols].abs().sum(axis=1).gt(0).astype("float32")
    return df


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
        return "replace" if name in {"text", "text_title", "text_content", "query"} else "append"
    return merge_mode


def load_requested_embeddings(processed_dir: Path, enabled: dict[str, bool]) -> list[tuple[str, np.ndarray, np.ndarray, str]]:
    merges: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    for name, is_enabled in enabled.items():
        if not is_enabled:
            continue
        values_path = processed_dir / f"{name}_embeddings.npy"
        id_col = EMBEDDING_ID_COLUMNS[name]
        ids_path = processed_dir / f"{name}_embedding_{id_col}s.npy"
        if not values_path.exists() or not ids_path.exists():
            raise FileNotFoundError(f"Missing {values_path} or {ids_path}")
        merges.append((name, np.load(values_path), np.load(ids_path), id_col))
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
        merged_groups = []
        for name, values, ids, id_col in merges:
            group = _resolve_feature_group(name, list(df.columns))
            df = merge_one(df, values, ids, id_col, group, mode=_resolve_mode(name, merge_mode))
            merged_groups.append(group)
        if "image_emb" in merged_groups:
            df = _append_visual_presence_flag(df, "image")
        if "video_emb" in merged_groups:
            df = _append_visual_presence_flag(df, "video")
        feat_cols = [
            c
            for c in df.columns
            if c.startswith(tuple(FEATURE_PREFIXES.values()))
        ]
        if feat_cols:
            df[feat_cols] = df[feat_cols].fillna(0.0).astype("float32")
        df.to_parquet(output_dir / f"{split}.parquet", index=False)
        split_shapes[split] = {"rows": int(len(df)), "feature_columns": int(len(feat_cols))}
        print(f"updated {output_dir / f'{split}.parquet'} with {len(feat_cols)} multimodal columns")

    metadata_path = processed_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sample = pd.read_parquet(output_dir / "train.parquet", columns=None)
    for group in FEATURE_PREFIXES:
        metadata[f"{group}_dim"] = len(_feature_columns(list(sample.columns), group))
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "source_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "merge_mode": merge_mode,
        "merged": [name for name, _, _, _ in merges],
        "metadata": {
            f"{group}_dim": int(metadata.get(f"{group}_dim", 0))
            for group in FEATURE_PREFIXES
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
    parser.add_argument("--text-title", action="store_true", help="Merge text_title_embeddings.npy if present.")
    parser.add_argument("--text-content", action="store_true", help="Merge text_content_embeddings.npy if present.")
    parser.add_argument("--query", action="store_true", help="Merge query_embeddings.npy by request_id if present.")
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
        enabled={
            "text": args.text,
            "text_title": args.text_title,
            "text_content": args.text_content,
            "query": args.query,
            "image": args.image,
            "video": args.video,
        },
        merge_mode=args.merge_mode,
    )


if __name__ == "__main__":
    main()
