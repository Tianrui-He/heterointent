from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterointent.data.preprocess import infer_metadata
from heterointent.data.qilin import (
    HISTORY_RATIO_SUMMARY_DIM,
    HISTORY_TEXT_SUMMARY_DIM,
    NUM_COLD_STAGES,
    TEXT_STAT_DIM,
    _add_history_semantic_summary,
    _build_item_feature_lookup,
    _read_parquet_dir,
    build_note_features,
    build_text_lookup_from_sidecar,
)
from heterointent.utils import write_json

NOTE_COLUMNS = [
    "note_idx",
    "note_title",
    "note_content",
    "note_type",
    "taxonomy1_id",
    "taxonomy2_id",
    "taxonomy3_id",
    "image_path",
    "imp_num",
    "click_num",
    "image_num",
    "video_duration",
]

ITEM_V2_COLUMNS = [
    "item_id",
    "raw_item_id",
    "text_stat_feat_0",
    "text_stat_feat_1",
    "text_stat_feat_2",
    "cold_stage_id",
    "has_image_emb",
]


def _load_item_map(processed_dir: Path) -> dict[int, int]:
    mapping = pd.read_parquet(processed_dir / "item_id_map.parquet")
    return dict(zip(mapping["raw_item_id"].astype(int), mapping["item_id"].astype(int)))


def _build_notes_table(qilin_dir: Path, item_map: dict[int, int], processed_dir: Path) -> pd.DataFrame:
    notes_raw = _read_parquet_dir(qilin_dir / "notes", columns=NOTE_COLUMNS)
    return build_note_features(notes_raw, item_map=item_map, text_hash_dim=0)


def _build_history_lookups(notes: pd.DataFrame, processed_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    text_lookup = build_text_lookup_from_sidecar(processed_dir, out_dim=HISTORY_TEXT_SUMMARY_DIM)
    ratio_lookup = _build_item_feature_lookup(notes, "ratio_feat_", HISTORY_RATIO_SUMMARY_DIM)
    return text_lookup, ratio_lookup


def _upgrade_split(
    src: Path,
    dst: Path,
    item_cols: pd.DataFrame,
    notes: pd.DataFrame,
    max_history: int,
    chunk_rows: int,
    *,
    text_lookup: np.ndarray | None = None,
    ratio_lookup: np.ndarray | None = None,
) -> None:
    pf = pq.ParquetFile(src)
    writer: pq.ParquetWriter | None = None
    drop_cols = [c for c in item_cols.columns if c not in {"item_id", "raw_item_id"}]
    for batch in pf.iter_batches(batch_size=chunk_rows):
        frame = batch.to_pandas()
        frame = frame.drop(columns=[c for c in drop_cols if c in frame.columns], errors="ignore")
        frame = frame.merge(item_cols, on=["item_id", "raw_item_id"], how="left")
        if "query" in frame.columns:
            frame["has_query"] = frame["query"].fillna("").astype(str).str.len().gt(0).astype("int8")
        history_cols = [
            c
            for c in frame.columns
            if c.startswith(("history_text_feat_", "history_text_last_feat_", "history_ratio_feat_"))
        ]
        if history_cols:
            frame = frame.drop(columns=history_cols, errors="ignore")
        frame = _add_history_semantic_summary(
            frame,
            notes,
            max_history=max_history,
            text_lookup=text_lookup,
            ratio_lookup=ratio_lookup,
        )
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            dst.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(dst, table.schema, compression="zstd")
        writer.write_table(table)
    if writer is not None:
        writer.close()


def _copy_sidecars(processed_dir: Path, output_dir: Path) -> None:
    if processed_dir.resolve() == output_dir.resolve():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in processed_dir.iterdir():
        if path.name.endswith(".parquet") and path.stem in {"train", "valid", "test"}:
            continue
        dst = output_dir / path.name
        if path.is_dir():
            if dst.exists():
                continue
            shutil.copytree(path, dst)
        elif not dst.exists():
            shutil.copy2(path, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add v2 feature-opt columns to existing processed parquet splits.")
    parser.add_argument("--processed-dir", default="data/run_latest/processed/qilin_base")
    parser.add_argument("--output-dir", default="data/run_latest/processed/qilin_v2")
    parser.add_argument("--qilin-dir", default="data/raw/Qilin")
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--chunk-rows", type=int, default=65536)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    qilin_dir = Path(args.qilin_dir)
    item_map = _load_item_map(processed_dir)
    notes = _build_notes_table(qilin_dir, item_map, processed_dir)
    text_lookup, ratio_lookup = _build_history_lookups(notes, processed_dir)
    item_cols = notes[ITEM_V2_COLUMNS].drop_duplicates(["item_id", "raw_item_id"])

    for split in ("train", "valid", "test"):
        src = processed_dir / f"{split}.parquet"
        dst = output_dir / f"{split}.parquet"
        print(f"upgrading {src} -> {dst}")
        _upgrade_split(
            src,
            dst,
            item_cols,
            notes,
            args.max_history,
            args.chunk_rows,
            text_lookup=text_lookup,
            ratio_lookup=ratio_lookup,
        )

    sample_parts: list[pd.DataFrame] = []
    for split in ("train", "valid", "test"):
        pf = pq.ParquetFile(output_dir / f"{split}.parquet")
        sample_parts.append(pf.read_row_group(0).to_pandas().head(4096))
    meta = infer_metadata(pd.concat(sample_parts, ignore_index=True), max_history=args.max_history)
    old_meta = json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))
    for key, value in old_meta.items():
        if key not in meta or meta.get(key) in (0, None):
            meta[key] = value
    meta["num_cold_stages"] = NUM_COLD_STAGES
    meta["history_text_dim"] = HISTORY_TEXT_SUMMARY_DIM
    meta["history_text_last_dim"] = HISTORY_TEXT_SUMMARY_DIM
    meta["history_ratio_dim"] = HISTORY_RATIO_SUMMARY_DIM
    meta["text_stat_dim"] = TEXT_STAT_DIM
    item_map_df = pd.read_parquet(processed_dir / "item_id_map.parquet")
    meta["num_items"] = int(item_map_df["item_id"].max()) + 1
    write_json(meta, output_dir / "metadata.json")
    _copy_sidecars(processed_dir, output_dir)
    print(f"wrote metadata to {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
