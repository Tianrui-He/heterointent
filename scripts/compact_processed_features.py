from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterointent.data.schema import FEATURE_PREFIXES


SPLITS = ("train", "valid", "test")
DEFAULT_GROUPS = ("text", "text_title", "text_content", "query", "image_emb")
SIDECAR_SOURCES = {
    "text": ("text", "item_id"),
    "text_title": ("text_title", "item_id"),
    "text_content": ("text_content", "item_id"),
    "query": ("query", "request_id"),
    "image_emb": ("image", "item_id"),
}


def _feature_columns(columns: list[str], group: str) -> list[str]:
    prefix = FEATURE_PREFIXES[group]

    def suffix_value(name: str) -> int:
        try:
            return int(name.rsplit("_", 1)[-1])
        except ValueError:
            return -1

    return sorted([col for col in columns if col.startswith(prefix)], key=suffix_value)


def _ids_file_name(source_name: str, id_col: str) -> str:
    return f"{source_name}_embedding_{id_col}s.npy"


def _sidecar_paths(source_dir: Path, group: str) -> tuple[str, Path, str, Path, str]:
    source_name, id_col = SIDECAR_SOURCES[group]
    values_name = f"{source_name}_embeddings.npy"
    ids_name = _ids_file_name(source_name, id_col)
    return values_name, source_dir / values_name, ids_name, source_dir / ids_name, id_col


def _eligible_groups(source_dir: Path, metadata: dict, requested_groups: list[str]) -> dict[str, dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}
    for group in requested_groups:
        if group not in FEATURE_PREFIXES or group not in SIDECAR_SOURCES:
            raise ValueError(f"Unsupported compact group: {group}")
        dim = int(metadata.get(f"{group}_dim", 0))
        if dim <= 0:
            continue
        values_name, values_path, ids_name, ids_path, id_col = _sidecar_paths(source_dir, group)
        if not values_path.exists() or not ids_path.exists():
            continue
        groups[group] = {
            "source": SIDECAR_SOURCES[group][0],
            "id_col": id_col,
            "values": values_name,
            "ids": ids_name,
            "dim": dim,
        }
    return groups


def _copy_support_files(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.name in {f"{split}.parquet" for split in SPLITS} or path.name == "metadata.json":
            continue
        target = output_dir / path.name
        if path.is_file():
            shutil.copy2(path, target)


def _compact_split(
    source_path: Path,
    output_path: Path,
    groups: dict[str, dict[str, object]],
    batch_size: int,
    compression: str,
) -> dict[str, int]:
    pf = pq.ParquetFile(source_path)
    schema_cols = pf.schema_arrow.names
    drop_cols: set[str] = set()
    dropped_by_group: dict[str, int] = {}
    for group in groups:
        cols = _feature_columns(schema_cols, group)
        drop_cols.update(cols)
        dropped_by_group[group] = len(cols)
    keep_cols = [col for col in schema_cols if col not in drop_cols]

    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for batch in pf.iter_batches(batch_size=batch_size, columns=keep_cols):
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression=compression)
            writer.write_table(table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        empty_schema = pa.schema([pf.schema_arrow.field(col) for col in keep_cols])
        arrays = [pa.array([], type=field.type) for field in empty_schema]
        pq.write_table(pa.Table.from_arrays(arrays, schema=empty_schema), output_path, compression=compression)

    return {
        "rows": rows,
        "original_columns": len(schema_cols),
        "kept_columns": len(keep_cols),
        "dropped_columns": len(drop_cols),
        **{f"dropped_{group}_columns": count for group, count in dropped_by_group.items()},
    }


def compact_processed_dir(
    processed_dir: Path,
    output_dir: Path,
    groups: list[str],
    batch_size: int = 65536,
    compression: str = "zstd",
) -> dict[str, object]:
    if processed_dir.resolve() == output_dir.resolve():
        raise ValueError("output_dir must be different from processed_dir")
    metadata_path = processed_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = _eligible_groups(processed_dir, metadata, groups)
    if not selected:
        raise ValueError("No eligible feature sidecars found to compact.")

    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_support_files(processed_dir, output_dir)

    split_summary: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        src = processed_dir / f"{split}.parquet"
        dst = output_dir / f"{split}.parquet"
        split_summary[split] = _compact_split(src, dst, selected, batch_size=batch_size, compression=compression)

    existing_sidecars = metadata.get("feature_sidecars", {})
    if not isinstance(existing_sidecars, dict):
        existing_sidecars = {}
    metadata["feature_sidecars"] = {**existing_sidecars, **selected}
    metadata["compact_source_dir"] = str(processed_dir)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "source_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "compacted_groups": sorted(selected),
        "feature_sidecars": selected,
        "splits": split_summary,
    }
    (output_dir / "compact_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop repeated embedding columns from processed parquet splits and use sidecar npy lookup at training time.")
    parser.add_argument("--processed-dir", default="data/run_latest/processed/qilin_v2")
    parser.add_argument("--output-dir", default="data/run_latest/processed/qilin_full_feature_opt_v2_compact")
    parser.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS), help="Feature groups to compact when a matching sidecar exists.")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--compression", default="zstd")
    args = parser.parse_args()

    summary = compact_processed_dir(
        processed_dir=Path(args.processed_dir),
        output_dir=Path(args.output_dir),
        groups=list(args.groups),
        batch_size=int(args.batch_size),
        compression=str(args.compression),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
