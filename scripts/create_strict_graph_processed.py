from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterointent.utils import read_json


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def create_strict_graph_processed(source_dir: Path, output_dir: Path) -> dict:
    source_dir = source_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_name = "graph_embedding.npy"

    for src in source_dir.iterdir():
        if src.name == graph_name or src.is_dir():
            continue
        link_or_copy(src, output_dir / src.name)

    metadata = read_json(source_dir / "metadata.json")
    graph = np.load(source_dir / graph_name)
    if graph.shape[0] != int(metadata["num_items"]):
        raise ValueError(f"Graph rows {graph.shape[0]} != num_items {metadata['num_items']}")

    train_items = pd.read_parquet(source_dir / "train.parquet", columns=["item_id"])["item_id"].astype(int).unique()
    mask = np.zeros(graph.shape[0], dtype=bool)
    mask[train_items] = True
    strict_graph = np.array(graph, copy=True)
    strict_graph[~mask] = 0.0
    np.save(output_dir / graph_name, strict_graph.astype("float32"))

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir.resolve()),
        "num_items": int(graph.shape[0]),
        "train_seen_items": int(mask.sum()),
        "zeroed_oov_items": int((~mask).sum()),
        "zeroed_oov_rate": float((~mask).mean()),
        "graph_embedding": graph_name,
    }
    (output_dir / "strict_graph_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a processed dir with graph rows zeroed for train-OOV items.")
    parser.add_argument("--source-dir", default="data/run_latest/processed/qilin_full_feature_opt_v2_compact")
    parser.add_argument("--output-dir", default="data/run_latest/processed/qilin_full_feature_opt_v2_compact_strict_graph_zero_oov")
    args = parser.parse_args()
    summary = create_strict_graph_processed(Path(args.source_dir), Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
