from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterointent.data.graph import smooth_item_features
from heterointent.utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen LightGCN/FREEDOM-style item graph embeddings.")
    parser.add_argument("--processed-dir", default="data/processed/qilin")
    parser.add_argument("--samples-file", default="train.parquet")
    parser.add_argument("--metadata-file", default="metadata.json")
    parser.add_argument("--output-file", default="graph_embedding.npy")
    parser.add_argument("--embed-dim", type=int, default=64)
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    metadata = read_json(processed / args.metadata_file)
    out = processed / args.output_file
    emb = smooth_item_features(
        samples_path=processed / args.samples_file,
        output_path=out,
        num_items=int(metadata["num_items"]),
        embed_dim=args.embed_dim,
    )
    print(f"wrote graph embeddings {emb.shape} to {out}")


if __name__ == "__main__":
    main()
