"""
Deprecated: prefer scripts/build_processed_compact.py, which encodes visual vectors
directly from image paths and compacts in one step. This copy helper remains for
reusing sidecars produced elsewhere.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


VISUAL_FILES = [
    "image_embeddings.npy",
    "image_embedding_item_ids.npy",
    "image_embedding_items.parquet",
    "image_embedding_summary.json",
    "video_embeddings.npy",
    "video_embedding_item_ids.npy",
    "video_embedding_items.parquet",
    "video_embedding_summary.json",
    "visual_embedding_summary.json",
]


def attach_visual_sidecars(
    target_dir: Path,
    source_dir: Path,
    image_emb_dim: int = 128,
    video_emb_dim: int = 128,
) -> dict:
    target_dir = Path(target_dir)
    source_dir = Path(source_dir)
    if not (source_dir / "image_embeddings.npy").exists():
        raise FileNotFoundError(f"Missing image embeddings in {source_dir}")

    copied: list[str] = []
    for name in VISUAL_FILES:
        src = source_dir / name
        if not src.exists():
            continue
        dst = target_dir / name
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)
        copied.append(name)

    meta_path = target_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["image_emb_dim"] = int(image_emb_dim)
    metadata["video_emb_dim"] = int(video_emb_dim)
    sidecars = metadata.get("feature_sidecars", {})
    if not isinstance(sidecars, dict):
        sidecars = {}
    sidecars["image_emb"] = {
        "source": "image",
        "id_col": "item_id",
        "values": "image_embeddings.npy",
        "ids": "image_embedding_item_ids.npy",
        "dim": int(image_emb_dim),
    }
    sidecars["video_emb"] = {
        "source": "video",
        "id_col": "item_id",
        "values": "video_embeddings.npy",
        "ids": "video_embedding_item_ids.npy",
        "dim": int(video_emb_dim),
    }
    metadata["feature_sidecars"] = sidecars
    metadata["visual_sidecar_source"] = str(source_dir)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "target_dir": str(target_dir),
        "source_dir": str(source_dir),
        "copied_files": copied,
        "image_emb_dim": int(image_emb_dim),
        "video_emb_dim": int(video_emb_dim),
    }
    (target_dir / "visual_sidecar_attach.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach precomputed visual embedding sidecars without re-encoding text.")
    parser.add_argument("--target-dir", default="data/processed/qilin_full_feature_opt_v2_compact")
    parser.add_argument("--source-dir", default="data/processed/qilin_full_multimodal_siglip_v2")
    parser.add_argument("--image-emb-dim", type=int, default=128)
    parser.add_argument("--video-emb-dim", type=int, default=128)
    args = parser.parse_args()
    summary = attach_visual_sidecars(
        target_dir=Path(args.target_dir),
        source_dir=Path(args.source_dir),
        image_emb_dim=int(args.image_emb_dim),
        video_emb_dim=int(args.video_emb_dim),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
