from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.build_text_embeddings import build_text_embeddings_for_processed
from scripts.build_visual_embeddings import build_visual_embeddings_for_processed
from scripts.compact_processed_features import DEFAULT_GROUPS, compact_processed_dir


def build_processed_compact(
    processed_dir: Path,
    output_dir: Path,
    *,
    qilin_dir: Path | None = None,
    groups: list[str] | None = None,
    build_text: bool = True,
    build_visual: bool = True,
    text_model_name: str = "BAAI/bge-small-zh-v1.5",
    text_batch_size: int = 256,
    text_max_length: int = 256,
    text_pooling: str = "cls",
    text_item_views: list[str] | None = None,
    encode_query: bool = True,
    text_device: str = "auto",
    image_root: Path | None = None,
    video_root: Path | None = None,
    visual_modalities: tuple[str, ...] = ("image", "video"),
    visual_model_name: str = "D:\\models\\siglip-base-patch16-224",
    visual_output_dim: int = 128,
    visual_compression: str = "pca",
    visual_batch_size: int = 64,
    visual_device: str = "auto",
    visual_cache_dir: Path | None = None,
    visual_fp16: bool = True,
    visual_image_workers: int = 4,
    visual_max_items: int = 0,
    visual_mock_encoder: bool = False,
    skip_existing_embeddings: bool = True,
    compact_batch_size: int = 65536,
    compact_compression: str = "zstd",
) -> dict[str, object]:
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    if processed_dir.resolve() == output_dir.resolve():
        raise ValueError("output_dir must differ from processed_dir")
    if not (processed_dir / "metadata.json").exists():
        raise FileNotFoundError(f"Missing processed metadata: {processed_dir / 'metadata.json'}")

    selected_groups = list(groups or DEFAULT_GROUPS)
    pipeline_summary: dict[str, object] = {
        "processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "groups": selected_groups,
    }

    if build_text:
        if qilin_dir is None:
            raise ValueError("qilin_dir is required when build_text=True")
        pipeline_summary["text"] = build_text_embeddings_for_processed(
            processed_dir,
            Path(qilin_dir),
            model_name=text_model_name,
            batch_size=text_batch_size,
            max_length=text_max_length,
            pooling=text_pooling,
            item_texts=text_item_views or ["joint", "title", "content"],
            encode_query=encode_query,
            device=text_device,
            skip_existing=skip_existing_embeddings,
            update_metadata=True,
        )

    if build_visual:
        if image_root is None and video_root is None:
            raise ValueError("image_root or video_root is required when build_visual=True")
        pipeline_summary["visual"] = build_visual_embeddings_for_processed(
            processed_dir,
            image_root=Path(image_root) if image_root else None,
            video_root=Path(video_root) if video_root else None,
            qilin_dir=Path(qilin_dir) if qilin_dir else None,
            modalities=visual_modalities,
            model_name=visual_model_name,
            output_dim=visual_output_dim,
            compression=visual_compression,
            batch_size=visual_batch_size,
            device=visual_device,
            cache_dir=visual_cache_dir,
            fp16=visual_fp16,
            image_workers=visual_image_workers,
            max_items=visual_max_items,
            mock_encoder=visual_mock_encoder,
            fallback_raw_id_paths=False,
            skip_existing=skip_existing_embeddings,
            update_metadata=True,
        )

    pipeline_summary["compact"] = compact_processed_dir(
        processed_dir=processed_dir,
        output_dir=output_dir,
        groups=selected_groups,
        batch_size=compact_batch_size,
        compression=compact_compression,
    )
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(pipeline_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return pipeline_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot pipeline: encode text/query + visual embeddings from paths on processed data, "
            "then emit compact parquet with mmap sidecars."
        )
    )
    parser.add_argument("--processed-dir", default="data/processed/qilin_full_feature_opt_v2")
    parser.add_argument("--output-dir", default="data/processed/qilin_full_feature_opt_v2_compact")
    parser.add_argument("--qilin-dir", default="data/raw/Qilin", help="Raw Qilin dir for notes/image_path.")
    parser.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS))

    parser.add_argument("--skip-text", action="store_true", help="Do not run text/query encoding.")
    parser.add_argument("--text-model-name", default="D:\\models\\bge-small-zh-v1.5")
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--text-max-length", type=int, default=256)
    parser.add_argument("--text-pooling", choices=["mean", "cls"], default="cls")
    parser.add_argument("--text-item-views", nargs="+", choices=["joint", "title", "content"], default=["joint", "title", "content"])
    parser.add_argument("--text-device", default="auto")

    parser.add_argument("--skip-visual", action="store_true", help="Do not run image/video encoding from local paths.")
    parser.add_argument("--image-root", default=None, help="Local image root, e.g. E:\\qilin\\...\\image")
    parser.add_argument("--video-root", default=None, help="Optional local video thumbnail root.")
    parser.add_argument("--visual-modality", choices=["image", "video", "both"], default="both")
    parser.add_argument("--visual-model-name", default="D:\\models\\siglip-base-patch16-224")
    parser.add_argument("--visual-output-dim", type=int, default=128)
    parser.add_argument("--visual-compression", choices=["auto", "pca", "random", "none"], default="pca")
    parser.add_argument("--visual-batch-size", type=int, default=64)
    parser.add_argument("--visual-device", default="auto")
    parser.add_argument("--visual-cache-dir", default="data/processed/visual_path_cache_siglip")
    parser.add_argument("--visual-fp16", action="store_true", default=True)
    parser.add_argument("--no-visual-fp16", action="store_false", dest="visual_fp16")
    parser.add_argument("--visual-image-workers", type=int, default=4)
    parser.add_argument("--visual-max-items", type=int, default=0)
    parser.add_argument("--mock-visual", action="store_true", help="Deterministic mock vectors for smoke tests only.")

    parser.add_argument("--force-reencode", action="store_true", help="Rebuild embeddings even when sidecar npy already exists.")
    parser.add_argument("--compact-batch-size", type=int, default=65536)
    parser.add_argument("--compact-compression", default="zstd")
    args = parser.parse_args()

    modalities = ("image", "video") if args.visual_modality == "both" else (args.visual_modality,)
    summary = build_processed_compact(
        processed_dir=Path(args.processed_dir),
        output_dir=Path(args.output_dir),
        qilin_dir=Path(args.qilin_dir) if args.qilin_dir else None,
        groups=list(args.groups),
        build_text=not args.skip_text,
        build_visual=not args.skip_visual,
        text_model_name=str(args.text_model_name),
        text_batch_size=int(args.text_batch_size),
        text_max_length=int(args.text_max_length),
        text_pooling=str(args.text_pooling),
        text_item_views=list(args.text_item_views),
        encode_query=not args.no_query,
        text_device=str(args.text_device),
        image_root=Path(args.image_root) if args.image_root else None,
        video_root=Path(args.video_root) if args.video_root else None,
        visual_modalities=modalities,
        visual_model_name=str(args.visual_model_name),
        visual_output_dim=int(args.visual_output_dim),
        visual_compression=str(args.visual_compression),
        visual_batch_size=int(args.visual_batch_size),
        visual_device=str(args.visual_device),
        visual_cache_dir=Path(args.visual_cache_dir) if args.visual_cache_dir else None,
        visual_fp16=bool(args.visual_fp16),
        visual_image_workers=int(args.visual_image_workers),
        visual_max_items=int(args.visual_max_items),
        visual_mock_encoder=bool(args.mock_visual),
        skip_existing_embeddings=not args.force_reencode,
        compact_batch_size=int(args.compact_batch_size),
        compact_compression=str(args.compact_compression),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
