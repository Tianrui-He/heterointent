from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from tqdm import tqdm


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
PATH_PREFIXES = {"image", "images", "video", "videos", "thumbnail", "thumbnails"}


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist() if str(v)]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v)]
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return []
        if text.startswith("[") and text.endswith("]"):
            return [part.strip().strip("'\"") for part in text[1:-1].split(",") if part.strip()]
        return [text]
    return []


def _read_notes(qilin_dir: Path | None) -> pd.DataFrame:
    if qilin_dir is None:
        return pd.DataFrame(columns=["note_idx", "image_path", "note_type"])
    notes_dir = qilin_dir / "notes"
    if not notes_dir.exists():
        return pd.DataFrame(columns=["note_idx", "image_path", "note_type"])
    frames: list[pd.DataFrame] = []
    wanted = ["note_idx", "image_path", "note_type"]
    for file in sorted(notes_dir.glob("*.parquet")):
        schema_cols = pq.ParquetFile(file).schema.names
        cols = [col for col in wanted if col in schema_cols]
        if "note_idx" in cols and "image_path" not in cols:
            cols = [*cols, "image_path"]
        if "note_idx" in cols:
            try:
                frames.append(pd.read_parquet(file, columns=cols))
            except Exception:
                fallback_cols = [col for col in cols if col in schema_cols]
                frames.append(pd.read_parquet(file, columns=fallback_cols))
    if not frames:
        return pd.DataFrame(columns=wanted)
    notes = pd.concat(frames, ignore_index=True)
    for col in wanted:
        if col not in notes:
            notes[col] = None
    return notes[wanted]


def _candidate_paths(root: Path | None, raw_item_id: int, image_paths: object) -> Iterable[Path]:
    if root is None:
        return []
    candidates: list[Path] = []
    for raw in _as_list(image_paths):
        rel = Path(str(raw).replace("\\", "/"))
        candidates.extend([root / rel, root / rel.name])
        parts = rel.parts
        if parts and parts[0].lower() in PATH_PREFIXES and len(parts) > 1:
            candidates.append(root.joinpath(*parts[1:]))
    for suffix in IMAGE_SUFFIXES:
        candidates.append(root / f"{raw_item_id}{suffix}")
    return candidates


def _resolve_existing_paths(root: Path | None, raw_item_id: int, image_paths: object, max_paths: int) -> list[Path]:
    seen: set[str] = set()
    found: list[Path] = []
    for candidate in _candidate_paths(root, raw_item_id, image_paths):
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            found.append(candidate)
            if len(found) >= max_paths:
                break
    return found


def load_visual_index(
    processed_dir: Path,
    qilin_dir: Path | None,
    image_root: Path | None,
    video_root: Path | None,
    modality: str,
    max_images_per_item: int,
) -> pd.DataFrame:
    item_map = pd.read_parquet(processed_dir / "item_id_map.parquet").sort_values("item_id")
    notes = _read_notes(qilin_dir)
    if not notes.empty:
        notes = item_map.merge(notes, left_on="raw_item_id", right_on="note_idx", how="left")
    else:
        notes = item_map.copy()
        notes["image_path"] = None
        notes["note_type"] = 0

    rows: list[dict[str, object]] = []
    for row in notes.itertuples(index=False):
        raw_item_id = int(getattr(row, "raw_item_id"))
        item_id = int(getattr(row, "item_id"))
        note_type = int(float(getattr(row, "note_type", 0) or 0))
        raw_paths = getattr(row, "image_path", None)

        if modality == "image":
            paths = _resolve_existing_paths(image_root, raw_item_id, raw_paths, max_images_per_item)
            source = "image_path" if paths else "missing"
        else:
            paths = _resolve_existing_paths(video_root, raw_item_id, raw_paths, max_images_per_item)
            source = "video_root"
            if not paths and note_type == 2:
                paths = _resolve_existing_paths(image_root, raw_item_id, raw_paths, max_images_per_item)
                source = "image_cover" if paths else "missing"
            elif not paths:
                source = "missing"

        path_strings = [str(path) for path in paths]
        rows.append(
            {
                "item_id": item_id,
                "raw_item_id": raw_item_id,
                "note_type": note_type,
                "modality": modality,
                "path": path_strings[0] if path_strings else "",
                "paths_json": json.dumps(path_strings, ensure_ascii=False),
                "path_count": len(path_strings),
                "source": source,
                "status": "found" if path_strings else "missing",
            }
        )
    return pd.DataFrame(rows)


def _stable_vector(key: str, dim: int) -> np.ndarray:
    chunks = []
    counter = 0
    while len(chunks) * 32 < dim:
        digest = hashlib.sha256(f"{key}:{counter}".encode("utf-8")).digest()
        chunks.append(np.frombuffer(digest, dtype=np.uint8).astype("float32"))
        counter += 1
    values = np.concatenate(chunks)[:dim]
    values = values / 127.5 - 1.0
    norm = np.linalg.norm(values)
    return values / norm if norm > 0 else values


def _mock_encode(items: pd.DataFrame, dim: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(items), dim), dtype="float32")
    counts = np.zeros(len(items), dtype="int32")
    for idx, row in enumerate(items.itertuples(index=False)):
        paths = json.loads(getattr(row, "paths_json") or "[]")
        if not paths:
            continue
        vecs = [_stable_vector(str(path), dim) for path in paths]
        values[idx] = np.mean(vecs, axis=0).astype("float32")
        counts[idx] = len(vecs)
    return values, counts


def _image_stats(path: Path, dim: int) -> np.ndarray:
    from PIL import Image, ImageStat

    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((64, 64))
    arr = np.asarray(rgb, dtype="float32") / 255.0
    stat = ImageStat.Stat(rgb)
    mean = np.asarray(stat.mean, dtype="float32") / 255.0
    std = np.asarray(stat.stddev, dtype="float32") / 255.0
    mins = arr.reshape(-1, 3).min(axis=0)
    maxs = arr.reshape(-1, 3).max(axis=0)
    gray = arr.mean(axis=2)
    grad_x = np.abs(np.diff(gray, axis=1)).mean()
    grad_y = np.abs(np.diff(gray, axis=0)).mean()
    hist_parts = [
        np.histogram(arr[:, :, channel], bins=16, range=(0.0, 1.0), density=True)[0].astype("float32")
        for channel in range(3)
    ]
    hist = np.concatenate(hist_parts)
    hist = hist / max(float(hist.sum()), 1e-12)
    values = np.concatenate(
        [
            mean,
            std,
            mins,
            maxs,
            np.asarray([float(gray.mean()), float(gray.std()), float(grad_x), float(grad_y)], dtype="float32"),
            hist,
        ]
    ).astype("float32")
    if values.shape[0] < dim:
        values = np.pad(values, (0, dim - values.shape[0]))
    return values[:dim]


def _handcrafted_encode(items: pd.DataFrame, dim: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(items), dim), dtype="float32")
    counts = np.zeros(len(items), dtype="int32")
    for idx, row in enumerate(tqdm(items.itertuples(index=False), total=len(items), desc="handcrafted visual features")):
        paths = [Path(path) for path in json.loads(getattr(row, "paths_json") or "[]")]
        vecs = []
        for path in paths:
            try:
                vecs.append(_image_stats(path, dim))
            except Exception:
                continue
        if vecs:
            values[idx] = np.mean(vecs, axis=0).astype("float32")
            counts[idx] = len(vecs)
    return _normalize_rows(values), counts


def _model_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _encode_with_transformers(
    items: pd.DataFrame,
    model_name: str,
    batch_size: int,
    device_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise SystemExit("Please install transformers and Pillow first: pip install transformers Pillow") from exc

    flat: list[tuple[int, Path]] = []
    for idx, row in enumerate(items.itertuples(index=False)):
        for path in json.loads(getattr(row, "paths_json") or "[]"):
            flat.append((idx, Path(path)))
    if not flat:
        return np.zeros((len(items), 0), dtype="float32"), np.zeros(len(items), dtype="int32")

    device = _model_device(device_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    sums: np.ndarray | None = None
    counts = np.zeros(len(items), dtype="int32")
    with torch.no_grad():
        for start in tqdm(range(0, len(flat), batch_size), desc="visual embeddings"):
            batch = flat[start : start + batch_size]
            images = []
            good_indices: list[int] = []
            for item_idx, path in batch:
                try:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                    good_indices.append(item_idx)
                except Exception:
                    continue
            if not images:
                continue
            inputs = processor(images=images, return_tensors="pt").to(device)
            if hasattr(model, "get_image_features"):
                emb = model.get_image_features(**inputs)
            else:
                out = model(**inputs)
                emb = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
            if not isinstance(emb, torch.Tensor):
                emb = emb.pooler_output if getattr(emb, "pooler_output", None) is not None else emb.last_hidden_state[:, 0]
            emb = torch.nn.functional.normalize(emb, dim=-1).cpu().numpy().astype("float32")
            if sums is None:
                sums = np.zeros((len(items), emb.shape[1]), dtype="float32")
            for item_idx, vec in zip(good_indices, emb):
                sums[item_idx] += vec
                counts[item_idx] += 1
    if sums is None:
        return np.zeros((len(items), 0), dtype="float32"), counts
    nonzero = counts > 0
    sums[nonzero] = sums[nonzero] / counts[nonzero, None]
    return _normalize_rows(sums), counts


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype("float32", copy=False)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, np.maximum(norms, 1e-12), out=np.zeros_like(values, dtype="float32"), where=norms > 0)


def compress_embeddings(values: np.ndarray, output_dim: int, method: str, random_state: int) -> tuple[np.ndarray, str]:
    if values.size == 0:
        return values.astype("float32", copy=False), "empty"
    raw_dim = int(values.shape[1])
    if method == "none" or output_dim <= 0 or output_dim >= raw_dim:
        return _normalize_rows(values.astype("float32", copy=False)), "none"

    nonzero = np.linalg.norm(values, axis=1) > 0
    compressed = np.zeros((values.shape[0], output_dim), dtype="float32")
    fit_values = values[nonzero].astype("float32", copy=False)
    used = "random_projection"
    if method in {"auto", "pca"} and fit_values.shape[0] > output_dim:
        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=output_dim, svd_solver="randomized", random_state=random_state)
            compressed[nonzero] = pca.fit_transform(fit_values).astype("float32", copy=False)
            used = "pca"
        except Exception:
            if method == "pca":
                raise
    if used == "random_projection":
        rng = np.random.default_rng(random_state)
        matrix = rng.standard_normal((raw_dim, output_dim)).astype("float32") / math.sqrt(output_dim)
        compressed[nonzero] = fit_values @ matrix
    return _normalize_rows(compressed), used


def build_embeddings_for_modality(args: argparse.Namespace, modality: str) -> dict[str, object]:
    processed_dir = Path(args.processed_dir)
    image_root = Path(args.image_root) if args.image_root else None
    video_root = Path(args.video_root) if args.video_root else None
    qilin_dir = Path(args.qilin_dir) if args.qilin_dir else None

    items = load_visual_index(
        processed_dir=processed_dir,
        qilin_dir=qilin_dir,
        image_root=image_root,
        video_root=video_root,
        modality=modality,
        max_images_per_item=int(args.max_images_per_item),
    )
    if args.max_items and int(args.max_items) > 0:
        items = items.head(int(args.max_items)).copy()

    encoder_name = str(getattr(args, "encoder", "transformers"))
    if encoder_name == "mock" or bool(getattr(args, "mock_encoder", False)):
        raw_values, encoded_counts = _mock_encode(items, dim=int(args.mock_dim))
        encoder_name = "mock"
    elif encoder_name == "handcrafted":
        raw_values, encoded_counts = _handcrafted_encode(items, dim=int(getattr(args, "handcrafted_dim", 64)))
    else:
        raw_values, encoded_counts = _encode_with_transformers(
            items,
            model_name=str(args.model_name),
            batch_size=int(args.batch_size),
            device_name=str(args.device),
        )

    if raw_values.shape[1] == 0:
        raw_values = np.zeros((len(items), int(args.output_dim)), dtype="float32")
        compression = "empty"
    else:
        raw_values = _normalize_rows(raw_values)
        raw_values[encoded_counts == 0] = 0.0
        raw_values, compression = compress_embeddings(
            raw_values,
            output_dim=int(args.output_dim),
            method=str(args.compression),
            random_state=int(args.seed),
        )
        raw_values[encoded_counts == 0] = 0.0

    items = items.copy()
    items["encoded_path_count"] = encoded_counts.astype("int32")
    items.loc[items["path_count"].astype(int).eq(0), "status"] = "missing"
    items.loc[items["path_count"].astype(int).gt(0) & items["encoded_path_count"].eq(0), "status"] = "error"
    items.loc[items["encoded_path_count"].gt(0), "status"] = "encoded"
    items["embedding_dim"] = int(raw_values.shape[1])
    items["compression"] = compression

    dtype = np.float16 if str(args.save_dtype) == "float16" else np.float32
    np.save(processed_dir / f"{modality}_embeddings.npy", raw_values.astype(dtype, copy=False))
    np.save(processed_dir / f"{modality}_embedding_item_ids.npy", items["item_id"].to_numpy(dtype="int64"))
    items.to_parquet(processed_dir / f"{modality}_embedding_items.parquet", index=False)

    summary = {
        "modality": modality,
        "items": int(len(items)),
        "found_items": int(items["path_count"].astype(int).gt(0).sum()),
        "encoded_items": int(items["encoded_path_count"].astype(int).gt(0).sum()),
        "missing_items": int(items["status"].eq("missing").sum()),
        "error_items": int(items["status"].eq("error").sum()),
        "embedding_dim": int(raw_values.shape[1]),
        "compression": compression,
        "save_dtype": str(args.save_dtype),
        "model_name": encoder_name if encoder_name != "transformers" else str(args.model_name),
    }
    (processed_dir / f"{modality}_embedding_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"wrote {modality} embeddings {tuple(raw_values.shape)} "
        f"encoded={summary['encoded_items']} missing={summary['missing_items']} to {processed_dir}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build parameter-friendly image/video thumbnail embeddings for processed Qilin items."
    )
    parser.add_argument("--modality", choices=["image", "video", "both"], default="image")
    parser.add_argument("--image-root", default=None, help="Directory containing Qilin images or image covers.")
    parser.add_argument("--video-root", default=None, help="Directory containing local video thumbnails/covers.")
    parser.add_argument("--qilin-dir", default=None, help="Optional raw Qilin directory. notes/image_path is used when present.")
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "processed" / "qilin_full"))
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--encoder",
        choices=["transformers", "handcrafted", "mock"],
        default="transformers",
        help="transformers uses CLIP/SigLIP; handcrafted uses local PIL image statistics; mock is for tests only.",
    )
    parser.add_argument("--output-dim", type=int, default=128, help="Compressed embedding dimension; <=0 keeps raw dimension.")
    parser.add_argument("--compression", choices=["auto", "pca", "random", "none"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images-per-item", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=0, help="Optional debug limit.")
    parser.add_argument("--save-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mock-encoder", action="store_true", help="Use deterministic hash vectors for tests/smoke runs.")
    parser.add_argument("--mock-dim", type=int, default=512)
    parser.add_argument("--handcrafted-dim", type=int, default=64)
    args = parser.parse_args()

    modalities = ["image", "video"] if args.modality == "both" else [args.modality]
    summaries = [build_embeddings_for_modality(args, modality) for modality in modalities]
    processed_dir = Path(args.processed_dir)
    (processed_dir / "visual_embedding_summary.json").write_text(
        json.dumps({"modalities": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
