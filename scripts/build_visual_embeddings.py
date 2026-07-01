from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
PATH_PREFIXES = {"image", "images", "thumbnail", "thumbnails"}
CACHE_INDEX_NAME = "path_embedding_index.parquet"
CACHE_VALUES_NAME = "path_embeddings.npy"
CACHE_META_NAME = "path_embedding_cache.json"


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


def _available_child_dirs(root: Path | None) -> set[str]:
    if root is None or not root.exists():
        return set()
    return {path.name.lower() for path in root.iterdir() if path.is_dir()}


def _part_number_from_path(path: Path | str) -> int | None:
    for part in Path(str(path).replace("\\", "/")).parts:
        lower = part.lower()
        if lower.startswith("part_") and lower[5:].isdigit():
            return int(lower[5:])
    return None


def _part_allowed(path: Path | str, part_min: int | None = None, part_max: int | None = None) -> bool:
    if part_min is None and part_max is None:
        return True
    part = _part_number_from_path(path)
    if part is None:
        return False
    if part_min is not None and part < int(part_min):
        return False
    if part_max is not None and part > int(part_max):
        return False
    return True


def _candidate_paths(
    root: Path | None,
    raw_item_id: int,
    image_paths: object,
    available_dirs: set[str] | None = None,
    fallback_raw_id_paths: bool = True,
) -> Iterable[tuple[Path, bool]]:
    if root is None:
        return []
    available = available_dirs or set()
    candidates: list[tuple[Path, bool]] = []
    for raw in _as_list(image_paths):
        rel = Path(str(raw).replace("\\", "/"))
        if rel.is_absolute():
            candidates.append((rel, False))
            continue
        parts = rel.parts
        if parts and parts[0].lower() in PATH_PREFIXES and len(parts) > 1:
            if parts[1].lower() in available:
                candidates.append((root.joinpath(*parts[1:]), True))
        else:
            if parts and parts[0].lower() in available:
                candidates.append((root / rel, True))
            if len(parts) <= 1:
                candidates.append((root / rel.name, False))
    if fallback_raw_id_paths:
        for suffix in IMAGE_SUFFIXES:
            candidates.append((root / f"{raw_item_id}{suffix}", False))
    return candidates


def _resolve_existing_paths(
    root: Path | None,
    raw_item_id: int,
    image_paths: object,
    max_paths: int,
    available_dirs: set[str] | None = None,
    fallback_raw_id_paths: bool = True,
    part_min: int | None = None,
    part_max: int | None = None,
) -> list[Path]:
    seen: set[str] = set()
    found: list[Path] = []
    for candidate, trusted_parent in _candidate_paths(
        root,
        raw_item_id,
        image_paths,
        available_dirs=available_dirs,
        fallback_raw_id_paths=fallback_raw_id_paths,
    ):
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if not _part_allowed(candidate, part_min=part_min, part_max=part_max):
            continue
        if trusted_parent or candidate.is_file():
            found.append(candidate)
            if len(found) >= max_paths:
                break
    return found


def load_visual_index(
    processed_dir: Path,
    qilin_dir: Path | None,
    image_root: Path | None,
    max_images_per_item: int,
    max_items: int = 0,
    fallback_raw_id_paths: bool = True,
    image_part_min: int | None = None,
    image_part_max: int | None = None,
) -> pd.DataFrame:
    item_map = pd.read_parquet(processed_dir / "item_id_map.parquet").sort_values("item_id")
    if int(max_items) > 0:
        item_map = item_map.head(int(max_items)).copy()
    notes = _read_notes(qilin_dir)
    if not notes.empty:
        notes = item_map.merge(notes, left_on="raw_item_id", right_on="note_idx", how="left")
    else:
        notes = item_map.copy()
        notes["image_path"] = None
        notes["note_type"] = 0
    image_dirs = _available_child_dirs(image_root)

    rows: list[dict[str, object]] = []
    for row in notes.itertuples(index=False):
        raw_item_id = int(getattr(row, "raw_item_id"))
        item_id = int(getattr(row, "item_id"))
        note_type = int(float(getattr(row, "note_type", 0) or 0))
        raw_paths = getattr(row, "image_path", None)

        paths = _resolve_existing_paths(
            image_root,
            raw_item_id,
            raw_paths,
            max_images_per_item,
            available_dirs=image_dirs,
            fallback_raw_id_paths=fallback_raw_id_paths,
            part_min=image_part_min,
            part_max=image_part_max,
        )
        source = "image_path" if paths else "missing"

        path_strings = [str(path) for path in paths]
        rows.append(
            {
                "item_id": item_id,
                "raw_item_id": raw_item_id,
                "note_type": note_type,
                "modality": "image",
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


class _ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = [str(path) for path in paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[int, np.ndarray, bool]:
        from PIL import Image

        try:
            with Image.open(self.paths[idx]) as image:
                arr = np.asarray(image.convert("RGB"))
            return idx, arr, True
        except Exception:
            return idx, np.empty((0, 0, 3), dtype=np.uint8), False


def _collate_image_batch(batch: list[tuple[int, np.ndarray, bool]]) -> tuple[list[int], list[np.ndarray]]:
    indices: list[int] = []
    images: list[np.ndarray] = []
    for idx, image, ok in batch:
        if ok and image.size:
            indices.append(int(idx))
            images.append(image)
    return indices, images


def _load_image_array(task: tuple[int, str, int]) -> tuple[int, np.ndarray | None]:
    idx, path, image_size = task
    try:
        try:
            import cv2

            arr = cv2.imread(path, cv2.IMREAD_COLOR)
            if arr is not None:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                if image_size > 0:
                    arr = cv2.resize(arr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                return idx, arr
        except ImportError:
            pass
        from PIL import Image, ImageOps

        with Image.open(path) as image:
            if image_size > 0:
                try:
                    image.draft("RGB", (image_size, image_size))
                except Exception:
                    pass
                image = ImageOps.exif_transpose(image).convert("RGB").resize(
                    (image_size, image_size),
                    Image.Resampling.BILINEAR,
                )
            else:
                image = ImageOps.exif_transpose(image).convert("RGB")
            return idx, np.asarray(image).copy()
    except Exception:
        return idx, None


def _path_signature(path: Path) -> dict[str, object]:
    # Qilin image shards are append-only in this workflow. Using the path as
    # the stable cache key avoids a slow per-image stat pass on external disks.
    return {"path": str(path), "size": -1, "mtime_ns": -1}


def _paths_from_items(items: pd.DataFrame) -> tuple[list[Path], dict[str, list[int]]]:
    path_to_items: dict[str, list[int]] = {}
    for item_idx, row in enumerate(items.itertuples(index=False)):
        seen_for_item: set[str] = set()
        for raw_path in json.loads(getattr(row, "paths_json") or "[]"):
            path = str(Path(raw_path))
            key = path.lower()
            if key in seen_for_item:
                continue
            seen_for_item.add(key)
            path_to_items.setdefault(path, []).append(item_idx)

    def sort_key(path: str) -> tuple[int, str]:
        part = _part_number_from_path(path)
        return (part if part is not None else 10**9, path.lower())

    paths = [Path(path) for path in sorted(path_to_items, key=sort_key)]
    return paths, path_to_items


def _part_counts_from_items(items: pd.DataFrame, encoded_only: bool = False) -> dict[str, int]:
    counts: dict[int, int] = {}
    for row in items.itertuples(index=False):
        if encoded_only and int(getattr(row, "encoded_path_count", 0) or 0) <= 0:
            continue
        for raw_path in json.loads(getattr(row, "paths_json") or "[]"):
            part = _part_number_from_path(raw_path)
            if part is not None:
                counts[part] = counts.get(part, 0) + 1
    return {str(part): int(counts[part]) for part in sorted(counts)}


def _part_range(part_counts: dict[str, int]) -> tuple[int | None, int | None]:
    if not part_counts:
        return None, None
    parts = [int(part) for part in part_counts]
    return min(parts), max(parts)


def _load_path_cache(cache_dir: Path | None, model_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    if cache_dir is None:
        return pd.DataFrame(columns=["path", "size", "mtime_ns", "row"]), np.zeros((0, 0), dtype="float32")
    index_path = cache_dir / CACHE_INDEX_NAME
    values_path = cache_dir / CACHE_VALUES_NAME
    meta_path = cache_dir / CACHE_META_NAME
    if not index_path.exists() or not values_path.exists():
        return pd.DataFrame(columns=["path", "size", "mtime_ns", "row"]), np.zeros((0, 0), dtype="float32")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cached_model = str(meta.get("model_name", ""))
        if cached_model and cached_model != str(model_name):
            raise ValueError(f"Visual cache model mismatch: {cached_model!r} != {model_name!r}")
    index = pd.read_parquet(index_path)
    values = np.load(values_path)
    if "row" not in index:
        index = index.copy()
        index["row"] = np.arange(len(index), dtype=np.int64)
    if len(index) != int(values.shape[0]):
        raise ValueError(f"Visual cache index/value mismatch: {len(index)} rows vs {values.shape[0]} embeddings")
    for col in ["path", "size", "mtime_ns", "row"]:
        if col not in index:
            raise ValueError(f"Visual cache index missing column: {col}")
    return index[["path", "size", "mtime_ns", "row"]].copy(), values


def _save_path_cache(
    cache_dir: Path,
    index: pd.DataFrame,
    values: np.ndarray,
    model_name: str,
    save_dtype: str,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if save_dtype == "float16" else np.float32
    index = index[["path", "size", "mtime_ns", "row"]].copy()
    index_tmp = cache_dir / f"{CACHE_INDEX_NAME}.tmp"
    values_tmp = cache_dir / f"{CACHE_VALUES_NAME}.tmp.npy"
    meta_tmp = cache_dir / f"{CACHE_META_NAME}.tmp"
    index.to_parquet(index_tmp, index=False)
    np.save(values_tmp, values.astype(dtype, copy=False))
    meta = {
        "model_name": str(model_name),
        "embedding_dim": int(values.shape[1]) if values.ndim == 2 and values.size else 0,
        "num_paths": int(values.shape[0]) if values.ndim == 2 else 0,
        "save_dtype": save_dtype,
    }
    meta_tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    index_tmp.replace(cache_dir / CACHE_INDEX_NAME)
    values_tmp.replace(cache_dir / CACHE_VALUES_NAME)
    meta_tmp.replace(cache_dir / CACHE_META_NAME)


def _embedding_from_output(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if getattr(output, "pooler_output", None) is not None:
        return output.pooler_output
    return output.last_hidden_state[:, 0]


def _processor_value(processor: object, name: str, default: object) -> object:
    value = getattr(processor, name, None)
    if value is None and hasattr(processor, "image_processor"):
        value = getattr(getattr(processor, "image_processor"), name, None)
    return default if value is None else value


def _fast_preprocess_image_arrays(images: list[np.ndarray], processor: object) -> dict[str, torch.Tensor]:
    """Vectorized preprocessing for RGB arrays already resized by _load_image_array."""

    arr = np.stack(images, axis=0).astype("float32", copy=False)
    if bool(_processor_value(processor, "do_rescale", True)):
        arr *= float(_processor_value(processor, "rescale_factor", 1.0 / 255.0))
    mean = np.asarray(_processor_value(processor, "image_mean", [0.5, 0.5, 0.5]), dtype="float32")
    std = np.asarray(_processor_value(processor, "image_std", [0.5, 0.5, 0.5]), dtype="float32")
    arr = (arr - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3)
    arr = np.ascontiguousarray(arr.transpose(0, 3, 1, 2))
    return {"pixel_values": torch.from_numpy(arr)}


def _forward_images_adaptive(
    images: list[np.ndarray],
    processor: object,
    model: torch.nn.Module,
    device: torch.device,
    fp16: bool,
    *,
    preprocessed: dict[str, torch.Tensor] | None = None,
) -> np.ndarray:
    try:
        if preprocessed is None:
            inputs = processor(images=images, return_tensors="pt", do_resize=False, do_center_crop=False)
        else:
            inputs = preprocessed
        if device.type == "cuda":
            inputs = {key: value.pin_memory().to(device, non_blocking=True) for key, value in inputs.items()}
        else:
            if isinstance(inputs, dict):
                inputs = {key: value.to(device) for key, value in inputs.items()}
            else:
                inputs = inputs.to(device)
        use_fp16 = bool(fp16 and device.type == "cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            if hasattr(model, "get_image_features"):
                output = model.get_image_features(**inputs)
            else:
                output = model(**inputs)
            emb = _embedding_from_output(output)
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.detach().cpu().numpy().astype("float32")
    except RuntimeError as exc:
        is_oom = "out of memory" in str(exc).lower()
        if not is_oom or len(images) <= 1:
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        midpoint = len(images) // 2
        left = _forward_images_adaptive(images[:midpoint], processor, model, device, fp16)
        right = _forward_images_adaptive(images[midpoint:], processor, model, device, fp16)
        return np.concatenate([left, right], axis=0)


def _load_image_batch(
    batch_paths: list[Path],
    start_offset: int,
    image_size: int,
    executor: ThreadPoolExecutor | None,
) -> tuple[list[int], list[np.ndarray]]:
    tasks = [(start_offset + offset, str(path), int(image_size)) for offset, path in enumerate(batch_paths)]
    if executor is not None:
        loaded = list(executor.map(_load_image_array, tasks))
    else:
        loaded = [_load_image_array(task) for task in tasks]
    batch_indices = [idx for idx, image in loaded if image is not None]
    images = [image for _, image in loaded if image is not None]
    return batch_indices, images


def _encode_paths_with_transformers(
    paths: list[Path],
    model_name: str,
    batch_size: int,
    device_name: str,
    fp16: bool,
    image_workers: int,
    image_size: int,
    prefetch_batches: int = 2,
    fast_preprocess: bool = True,
    *,
    processor: object | None = None,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
) -> tuple[list[int], np.ndarray]:
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise SystemExit("Please install transformers and Pillow first: pip install transformers Pillow") from exc

    if not paths:
        return [], np.zeros((0, 0), dtype="float32")

    owns_model = processor is None or model is None
    if device is None:
        device = _model_device(device_name)
    if owns_model:
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

    batch_size_int = max(int(batch_size), 1)
    workers = max(int(image_workers), 0)
    batch_starts = list(range(0, len(paths), batch_size_int))
    print(
        "visual encoder "
        f"paths={len(paths)} batches={len(batch_starts)} batch_size={batch_size_int} "
        f"workers={workers} prefetch={int(prefetch_batches)} fp16={bool(fp16)} "
        f"fast_preprocess={bool(fast_preprocess)} device={device}"
    )
    if int(prefetch_batches) <= 0:
        executor = ThreadPoolExecutor(max_workers=workers) if workers > 0 else None
        good_path_indices: list[int] = []
        chunks: list[np.ndarray] = []
        try:
            for start in tqdm(batch_starts, desc="visual path embeddings"):
                batch_paths = paths[start : start + batch_size_int]
                batch_indices, images = _load_image_batch(batch_paths, start, image_size, executor)
                if not images:
                    continue
                preprocessed = _fast_preprocess_image_arrays(images, processor) if fast_preprocess else None
                emb = _forward_images_adaptive(
                    images=images,
                    processor=processor,
                    model=model,
                    device=device,
                    fp16=fp16,
                    preprocessed=preprocessed,
                )
                good_path_indices.extend(batch_indices)
                chunks.append(emb)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        if not chunks:
            return [], np.zeros((0, 0), dtype="float32")
        return good_path_indices, np.concatenate(chunks, axis=0)

    prefetch_depth = max(int(prefetch_batches), 1)
    load_queue: Queue[tuple[list[int], list[np.ndarray]] | None] = Queue(maxsize=prefetch_depth)
    tensor_queue: Queue[tuple[list[int], dict[str, torch.Tensor] | None] | None] = Queue(maxsize=1)
    worker_errors: list[BaseException] = []

    def prefetch_worker() -> None:
        executor = ThreadPoolExecutor(max_workers=workers) if workers > 0 else None
        try:
            for start in batch_starts:
                batch_paths = paths[start : start + batch_size_int]
                load_queue.put(_load_image_batch(batch_paths, start, image_size, executor))
        except BaseException as exc:
            worker_errors.append(exc)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
            load_queue.put(None)

    def processor_worker() -> None:
        try:
            while True:
                item = load_queue.get()
                if item is None:
                    tensor_queue.put(None)
                    break
                batch_indices, images = item
                if not images:
                    tensor_queue.put((batch_indices, None))
                    continue
                if fast_preprocess:
                    inputs = _fast_preprocess_image_arrays(images, processor)
                else:
                    inputs = processor(images=images, return_tensors="pt", do_resize=False, do_center_crop=False)
                tensor_queue.put((batch_indices, inputs))
        except BaseException as exc:
            worker_errors.append(exc)
            tensor_queue.put(None)

    loader = threading.Thread(target=prefetch_worker, daemon=True)
    prep = threading.Thread(target=processor_worker, daemon=True)
    loader.start()
    prep.start()

    good_path_indices: list[int] = []
    chunks: list[np.ndarray] = []
    for _ in tqdm(batch_starts, desc="visual path embeddings"):
        idle_checks = 0
        while True:
            if worker_errors:
                raise RuntimeError("Visual prefetch worker failed") from worker_errors[0]
            try:
                item = tensor_queue.get(timeout=30)
                break
            except Empty:
                idle_checks += 1
                if worker_errors:
                    raise RuntimeError("Visual prefetch worker failed") from worker_errors[0]
                if not loader.is_alive() and not prep.is_alive():
                    raise RuntimeError("Visual prefetch workers stopped without producing a batch")
                if idle_checks >= 20:
                    raise TimeoutError("No visual batch produced for 10 minutes; restart with fewer image workers")
        if item is None:
            break
        batch_indices, inputs = item
        if inputs is None:
            continue
        emb = _forward_images_adaptive(
            images=[],
            processor=processor,
            model=model,
            device=device,
            fp16=fp16,
            preprocessed=inputs,
        )
        good_path_indices.extend(batch_indices)
        chunks.append(emb)
    loader.join(timeout=1.0)
    prep.join(timeout=1.0)

    if not chunks:
        return [], np.zeros((0, 0), dtype="float32")
    return good_path_indices, np.concatenate(chunks, axis=0)


def _encode_with_transformers(
    items: pd.DataFrame,
    model_name: str,
    batch_size: int,
    device_name: str,
    cache_dir: Path | None = None,
    fp16: bool = False,
    image_workers: int = 0,
    cache_save_dtype: str = "float16",
    image_size: int = 224,
    prefetch_batches: int = 2,
    fast_preprocess: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    unique_paths, path_to_items = _paths_from_items(items)
    if not unique_paths:
        return np.zeros((len(items), 0), dtype="float32"), np.zeros(len(items), dtype="int32")

    signatures = []
    for path in unique_paths:
        try:
            signatures.append(_path_signature(path))
        except OSError:
            continue
    current = pd.DataFrame(signatures)
    if current.empty:
        return np.zeros((len(items), 0), dtype="float32"), np.zeros(len(items), dtype="int32")

    cache_index, cache_values = _load_path_cache(cache_dir, model_name=model_name)
    cached_lookup = {
        (str(row.path), int(row.size), int(row.mtime_ns)): int(row.row)
        for row in cache_index.itertuples(index=False)
    }
    missing_mask = [
        (str(row.path), int(row.size), int(row.mtime_ns)) not in cached_lookup
        for row in current.itertuples(index=False)
    ]
    missing = current.loc[missing_mask].reset_index(drop=True)
    newly_encoded_paths = 0
    print(
        "visual cache "
        f"unique_paths={len(unique_paths)} cached={len(unique_paths) - len(missing)} "
        f"missing={len(missing)} cache_dir={cache_dir or ''}"
    )
    if not missing.empty:
        missing_paths = [Path(path) for path in missing["path"].tolist()]
        processor = None
        model = None
        device = _model_device(device_name)
        try:
            from transformers import AutoImageProcessor, AutoModel

            if device.type == "cuda":
                torch.backends.cudnn.benchmark = True
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
            model.eval()
            save_every = max(int(batch_size) * 128, 65536)
            for chunk_start in range(0, len(missing_paths), save_every):
                chunk_paths = missing_paths[chunk_start : chunk_start + save_every]
                chunk_frame = missing.iloc[chunk_start : chunk_start + len(chunk_paths)].reset_index(drop=True)
                good_rel_indices, new_values = _encode_paths_with_transformers(
                    chunk_paths,
                    model_name=model_name,
                    batch_size=batch_size,
                    device_name=device_name,
                    fp16=fp16,
                    image_workers=image_workers,
                    image_size=image_size,
                    prefetch_batches=prefetch_batches,
                    fast_preprocess=fast_preprocess,
                    processor=processor,
                    model=model,
                    device=device,
                )
                if not good_rel_indices:
                    continue
                good_missing = chunk_frame.iloc[good_rel_indices].copy().reset_index(drop=True)
                if cache_values.size and int(cache_values.shape[1]) != int(new_values.shape[1]):
                    raise ValueError(
                        f"Visual cache embedding dim mismatch: {cache_values.shape[1]} != {new_values.shape[1]}"
                    )
                start_row = int(cache_values.shape[0]) if cache_values.size else 0
                good_missing["row"] = np.arange(start_row, start_row + len(good_missing), dtype=np.int64)
                cache_index = pd.concat([cache_index, good_missing[["path", "size", "mtime_ns", "row"]]], ignore_index=True)
                cache_values = new_values if not cache_values.size else np.concatenate([cache_values, new_values], axis=0)
                newly_encoded_paths += len(good_missing)
                if cache_dir is not None:
                    _save_path_cache(cache_dir, cache_index, cache_values, model_name, cache_save_dtype)
        finally:
            del processor, model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    path_row_lookup = {
        str(row.path): int(row.row)
        for row in cache_index.itertuples(index=False)
        if Path(str(row.path)).exists()
    }
    raw_dim = int(cache_values.shape[1]) if cache_values.ndim == 2 and cache_values.size else 0
    if raw_dim == 0:
        return np.zeros((len(items), 0), dtype="float32"), np.zeros(len(items), dtype="int32")

    sums = np.zeros((len(items), raw_dim), dtype="float32")
    counts = np.zeros(len(items), dtype="int32")
    for path, item_indices in path_to_items.items():
        row_idx = path_row_lookup.get(path)
        if row_idx is None:
            continue
        vec = cache_values[row_idx].astype("float32", copy=False)
        for item_idx in item_indices:
            sums[item_idx] += vec
            counts[item_idx] += 1

    print(
        "visual cache "
        f"unique_paths={len(unique_paths)} cached={len(unique_paths) - len(missing)} "
        f"newly_encoded={newly_encoded_paths} cache_dir={cache_dir or ''}"
    )
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
    if modality != "image":
        raise ValueError("Only image visual embeddings are enabled in the current mainline")
    processed_dir = Path(args.processed_dir)
    image_root = Path(args.image_root) if args.image_root else None
    qilin_dir = Path(args.qilin_dir) if args.qilin_dir else None

    items = load_visual_index(
        processed_dir=processed_dir,
        qilin_dir=qilin_dir,
        image_root=image_root,
        max_images_per_item=int(args.max_images_per_item),
        max_items=int(args.max_items),
        fallback_raw_id_paths=bool(getattr(args, "fallback_raw_id_paths", False) or not getattr(args, "qilin_dir", None)),
        image_part_min=getattr(args, "image_part_min", None),
        image_part_max=getattr(args, "image_part_max", None),
    )

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
            cache_dir=Path(args.cache_dir) if getattr(args, "cache_dir", None) else None,
            fp16=bool(getattr(args, "fp16", False)),
            image_workers=int(getattr(args, "image_workers", 0)),
            cache_save_dtype=str(getattr(args, "cache_save_dtype", "float16")),
            image_size=int(getattr(args, "image_size", 224)),
            prefetch_batches=int(getattr(args, "prefetch_batches", 2)),
            fast_preprocess=bool(getattr(args, "fast_preprocess", True)),
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
    found_part_counts = _part_counts_from_items(items, encoded_only=False)
    encoded_part_counts = _part_counts_from_items(items, encoded_only=True)
    found_part_min, found_part_max = _part_range(found_part_counts)
    encoded_part_min, encoded_part_max = _part_range(encoded_part_counts)

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
        "image_part_min": getattr(args, "image_part_min", None),
        "image_part_max": getattr(args, "image_part_max", None),
        "found_part_min": found_part_min,
        "found_part_max": found_part_max,
        "encoded_part_min": encoded_part_min,
        "encoded_part_max": encoded_part_max,
        "found_part_counts": found_part_counts,
        "encoded_part_counts": encoded_part_counts,
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


def apply_visual_sidecar_metadata(processed_dir: Path, summaries: list[dict[str, object]]) -> None:
    meta_path = processed_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {processed_dir}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    sidecars = metadata.get("feature_sidecars", {})
    if not isinstance(sidecars, dict):
        sidecars = {}
    for summary in summaries:
        modality = str(summary["modality"])
        dim = int(summary["embedding_dim"])
        if modality == "image":
            metadata["image_emb_dim"] = dim
            sidecars["image_emb"] = {
                "source": "image",
                "id_col": "item_id",
                "values": "image_embeddings.npy",
                "ids": "image_embedding_item_ids.npy",
                "dim": dim,
            }
    metadata["feature_sidecars"] = sidecars
    metadata.pop("visual_sidecar_source", None)
    metadata["visual_embedding_source"] = "build_visual_embeddings"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def visual_sidecar_ready(processed_dir: Path, modality: str) -> bool:
    processed_dir = Path(processed_dir)
    values = processed_dir / f"{modality}_embeddings.npy"
    ids = processed_dir / f"{modality}_embedding_item_ids.npy"
    return values.exists() and ids.exists()


def build_visual_embeddings_for_processed(
    processed_dir: Path,
    *,
    image_root: Path | None = None,
    qilin_dir: Path | None = None,
    modalities: tuple[str, ...] = ("image",),
    model_name: str = "openai/clip-vit-base-patch32",
    output_dim: int = 128,
    compression: str = "auto",
    batch_size: int = 32,
    device: str = "auto",
    cache_dir: Path | None = None,
    fp16: bool = False,
    image_workers: int = 0,
    cache_save_dtype: str = "float16",
    image_size: int = 224,
    prefetch_batches: int = 2,
    fast_preprocess: bool = True,
    max_images_per_item: int = 4,
    max_items: int = 0,
    image_part_min: int | None = None,
    image_part_max: int | None = None,
    save_dtype: str = "float32",
    seed: int = 2026,
    encoder: str = "transformers",
    mock_encoder: bool = False,
    mock_dim: int = 512,
    handcrafted_dim: int = 64,
    fallback_raw_id_paths: bool = False,
    skip_existing: bool = True,
    update_metadata: bool = True,
) -> list[dict[str, object]]:
    processed_dir = Path(processed_dir)
    unsupported = set(modalities) - {"image"}
    if unsupported:
        raise ValueError(f"Only image visual embeddings are enabled in the current mainline, got: {sorted(unsupported)}")
    summaries: list[dict[str, object]] = []
    for modality in modalities:
        if skip_existing and visual_sidecar_ready(processed_dir, modality):
            summary_path = processed_dir / f"{modality}_embedding_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                part_range_requested = image_part_min is not None or image_part_max is not None
                part_range_matches = (
                    not part_range_requested
                    or (
                        summary.get("image_part_min") == image_part_min
                        and summary.get("image_part_max") == image_part_max
                    )
                )
                if part_range_matches:
                    summaries.append(summary)
                    print(f"skip existing {modality} embeddings in {processed_dir}")
                    continue
                print(f"rebuild {modality} embeddings because requested image part range changed")
        args = argparse.Namespace(
            processed_dir=str(processed_dir),
            image_root=str(image_root) if image_root else None,
            qilin_dir=str(qilin_dir) if qilin_dir else None,
            model_name=model_name,
            output_dim=output_dim,
            compression=compression,
            batch_size=batch_size,
            device=device,
            cache_dir=str(cache_dir) if cache_dir else None,
            fp16=fp16,
            image_workers=image_workers,
            cache_save_dtype=cache_save_dtype,
            image_size=image_size,
            prefetch_batches=prefetch_batches,
            fast_preprocess=fast_preprocess,
            max_images_per_item=max_images_per_item,
            max_items=max_items,
            image_part_min=image_part_min,
            image_part_max=image_part_max,
            save_dtype=save_dtype,
            seed=seed,
            encoder=encoder,
            mock_encoder=mock_encoder,
            mock_dim=mock_dim,
            handcrafted_dim=handcrafted_dim,
            fallback_raw_id_paths=fallback_raw_id_paths,
        )
        summaries.append(build_embeddings_for_modality(args, modality))
    if summaries:
        (processed_dir / "visual_embedding_summary.json").write_text(
            json.dumps({"modalities": summaries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if update_metadata:
            apply_visual_sidecar_metadata(processed_dir, summaries)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build parameter-friendly image thumbnail embeddings for processed Qilin items."
    )
    parser.add_argument("--modality", choices=["image"], default="image")
    parser.add_argument("--image-root", default=None, help="Directory containing Qilin images or image covers.")
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default=None, help="Persistent raw visual path embedding cache directory.")
    parser.add_argument(
        "--cache-save-dtype",
        choices=["float32", "float16"],
        default="float16",
        help="Storage dtype for cached raw SigLIP/CLIP path embeddings.",
    )
    parser.add_argument("--fp16", action="store_true", help="Use CUDA autocast float16 for transformer image encoding.")
    parser.add_argument("--image-workers", type=int, default=8, help="Parallel image loading workers for transformer encoding.")
    parser.add_argument("--prefetch-batches", type=int, default=3, help="Number of image batches to prefetch while GPU encodes.")
    parser.add_argument("--fast-preprocess", action="store_true", default=True, help="Use vectorized preprocessing for already resized RGB arrays.")
    parser.add_argument("--no-fast-preprocess", action="store_false", dest="fast_preprocess")
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Pre-resize images before processor; 224 matches SigLIP/CLIP ViT-B defaults. Use 0 to disable.",
    )
    parser.add_argument(
        "--fallback-raw-id-paths",
        action="store_true",
        help="Also probe image_root/{raw_item_id}.jpg style paths. Disabled by default for Qilin part_x images.",
    )
    parser.add_argument("--max-images-per-item", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=0, help="Optional debug limit.")
    parser.add_argument("--image-part-min", type=int, default=None, help="Only use image paths from part_N >= this value.")
    parser.add_argument("--image-part-max", type=int, default=None, help="Only use image paths from part_N <= this value.")
    parser.add_argument("--save-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mock-encoder", action="store_true", help="Use deterministic hash vectors for tests/smoke runs.")
    parser.add_argument("--mock-dim", type=int, default=512)
    parser.add_argument("--handcrafted-dim", type=int, default=64)
    args = parser.parse_args()

    build_visual_embeddings_for_processed(
        Path(args.processed_dir),
        image_root=Path(args.image_root) if args.image_root else None,
        qilin_dir=Path(args.qilin_dir) if args.qilin_dir else None,
        modalities=("image",),
        model_name=str(args.model_name),
        output_dim=int(args.output_dim),
        compression=str(args.compression),
        batch_size=int(args.batch_size),
        device=str(args.device),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        fp16=bool(args.fp16),
        image_workers=int(args.image_workers),
        cache_save_dtype=str(args.cache_save_dtype),
        image_size=int(args.image_size),
        prefetch_batches=int(args.prefetch_batches),
        fast_preprocess=bool(args.fast_preprocess),
        max_images_per_item=int(args.max_images_per_item),
        max_items=int(args.max_items),
        image_part_min=args.image_part_min,
        image_part_max=args.image_part_max,
        save_dtype=str(args.save_dtype),
        seed=int(args.seed),
        encoder=str(args.encoder),
        mock_encoder=bool(args.mock_encoder),
        mock_dim=int(args.mock_dim),
        handcrafted_dim=int(args.handcrafted_dim),
        fallback_raw_id_paths=bool(args.fallback_raw_id_paths or not args.qilin_dir),
        skip_existing=False,
        update_metadata=True,
    )


if __name__ == "__main__":
    main()
