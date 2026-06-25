from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from heterointent.data.io import read_table
from heterointent.data.schema import (
    AUX_REGRESSION_TARGETS,
    AUX_TASKS,
    FEATURE_PREFIXES,
    INTENT_COLUMNS,
    TASKS,
    history_columns,
    history_taxonomy1_columns,
    history_taxonomy2_columns,
    history_taxonomy_columns,
    history_type_columns,
    prefixed_columns,
)

SCALAR_FEATURE_COLUMNS = (
    "cold_stage_id",
    "item_imp_bucket",
    "item_click_bucket",
    "has_query",
    "has_image_emb",
    "has_video_emb",
)

SIDECAR_FEATURE_SPECS = {
    "text": ("text", "item_id"),
    "text_title": ("text_title", "item_id"),
    "text_content": ("text_content", "item_id"),
    "query": ("query", "request_id"),
    "image": ("image", "item_id"),
    "video": ("video", "item_id"),
    "image_emb": ("image", "item_id"),
    "video_emb": ("video", "item_id"),
}


class RankingDataset(Dataset):
    def __init__(self, table_path: str | Path, metadata: dict[str, Any], tensor_device: str | None = None):
        self.table_path = Path(table_path)
        self.metadata = metadata
        self.max_history = int(metadata.get("max_history", 20))
        self.feature_cols = {
            name: [f"{prefix}{i}" for i in range(int(self.metadata.get(f"{name}_dim", 0)))]
            for name, prefix in FEATURE_PREFIXES.items()
        }
        if self.table_path.suffix == ".parquet":
            self.tensors = self._build_tensors_streaming()
        else:
            df = read_table(self.table_path).reset_index(drop=True)
            df = self._ensure_columns(df)
            self.tensors = self._build_tensors(df)
        if tensor_device:
            device = torch.device(tensor_device)
            self.tensors = {key: value.to(device, non_blocking=device.type == "cuda") for key, value in self.tensors.items()}

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        defaults = {
            "request_id": 0,
            "session_id": 0,
            "user_id": 0,
            "item_id": 0,
            "item_type": 0,
            "taxonomy1_id": 0,
            "taxonomy2_id": 0,
            "taxonomy_id": 0,
            "gender_id": 0,
            "platform_id": 0,
            "age_id": 0,
            "location_id": 0,
            "timestamp": 0,
            "position": 0,
            "next_item_type": 0,
            "click": 0,
            "collect": 0,
            "share": 0,
            "like": 0,
            "comment": 0,
            "page_time_log": 0.0,
            **{col: 0 for col in INTENT_COLUMNS},
        }
        for col, value in defaults.items():
            if col not in df.columns:
                df[col] = value
        history_defaults = [
            *history_columns(self.max_history),
            *history_type_columns(self.max_history),
            *history_taxonomy_columns(self.max_history),
            *history_taxonomy1_columns(self.max_history),
            *history_taxonomy2_columns(self.max_history),
        ]
        for col in history_defaults:
            if col not in df.columns:
                df[col] = 0
        for name, prefix in FEATURE_PREFIXES.items():
            dim = int(self.metadata.get(f"{name}_dim", 0))
            existing = set(prefixed_columns(list(df.columns), prefix))
            for i in range(dim):
                col = f"{prefix}{i}"
                if col not in existing:
                    df[col] = 0.0
        return df

    def _long_tensor(self, df: pd.DataFrame, column: str) -> torch.Tensor:
        return torch.from_numpy(df[column].to_numpy(dtype=np.int64, copy=True))

    def _float_tensor(self, df: pd.DataFrame, columns: list[str]) -> torch.Tensor:
        if not columns:
            return torch.zeros((len(df), 0), dtype=torch.float32)
        values = df[columns].fillna(0.0).to_numpy(dtype=np.float32, copy=True)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(values)

    def _build_tensors(self, df: pd.DataFrame) -> dict[str, torch.Tensor]:
        tensors = {
            "request_id": self._long_tensor(df, "request_id"),
            "session_id": self._long_tensor(df, "session_id"),
            "user_id": self._long_tensor(df, "user_id"),
            "item_id": self._long_tensor(df, "item_id"),
            "item_type": self._long_tensor(df, "item_type"),
            "taxonomy1_id": self._long_tensor(df, "taxonomy1_id"),
            "taxonomy2_id": self._long_tensor(df, "taxonomy2_id"),
            "taxonomy_id": self._long_tensor(df, "taxonomy_id"),
            "gender_id": self._long_tensor(df, "gender_id"),
            "platform_id": self._long_tensor(df, "platform_id"),
            "age_id": self._long_tensor(df, "age_id"),
            "location_id": self._long_tensor(df, "location_id"),
            "position": self._long_tensor(df, "position"),
            "history_items": torch.from_numpy(df[history_columns(self.max_history)].to_numpy(dtype=np.int64, copy=True)),
            "history_item_types": torch.from_numpy(df[history_type_columns(self.max_history)].to_numpy(dtype=np.int64, copy=True)),
            "history_taxonomy_ids": torch.from_numpy(df[history_taxonomy_columns(self.max_history)].to_numpy(dtype=np.int64, copy=True)),
            "history_taxonomy1_ids": torch.from_numpy(df[history_taxonomy1_columns(self.max_history)].to_numpy(dtype=np.int64, copy=True)),
            "history_taxonomy2_ids": torch.from_numpy(df[history_taxonomy2_columns(self.max_history)].to_numpy(dtype=np.int64, copy=True)),
            "labels": torch.from_numpy(df[list(TASKS)].to_numpy(dtype=np.float32, copy=True)),
            "aux_labels": torch.from_numpy(df[list(AUX_TASKS)].to_numpy(dtype=np.float32, copy=True)),
            "page_time_log": torch.from_numpy(df[list(AUX_REGRESSION_TARGETS)].to_numpy(dtype=np.float32, copy=True)).squeeze(-1),
            "next_item_type": self._long_tensor(df, "next_item_type"),
            "target_item_type": self._long_tensor(df, "target_item_type"),
            "target_taxonomy_id": self._long_tensor(df, "target_taxonomy_id"),
            "hist_dominant_item_type": self._long_tensor(df, "hist_dominant_item_type"),
            "hist_dominant_taxonomy_id": self._long_tensor(df, "hist_dominant_taxonomy_id"),
            "is_type_shift": self._long_tensor(df, "is_type_shift"),
            "is_taxonomy_shift": self._long_tensor(df, "is_taxonomy_shift"),
            "has_intent_target": self._long_tensor(df, "has_intent_target"),
        }
        for col in SCALAR_FEATURE_COLUMNS:
            if col in df.columns:
                tensors[col] = self._long_tensor(df, col)
            elif col == "has_query" and "query" in df.columns:
                tensors[col] = torch.from_numpy(df["query"].fillna("").astype(str).str.len().gt(0).to_numpy(dtype=np.int64))
            else:
                tensors[col] = torch.zeros(len(df), dtype=torch.int64)
        for name, cols in self.feature_cols.items():
            tensors[f"{name}_feat"] = self._float_tensor(df, cols)
        return tensors

    def _required_parquet_columns(self) -> list[str]:
        columns = [
            "request_id",
            "session_id",
            "user_id",
            "item_id",
            "item_type",
            "taxonomy1_id",
            "taxonomy2_id",
            "taxonomy_id",
            "gender_id",
            "platform_id",
            "age_id",
            "location_id",
            "position",
            *history_columns(self.max_history),
            *history_type_columns(self.max_history),
            *history_taxonomy_columns(self.max_history),
            *history_taxonomy1_columns(self.max_history),
            *history_taxonomy2_columns(self.max_history),
            *TASKS,
            *AUX_TASKS,
            *AUX_REGRESSION_TARGETS,
            "next_item_type",
            *INTENT_COLUMNS,
            *SCALAR_FEATURE_COLUMNS,
        ]
        for cols in self.feature_cols.values():
            columns.extend(cols)
        columns.append("query")
        schema_cols = set(pq.read_schema(self.table_path).names)
        return sorted({col for col in columns if col in schema_cols})

    @staticmethod
    def _embedding_ids_name(id_col: str) -> str:
        return f"{id_col}s"

    def _metadata_sidecar_config(self, name: str) -> dict[str, Any] | None:
        configs = self.metadata.get("feature_sidecars", {})
        if not isinstance(configs, dict):
            return None
        config = configs.get(name)
        return config if isinstance(config, dict) else None

    def _resolve_sidecar_feature_specs(self, schema_cols: set[str]) -> dict[str, dict[str, Any]]:
        specs: dict[str, dict[str, Any]] = {}
        processed_dir = self.table_path.parent
        for name, cols in self.feature_cols.items():
            if not cols or all(col in schema_cols for col in cols):
                continue
            metadata_config = self._metadata_sidecar_config(name)
            if metadata_config:
                source_name = str(metadata_config.get("source", name))
                id_col = str(metadata_config.get("id_col", SIDECAR_FEATURE_SPECS.get(name, (name, "item_id"))[1]))
                values_name = str(metadata_config.get("values", f"{source_name}_embeddings.npy"))
                ids_name = str(metadata_config.get("ids", f"{source_name}_embedding_{self._embedding_ids_name(id_col)}.npy"))
            elif name in SIDECAR_FEATURE_SPECS:
                source_name, id_col = SIDECAR_FEATURE_SPECS[name]
                values_name = f"{source_name}_embeddings.npy"
                ids_name = f"{source_name}_embedding_{self._embedding_ids_name(id_col)}.npy"
            else:
                continue

            values_path = processed_dir / values_name
            ids_path = processed_dir / ids_name
            if not values_path.exists() or not ids_path.exists():
                continue
            values = np.load(values_path, mmap_mode="r")
            ids = np.load(ids_path, mmap_mode="r")
            if values.ndim != 2:
                raise ValueError(f"Expected 2D sidecar embeddings in {values_path}, got shape {values.shape}")
            if values.shape[1] != len(cols):
                raise ValueError(
                    f"Sidecar dimension mismatch for {name}: metadata expects {len(cols)}, "
                    f"but {values_path.name} has {values.shape[1]}"
                )
            ids_array = np.asarray(ids, dtype=np.int64)
            max_id = int(ids_array.max(initial=0))
            lookup = np.full(max_id + 1, -1, dtype=np.int64)
            valid_ids = ids_array >= 0
            lookup[ids_array[valid_ids]] = np.arange(len(ids_array), dtype=np.int64)[valid_ids]
            specs[name] = {
                "values": values,
                "lookup": lookup,
                "id_col": id_col,
                "dim": len(cols),
                "values_path": str(values_path),
            }
        return specs

    @staticmethod
    def _sidecar_values_for_batch(df: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
        dim = int(spec["dim"])
        out = np.zeros((len(df), dim), dtype=np.float32)
        id_col = str(spec["id_col"])
        if id_col not in df.columns or len(df) == 0:
            return out
        raw_ids = df[id_col].fillna(0).to_numpy(dtype=np.int64, copy=False)
        lookup: np.ndarray = spec["lookup"]
        in_range = (raw_ids >= 0) & (raw_ids < len(lookup))
        positions = np.full(len(raw_ids), -1, dtype=np.int64)
        positions[in_range] = lookup[raw_ids[in_range]]
        valid = positions >= 0
        if bool(valid.any()):
            out[valid] = np.asarray(spec["values"][positions[valid], :dim], dtype=np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_tensors_streaming(self, chunk_rows: int = 65536) -> dict[str, torch.Tensor]:
        pf = pq.ParquetFile(self.table_path)
        num_rows = pf.metadata.num_rows
        columns = self._required_parquet_columns()
        schema_cols = set(pq.read_schema(self.table_path).names)
        sidecar_specs = self._resolve_sidecar_feature_specs(schema_cols)
        buffers: dict[str, Any] = {
            "request_id": np.zeros(num_rows, dtype=np.int64),
            "session_id": np.zeros(num_rows, dtype=np.int64),
            "user_id": np.zeros(num_rows, dtype=np.int64),
            "item_id": np.zeros(num_rows, dtype=np.int64),
            "item_type": np.zeros(num_rows, dtype=np.int64),
            "taxonomy1_id": np.zeros(num_rows, dtype=np.int64),
            "taxonomy2_id": np.zeros(num_rows, dtype=np.int64),
            "taxonomy_id": np.zeros(num_rows, dtype=np.int64),
            "gender_id": np.zeros(num_rows, dtype=np.int64),
            "platform_id": np.zeros(num_rows, dtype=np.int64),
            "age_id": np.zeros(num_rows, dtype=np.int64),
            "location_id": np.zeros(num_rows, dtype=np.int64),
            "position": np.zeros(num_rows, dtype=np.int64),
            "history_items": np.zeros((num_rows, self.max_history), dtype=np.int64),
            "history_item_types": np.zeros((num_rows, self.max_history), dtype=np.int64),
            "history_taxonomy_ids": np.zeros((num_rows, self.max_history), dtype=np.int64),
            "history_taxonomy1_ids": np.zeros((num_rows, self.max_history), dtype=np.int64),
            "history_taxonomy2_ids": np.zeros((num_rows, self.max_history), dtype=np.int64),
            "labels": np.zeros((num_rows, len(TASKS)), dtype=np.float32),
            "aux_labels": np.zeros((num_rows, len(AUX_TASKS)), dtype=np.float32),
            "page_time_log": np.zeros(num_rows, dtype=np.float32),
            "next_item_type": np.zeros(num_rows, dtype=np.int64),
            "target_item_type": np.zeros(num_rows, dtype=np.int64),
            "target_taxonomy_id": np.zeros(num_rows, dtype=np.int64),
            "hist_dominant_item_type": np.zeros(num_rows, dtype=np.int64),
            "hist_dominant_taxonomy_id": np.zeros(num_rows, dtype=np.int64),
            "is_type_shift": np.zeros(num_rows, dtype=np.int64),
            "is_taxonomy_shift": np.zeros(num_rows, dtype=np.int64),
            "has_intent_target": np.zeros(num_rows, dtype=np.int64),
        }
        for col in SCALAR_FEATURE_COLUMNS:
            buffers[col] = np.zeros(num_rows, dtype=np.int64)
        for name, cols in self.feature_cols.items():
            buffers[f"{name}_feat"] = np.zeros((num_rows, len(cols)), dtype=np.float32)

        offset = 0
        for batch in pf.iter_batches(batch_size=chunk_rows, columns=columns):
            df = batch.to_pandas()
            end = offset + len(df)
            for key in (
                "request_id",
                "session_id",
                "user_id",
                "item_id",
                "item_type",
                "taxonomy1_id",
                "taxonomy2_id",
                "taxonomy_id",
                "gender_id",
                "platform_id",
                "age_id",
                "location_id",
                "position",
                "next_item_type",
                "target_item_type",
                "target_taxonomy_id",
                "hist_dominant_item_type",
                "hist_dominant_taxonomy_id",
                "is_type_shift",
                "is_taxonomy_shift",
                "has_intent_target",
            ):
                if key in df.columns:
                    buffers[key][offset:end] = df[key].fillna(0).to_numpy(dtype=np.int64, copy=False)
            if all(col in df.columns for col in history_columns(self.max_history)):
                buffers["history_items"][offset:end] = df[history_columns(self.max_history)].to_numpy(dtype=np.int64, copy=False)
            if all(col in df.columns for col in history_type_columns(self.max_history)):
                buffers["history_item_types"][offset:end] = df[history_type_columns(self.max_history)].to_numpy(dtype=np.int64, copy=False)
            if all(col in df.columns for col in history_taxonomy_columns(self.max_history)):
                buffers["history_taxonomy_ids"][offset:end] = df[history_taxonomy_columns(self.max_history)].to_numpy(dtype=np.int64, copy=False)
            if all(col in df.columns for col in history_taxonomy1_columns(self.max_history)):
                buffers["history_taxonomy1_ids"][offset:end] = df[history_taxonomy1_columns(self.max_history)].to_numpy(dtype=np.int64, copy=False)
            if all(col in df.columns for col in history_taxonomy2_columns(self.max_history)):
                buffers["history_taxonomy2_ids"][offset:end] = df[history_taxonomy2_columns(self.max_history)].to_numpy(dtype=np.int64, copy=False)
            if all(task in df.columns for task in TASKS):
                buffers["labels"][offset:end] = df[list(TASKS)].to_numpy(dtype=np.float32, copy=False)
            if all(task in df.columns for task in AUX_TASKS):
                buffers["aux_labels"][offset:end] = df[list(AUX_TASKS)].to_numpy(dtype=np.float32, copy=False)
            if AUX_REGRESSION_TARGETS[0] in df.columns:
                buffers["page_time_log"][offset:end] = df[AUX_REGRESSION_TARGETS[0]].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
            for col in SCALAR_FEATURE_COLUMNS:
                if col in df.columns:
                    buffers[col][offset:end] = df[col].fillna(0).to_numpy(dtype=np.int64, copy=False)
                elif col == "has_query" and "query" in df.columns:
                    buffers[col][offset:end] = df["query"].fillna("").astype(str).str.len().gt(0).to_numpy(dtype=np.int64)
            for name, cols in self.feature_cols.items():
                if cols and all(col in df.columns for col in cols):
                    feat = df[cols].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
                    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                    buffers[f"{name}_feat"][offset:end] = feat
                elif cols and name in sidecar_specs:
                    buffers[f"{name}_feat"][offset:end] = self._sidecar_values_for_batch(df, sidecar_specs[name])
            offset = end

        tensors = {
            "request_id": torch.from_numpy(buffers["request_id"]),
            "session_id": torch.from_numpy(buffers["session_id"]),
            "user_id": torch.from_numpy(buffers["user_id"]),
            "item_id": torch.from_numpy(buffers["item_id"]),
            "item_type": torch.from_numpy(buffers["item_type"]),
            "taxonomy1_id": torch.from_numpy(buffers["taxonomy1_id"]),
            "taxonomy2_id": torch.from_numpy(buffers["taxonomy2_id"]),
            "taxonomy_id": torch.from_numpy(buffers["taxonomy_id"]),
            "gender_id": torch.from_numpy(buffers["gender_id"]),
            "platform_id": torch.from_numpy(buffers["platform_id"]),
            "age_id": torch.from_numpy(buffers["age_id"]),
            "location_id": torch.from_numpy(buffers["location_id"]),
            "position": torch.from_numpy(buffers["position"]),
            "history_items": torch.from_numpy(buffers["history_items"]),
            "history_item_types": torch.from_numpy(buffers["history_item_types"]),
            "history_taxonomy_ids": torch.from_numpy(buffers["history_taxonomy_ids"]),
            "history_taxonomy1_ids": torch.from_numpy(buffers["history_taxonomy1_ids"]),
            "history_taxonomy2_ids": torch.from_numpy(buffers["history_taxonomy2_ids"]),
            "labels": torch.from_numpy(buffers["labels"]),
            "aux_labels": torch.from_numpy(buffers["aux_labels"]),
            "page_time_log": torch.from_numpy(buffers["page_time_log"]),
            "next_item_type": torch.from_numpy(buffers["next_item_type"]),
            "target_item_type": torch.from_numpy(buffers["target_item_type"]),
            "target_taxonomy_id": torch.from_numpy(buffers["target_taxonomy_id"]),
            "hist_dominant_item_type": torch.from_numpy(buffers["hist_dominant_item_type"]),
            "hist_dominant_taxonomy_id": torch.from_numpy(buffers["hist_dominant_taxonomy_id"]),
            "is_type_shift": torch.from_numpy(buffers["is_type_shift"]),
            "is_taxonomy_shift": torch.from_numpy(buffers["is_taxonomy_shift"]),
            "has_intent_target": torch.from_numpy(buffers["has_intent_target"]),
        }
        for col in SCALAR_FEATURE_COLUMNS:
            tensors[col] = torch.from_numpy(buffers[col])
        for name in self.feature_cols:
            tensors[f"{name}_feat"] = torch.from_numpy(buffers[f"{name}_feat"])
        return tensors

    def __len__(self) -> int:
        return int(self.tensors["item_id"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: value[idx] for key, value in self.tensors.items()}


class FastTensorBatchLoader:
    """Batch tensors directly, bypassing per-sample DataLoader collation overhead."""

    def __init__(self, dataset: RankingDataset, batch_size: int, shuffle: bool, pin_memory: bool = False):
        self.tensors = dataset.tensors
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.pin_memory = bool(pin_memory and torch.cuda.is_available())
        self.num_samples = len(dataset)

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def _maybe_pin(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.pin_memory:
            return batch
        return {key: value.pin_memory() for key, value in batch.items()}

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self.shuffle:
            indices = torch.randperm(self.num_samples)
            for start in range(0, self.num_samples, self.batch_size):
                idx = indices[start:start + self.batch_size]
                yield self._maybe_pin({key: value.index_select(0, idx) for key, value in self.tensors.items()})
        else:
            for start in range(0, self.num_samples, self.batch_size):
                end = min(start + self.batch_size, self.num_samples)
                yield self._maybe_pin({key: value[start:end] for key, value in self.tensors.items()})


class RequestPreservingBatchLoader:
    """Batch complete request groups so ranking losses see within-request candidates."""

    def __init__(self, dataset: RankingDataset, batch_size: int, shuffle: bool, pin_memory: bool = False):
        self.tensors = dataset.tensors
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.pin_memory = bool(pin_memory and torch.cuda.is_available())
        self.num_samples = len(dataset)
        self.groups = self._build_groups(dataset.tensors["request_id"])
        self.group_sizes = [int(group.numel()) for group in self.groups]

    @staticmethod
    def _build_groups(request_ids: torch.Tensor) -> list[torch.Tensor]:
        ids = request_ids.detach().cpu().numpy()
        if ids.size == 0:
            return []
        order = np.argsort(ids, kind="mergesort")
        sorted_ids = ids[order]
        starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
        ends = np.r_[starts[1:], len(order)]
        return [torch.from_numpy(order[start:end].astype(np.int64, copy=True)) for start, end in zip(starts, ends)]

    def __len__(self) -> int:
        if not self.group_sizes:
            return 0
        batches = 0
        current = 0
        for size in self.group_sizes:
            if current > 0 and current + size > self.batch_size:
                batches += 1
                current = 0
            current += size
        return batches + int(current > 0)

    def _maybe_pin(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.pin_memory:
            return batch
        return {key: value.pin_memory() for key, value in batch.items()}

    def _batch_from_groups(self, groups: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        idx = groups[0] if len(groups) == 1 else torch.cat(groups)
        return self._maybe_pin({key: value.index_select(0, idx) for key, value in self.tensors.items()})

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if not self.groups:
            return
        group_order = torch.randperm(len(self.groups)).tolist() if self.shuffle else list(range(len(self.groups)))
        pending: list[torch.Tensor] = []
        pending_size = 0
        for group_idx in group_order:
            group = self.groups[group_idx]
            group_size = int(group.numel())
            if pending and pending_size + group_size > self.batch_size:
                yield self._batch_from_groups(pending)
                pending = []
                pending_size = 0
            pending.append(group)
            pending_size += group_size
        if pending:
            yield self._batch_from_groups(pending)


def build_dataloader(
    table_path: str | Path,
    metadata: dict[str, Any],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    fast_loader: bool = False,
    request_preserving: bool = False,
    tensor_device: str | None = None,
):
    dataset = RankingDataset(table_path, metadata, tensor_device=tensor_device)
    on_device = tensor_device is not None and str(tensor_device).startswith("cuda")
    pin_memory = bool(pin_memory and not on_device)
    if request_preserving:
        return RequestPreservingBatchLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
    if fast_loader:
        return FastTensorBatchLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)
