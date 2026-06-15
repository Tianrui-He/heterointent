from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from heterointent.data.io import read_table
from heterointent.data.schema import (
    FEATURE_PREFIXES,
    INTENT_COLUMNS,
    TASKS,
    history_columns,
    history_taxonomy_columns,
    history_type_columns,
    prefixed_columns,
)


class RankingDataset(Dataset):
    def __init__(self, table_path: str | Path, metadata: dict[str, Any]):
        self.table_path = Path(table_path)
        self.metadata = metadata
        self.max_history = int(metadata.get("max_history", 20))

        df = read_table(self.table_path).reset_index(drop=True)
        self.df = self._ensure_columns(df)
        self.feature_cols = {
            name: [f"{prefix}{i}" for i in range(int(self.metadata.get(f"{name}_dim", 0)))]
            for name, prefix in FEATURE_PREFIXES.items()
        }
        self.tensors = self._build_tensors(self.df)
        self.df = None

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        defaults = {
            "request_id": 0,
            "session_id": 0,
            "user_id": 0,
            "item_id": 0,
            "item_type": 0,
            "taxonomy_id": 0,
            "timestamp": 0,
            "position": 0,
            "next_item_type": 0,
            "click": 0,
            "collect": 0,
            "share": 0,
            **{col: 0 for col in INTENT_COLUMNS},
        }
        for col, value in defaults.items():
            if col not in df.columns:
                df[col] = value
        for col in [*history_columns(self.max_history), *history_type_columns(self.max_history), *history_taxonomy_columns(self.max_history)]:
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
        values = df[columns].to_numpy(dtype=np.float32, copy=True)
        return torch.from_numpy(values)

    def _build_tensors(self, df: pd.DataFrame) -> dict[str, torch.Tensor]:
        tensors = {
            "request_id": self._long_tensor(df, "request_id"),
            "session_id": self._long_tensor(df, "session_id"),
            "user_id": self._long_tensor(df, "user_id"),
            "item_id": self._long_tensor(df, "item_id"),
            "item_type": self._long_tensor(df, "item_type"),
            "taxonomy_id": self._long_tensor(df, "taxonomy_id"),
            "position": self._long_tensor(df, "position"),
            "history_items": torch.from_numpy(df[history_columns(self.max_history)].to_numpy(dtype=np.int64, copy=True)),
            "history_item_types": torch.from_numpy(df[history_type_columns(self.max_history)].to_numpy(dtype=np.int64, copy=True)),
            "history_taxonomy_ids": torch.from_numpy(df[history_taxonomy_columns(self.max_history)].to_numpy(dtype=np.int64, copy=True)),
            "labels": torch.from_numpy(df[list(TASKS)].to_numpy(dtype=np.float32, copy=True)),
            "next_item_type": self._long_tensor(df, "next_item_type"),
            "target_item_type": self._long_tensor(df, "target_item_type"),
            "target_taxonomy_id": self._long_tensor(df, "target_taxonomy_id"),
            "hist_dominant_item_type": self._long_tensor(df, "hist_dominant_item_type"),
            "hist_dominant_taxonomy_id": self._long_tensor(df, "hist_dominant_taxonomy_id"),
            "is_type_shift": self._long_tensor(df, "is_type_shift"),
            "is_taxonomy_shift": self._long_tensor(df, "is_taxonomy_shift"),
            "has_intent_target": self._long_tensor(df, "has_intent_target"),
        }
        for name, cols in self.feature_cols.items():
            tensors[f"{name}_feat"] = self._float_tensor(df, cols)
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


def build_dataloader(
    table_path: str | Path,
    metadata: dict[str, Any],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    fast_loader: bool = False,
):
    dataset = RankingDataset(table_path, metadata)
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
