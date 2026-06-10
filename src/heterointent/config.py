from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dictionary with attribute access for experiment configs."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def copy(self) -> "Config":
        return Config(deepcopy(dict(self)))


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config(data)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_plain_dict(config), f, allow_unicode=True, sort_keys=False)


def to_plain_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_plain_dict(v) for v in obj)
    return obj
