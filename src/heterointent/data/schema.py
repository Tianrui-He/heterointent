from __future__ import annotations

TASKS = ("click", "collect", "share")

BASE_COLUMNS = [
    "request_id",
    "session_id",
    "user_id",
    "item_id",
    "item_type",
    "taxonomy_id",
    "timestamp",
    "position",
    "click",
    "collect",
    "share",
    "next_item_type",
]

FEATURE_PREFIXES = {
    "text": "text_feat_",
    "image": "image_feat_",
    "video": "video_feat_",
    "dense": "dense_feat_",
}


def prefixed_columns(columns: list[str], prefix: str) -> list[str]:
    return sorted([c for c in columns if c.startswith(prefix)])


def history_columns(max_history: int) -> list[str]:
    return [f"hist_item_{i}" for i in range(max_history)]
