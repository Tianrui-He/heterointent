from __future__ import annotations

TASKS = ("click", "collect", "share")
AUX_TASKS = ("like", "comment")
AUX_REGRESSION_TARGETS = ("page_time_log",)

INTENT_COLUMNS = [
    "target_item_type",
    "target_taxonomy_id",
    "hist_dominant_item_type",
    "hist_dominant_taxonomy_id",
    "is_type_shift",
    "is_taxonomy_shift",
    "has_intent_target",
]

BASE_COLUMNS = [
    "request_id",
    "session_id",
    "raw_user_id",
    "raw_item_id",
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
    "timestamp",
    "position",
    "click",
    "collect",
    "share",
    *AUX_TASKS,
    *AUX_REGRESSION_TARGETS,
    "next_item_type",
    *INTENT_COLUMNS,
]

FEATURE_PREFIXES = {
    "text": "text_feat_",
    "text_title": "text_title_feat_",
    "text_content": "text_content_feat_",
    "text_stat": "text_stat_feat_",
    "query": "query_feat_",
    "query_cross": "query_cross_feat_",
    "image": "image_feat_",
    "video": "video_feat_",
    "dense": "dense_feat_",
    "image_meta": "image_meta_feat_",
    "video_meta": "video_meta_feat_",
    "image_emb": "image_emb_feat_",
    "video_emb": "video_emb_feat_",
    "item_dense": "item_dense_feat_",
    "user_dense": "user_dense_feat_",
    "ratio": "ratio_feat_",
    "cross": "cross_feat_",
    "history_text": "history_text_feat_",
    "history_text_last": "history_text_last_feat_",
    "history_ratio": "history_ratio_feat_",
}

# cold_stage_id: 0=pad, 1=extreme_cold, 2=cold, 3=warm, 4=hot
NUM_COLD_STAGES = 5


def prefixed_columns(columns: list[str], prefix: str) -> list[str]:
    return sorted([c for c in columns if c.startswith(prefix)])


def history_columns(max_history: int) -> list[str]:
    return [f"hist_item_{i}" for i in range(max_history)]


def history_type_columns(max_history: int) -> list[str]:
    return [f"hist_item_type_{i}" for i in range(max_history)]


def history_taxonomy_columns(max_history: int) -> list[str]:
    return [f"hist_taxonomy_id_{i}" for i in range(max_history)]


def history_taxonomy1_columns(max_history: int) -> list[str]:
    return [f"hist_taxonomy1_id_{i}" for i in range(max_history)]


def history_taxonomy2_columns(max_history: int) -> list[str]:
    return [f"hist_taxonomy2_id_{i}" for i in range(max_history)]
