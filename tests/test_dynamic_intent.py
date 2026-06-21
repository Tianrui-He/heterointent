from __future__ import annotations

import numpy as np
import pandas as pd

from heterointent.data.dynamic_intent import annotate_dynamic_intents, build_item_intent_lookups
from heterointent.evaluation.metrics import compute_ranking_metrics


def test_dynamic_intent_annotation_marks_shift() -> None:
    item_features = pd.DataFrame(
        [
            {"item_id": 1, "item_type": 1, "taxonomy_id": 10},
            {"item_id": 2, "item_type": 1, "taxonomy_id": 10},
            {"item_id": 3, "item_type": 2, "taxonomy_id": 20},
        ]
    )
    type_lookup, taxonomy_lookup = build_item_intent_lookups(item_features)
    df = pd.DataFrame(
        [
            {
                "request_id": 1,
                "item_id": 3,
                "item_type": 2,
                "taxonomy_id": 20,
                "position": 1,
                "click": 1,
                "collect": 0,
                "share": 0,
                "hist_item_0": 1,
                "hist_item_1": 2,
            },
            {
                "request_id": 1,
                "item_id": 2,
                "item_type": 1,
                "taxonomy_id": 10,
                "position": 2,
                "click": 0,
                "collect": 0,
                "share": 0,
                "hist_item_0": 1,
                "hist_item_1": 2,
            },
        ]
    )
    out = annotate_dynamic_intents(df, type_lookup, taxonomy_lookup, max_history=2)
    assert out["target_item_type"].nunique() == 1
    assert int(out["target_item_type"].iloc[0]) == 2
    assert int(out["hist_dominant_item_type"].iloc[0]) == 1
    assert int(out["is_type_shift"].iloc[0]) == 1
    assert int(out["has_intent_target"].iloc[0]) == 1


def test_dynamic_intent_metrics_exclude_no_target_requests() -> None:
    df = pd.DataFrame(
        [
            {
                "request_id": 1,
                "item_id": 1,
                "score": 0.9,
                "click": 1,
                "collect": 0,
                "share": 0,
                "p_click": 0.9,
                "p_collect": 0.1,
                "p_share": 0.1,
                "has_intent_target": 1,
                "is_type_shift": 1,
                "is_taxonomy_shift": 1,
                "intent_type_hit@1": 1.0,
                "intent_type_hit@2": 1.0,
                "intent_taxonomy_hit@1": 0.0,
                "intent_taxonomy_hit@5": 1.0,
                "intent_taxonomy_mrr": 0.5,
                "attention_type_target_mass": 0.7,
                "attention_taxonomy_target_mass": 0.2,
            },
            {
                "request_id": 2,
                "item_id": 2,
                "score": 0.8,
                "click": 0,
                "collect": 0,
                "share": 0,
                "p_click": 0.1,
                "p_collect": 0.1,
                "p_share": 0.1,
                "has_intent_target": 0,
                "is_type_shift": 0,
                "is_taxonomy_shift": 0,
                "intent_type_hit@1": 0.0,
                "intent_type_hit@2": 0.0,
                "intent_taxonomy_hit@1": 0.0,
                "intent_taxonomy_hit@5": 0.0,
                "intent_taxonomy_mrr": 0.0,
                "attention_type_target_mass": 0.0,
                "attention_taxonomy_target_mass": 0.0,
            },
        ]
    )
    metrics = compute_ranking_metrics(df, topk=1, include_diagnostics=True)
    assert metrics["intent_target_requests"] == 1.0
    assert metrics["intent_type_acc@1"] == 1.0
    assert metrics["intent_taxonomy_acc@5"] == 1.0
    assert metrics["shift_taxonomy_hit@5"] == 1.0
    assert np.isclose(metrics["attention_target_mass"], 0.45)
