from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

from audit_reliability import candidate_stats
from create_strict_graph_processed import create_strict_graph_processed


def test_candidate_stats_counts_oracle_and_duplicates() -> None:
    df = pd.DataFrame(
        [
            {"request_id": 1, "item_id": 1, "position": 1, "click": 1, "collect": 0, "share": 0},
            {"request_id": 1, "item_id": 1, "position": 2, "click": 0, "collect": 1, "share": 0},
            {"request_id": 2, "item_id": 2, "position": 1, "click": 0, "collect": 0, "share": 1},
            {"request_id": 2, "item_id": 3, "position": 2, "click": 0, "collect": 0, "share": 0},
        ]
    )

    stats = candidate_stats(df)

    assert stats["rows"] == 4
    assert stats["requests"] == 2
    assert stats["duplicate_request_item_rows"] == 1
    assert stats["candidate_count"]["mean"] == 1.5
    assert stats["click"]["positive_request_rate"] == 0.5
    assert stats["collect"]["positive_request_rate"] == 0.5
    assert stats["share"]["positive_request_rate"] == 0.5
    assert abs(stats["oracle_weighted_hit@20"] - 0.5) < 1e-6


def test_create_strict_graph_processed_zeroes_train_oov_items() -> None:
    root = Path("outputs/test_audit_tools/strict_graph")
    source = root / "source"
    target = root / "target"
    if root.exists():
        shutil.rmtree(root)
    source.mkdir(parents=True)
    (source / "metadata.json").write_text(json.dumps({"num_items": 4}), encoding="utf-8")
    pd.DataFrame({"item_id": [1, 2]}).to_parquet(source / "train.parquet", index=False)
    pd.DataFrame({"x": [1]}).to_parquet(source / "dummy.parquet", index=False)
    graph = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype="float32")
    np.save(source / "graph_embedding.npy", graph)

    summary = create_strict_graph_processed(source, target)
    strict = np.load(target / "graph_embedding.npy")

    assert summary["train_seen_items"] == 2
    assert summary["zeroed_oov_items"] == 2
    assert np.allclose(strict[0], 0.0)
    assert np.allclose(strict[1], graph[1])
    assert np.allclose(strict[2], graph[2])
    assert np.allclose(strict[3], 0.0)
