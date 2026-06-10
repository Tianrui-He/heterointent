from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from heterointent.data.io import read_table


def build_cooccurrence_edges(
    samples: pd.DataFrame,
    min_count: int = 2,
    max_neighbors: int = 20,
) -> dict[int, list[tuple[int, float]]]:
    """Build a denoised item graph from request/session co-occurrence."""

    counter: Counter[tuple[int, int]] = Counter()
    group_key = "session_id" if "session_id" in samples.columns else "request_id"
    for _, group in samples.groupby(group_key, sort=False):
        items = group["item_id"].dropna().astype(int).unique().tolist()
        if len(items) > 80:
            items = items[:80]
        for i, src in enumerate(items):
            for dst in items[i + 1 :]:
                if src == dst:
                    continue
                a, b = sorted((src, dst))
                counter[(a, b)] += 1

    adjacency: dict[int, list[tuple[int, float]]] = {}
    for (a, b), count in counter.items():
        if count < min_count:
            continue
        adjacency.setdefault(a, []).append((b, float(count)))
        adjacency.setdefault(b, []).append((a, float(count)))

    for node, neighbors in adjacency.items():
        total = sum(w for _, w in neighbors)
        normalized = [(dst, w / max(total, 1e-12)) for dst, w in sorted(neighbors, key=lambda x: -x[1])[:max_neighbors]]
        adjacency[node] = normalized
    return adjacency


def smooth_item_features(
    samples_path: str | Path,
    output_path: str | Path,
    num_items: int,
    embed_dim: int,
    feature_prefix: str = "text_feat_",
    layers: int = 2,
    seed: int = 2026,
) -> np.ndarray:
    """Approximate FREEDOM/LightGCN-style frozen item graph embeddings."""

    rng = np.random.default_rng(seed)
    df = read_table(samples_path)
    feature_cols = sorted([c for c in df.columns if c.startswith(feature_prefix)])
    if feature_cols:
        item_feat = df.groupby("item_id")[feature_cols].mean()
        base = rng.normal(scale=0.01, size=(num_items, embed_dim)).astype("float32")
        values = item_feat.to_numpy(dtype="float32")
        if values.shape[1] >= embed_dim:
            projected = values[:, :embed_dim]
        else:
            projected = np.pad(values, ((0, 0), (0, embed_dim - values.shape[1])))
        base[item_feat.index.to_numpy(dtype=int)] = projected.astype("float32")
    else:
        base = rng.normal(scale=0.01, size=(num_items, embed_dim)).astype("float32")

    adjacency = build_cooccurrence_edges(df)
    emb = base.copy()
    for _ in range(layers):
        new_emb = emb.copy()
        for node, neighbors in adjacency.items():
            if node >= num_items:
                continue
            agg = np.zeros(embed_dim, dtype="float32")
            for dst, weight in neighbors:
                if dst < num_items:
                    agg += weight * emb[dst]
            new_emb[node] = 0.5 * emb[node] + 0.5 * agg
        emb = new_emb
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, emb.astype("float32"))
    return emb
