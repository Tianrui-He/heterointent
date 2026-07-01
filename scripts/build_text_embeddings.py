from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


def pool_hidden_state(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        return last_hidden_state[:, 0]
    if pooling == "mean":
        return mean_pool(last_hidden_state, attention_mask)
    raise ValueError(f"Unsupported pooling: {pooling}")


def load_notes(qilin_dir: Path, item_map_path: Path) -> pd.DataFrame:
    item_map = pd.read_parquet(item_map_path)
    raw_needed = set(item_map["raw_item_id"].astype(int).tolist())
    frames = []
    for file in sorted((qilin_dir / "notes").glob("*.parquet")):
        df = pd.read_parquet(file, columns=["note_idx", "note_title", "note_content"])
        df = df[df["note_idx"].astype(int).isin(raw_needed)]
        if len(df):
            frames.append(df)
    notes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["note_idx", "note_title", "note_content"])
    notes = notes.merge(item_map, left_on="note_idx", right_on="raw_item_id", how="inner")
    notes["title"] = notes["note_title"].fillna("").astype(str)
    notes["content"] = notes["note_content"].fillna("").astype(str)
    notes["joint"] = notes["title"] + " " + notes["content"]
    return notes[["item_id", "raw_item_id", "title", "content", "joint"]].sort_values("item_id").reset_index(drop=True)


def load_queries(processed_dir: Path) -> pd.DataFrame:
    frames = []
    for split in ("train", "valid", "test"):
        path = processed_dir / f"{split}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=["request_id", "query"])
        except Exception:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["request_id", "query"])
    queries = pd.concat(frames, ignore_index=True)
    queries["query"] = queries["query"].fillna("").astype(str)
    queries = queries.drop_duplicates("request_id", keep="first")
    return queries.sort_values("request_id").reset_index(drop=True)


def output_name_for_item_text(kind: str) -> str:
    return "text" if kind == "joint" else f"text_{kind}"


def encode_text_frame(
    frame: pd.DataFrame,
    text_col: str,
    batch_size: int,
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    pooling: str,
) -> np.ndarray:
    all_embs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(frame), batch_size), desc=f"{text_col} embeddings"):
            texts = frame[text_col].iloc[start:start + batch_size].fillna("").astype(str).tolist()
            batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            out = model(**batch)
            emb = pool_hidden_state(out.last_hidden_state, batch["attention_mask"], pooling)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            all_embs.append(emb.float().cpu().numpy().astype("float32"))
    return np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, 0), dtype="float32")


def write_embedding_artifacts(
    processed: Path,
    name: str,
    values: np.ndarray,
    ids: np.ndarray,
    id_column: str,
    extra_columns: pd.DataFrame,
    config: dict,
) -> None:
    table_suffix = "items" if id_column == "item_id" else "requests"
    np.save(processed / f"{name}_embeddings.npy", values)
    np.save(processed / f"{name}_embedding_{id_column}s.npy", ids)
    extra_columns.to_parquet(processed / f"{name}_embedding_{table_suffix}.parquet", index=False)
    (processed / f"{name}_embedding_config.json").write_text(
        json.dumps(
            {
                **config,
                "num_rows": int(len(extra_columns)),
                "embedding_shape": list(values.shape),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


TEXT_DIM_KEYS = {
    "text": "text_dim",
    "text_title": "text_title_dim",
    "text_content": "text_content_dim",
    "query": "query_dim",
}


def text_sidecar_ready(processed_dir: Path, name: str) -> bool:
    processed_dir = Path(processed_dir)
    values = processed_dir / f"{name}_embeddings.npy"
    id_col = "item_id" if name != "query" else "request_id"
    ids = processed_dir / f"{name}_embedding_{id_col}s.npy"
    return values.exists() and ids.exists()


def apply_text_sidecar_metadata(processed_dir: Path, built: list[tuple[str, int]]) -> None:
    meta_path = processed_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {processed_dir}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    sidecars = metadata.get("feature_sidecars", {})
    if not isinstance(sidecars, dict):
        sidecars = {}
    for name, dim in built:
        dim_key = TEXT_DIM_KEYS[name]
        metadata[dim_key] = int(dim)
        id_col = "request_id" if name == "query" else "item_id"
        sidecars[name] = {
            "source": name,
            "id_col": id_col,
            "values": f"{name}_embeddings.npy",
            "ids": f"{name}_embedding_{id_col}s.npy",
            "dim": int(dim),
        }
    metadata["feature_sidecars"] = sidecars
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def build_text_embeddings_for_processed(
    processed_dir: Path,
    qilin_dir: Path,
    *,
    model_name: str,
    batch_size: int = 64,
    max_length: int = 256,
    pooling: str = "mean",
    item_texts: list[str] | None = None,
    encode_query: bool = True,
    device: str = "auto",
    skip_existing: bool = True,
    update_metadata: bool = True,
) -> dict[str, object]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Please install transformers first: pip install transformers") from exc

    processed_dir = Path(processed_dir)
    qilin_dir = Path(qilin_dir)
    resolved = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(resolved)
    model.eval()

    config = {"model_name": model_name, "pooling": pooling, "max_length": max_length}
    views = [name for name in (item_texts or ["joint", "title", "content"]) if name != "none"]
    built: list[tuple[str, int]] = []
    summary: dict[str, object] = {"views": {}, "query": None}

    for kind in views:
        name = output_name_for_item_text(kind)
        if skip_existing and text_sidecar_ready(processed_dir, name):
            values = np.load(processed_dir / f"{name}_embeddings.npy", mmap_mode="r")
            built.append((name, int(values.shape[1])))
            summary["views"][name] = {"shape": list(values.shape), "skipped": True}
            print(f"skip existing {name} embeddings in {processed_dir}")
            continue
        notes = load_notes(qilin_dir, processed_dir / "item_id_map.parquet")
        values = encode_text_frame(notes, kind, batch_size, tokenizer, model, resolved, max_length, pooling)
        write_embedding_artifacts(
            processed_dir,
            name,
            values,
            notes["item_id"].to_numpy(dtype="int64"),
            "item_id",
            notes[["item_id", "raw_item_id"]],
            {**config, "text_view": kind},
        )
        built.append((name, int(values.shape[1])))
        summary["views"][name] = {"shape": list(values.shape), "skipped": False}
        print(f"wrote {name} embeddings {values.shape} to {processed_dir} with pooling={pooling}")

    if encode_query:
        if skip_existing and text_sidecar_ready(processed_dir, "query"):
            values = np.load(processed_dir / "query_embeddings.npy", mmap_mode="r")
            built.append(("query", int(values.shape[1])))
            summary["query"] = {"shape": list(values.shape), "skipped": True}
            print(f"skip existing query embeddings in {processed_dir}")
        else:
            queries = load_queries(processed_dir)
            if queries.empty:
                raise SystemExit("No query column found in processed train/valid/test files. Re-run prepare_qilin first.")
            values = encode_text_frame(queries, "query", batch_size, tokenizer, model, resolved, max_length, pooling)
            write_embedding_artifacts(
                processed_dir,
                "query",
                values,
                queries["request_id"].to_numpy(dtype="int64"),
                "request_id",
                queries[["request_id"]],
                {**config, "text_view": "query"},
            )
            built.append(("query", int(values.shape[1])))
            summary["query"] = {"shape": list(values.shape), "skipped": False}
            print(f"wrote query embeddings {values.shape} to {processed_dir} with pooling={pooling}")

    if update_metadata and built:
        apply_text_sidecar_metadata(processed_dir, built)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BGE/Transformer text embeddings for processed Qilin items.")
    parser.add_argument("--qilin-dir", default=str(ROOT / "data" / "raw" / "Qilin"))
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "run_latest" / "processed" / "qilin_base"))
    parser.add_argument("--model-name", default="D:\\models\\bge-small-zh-v1.5")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="cls", help="Use cls for BGE-style encoders; mean preserves the previous behavior.")
    parser.add_argument(
        "--item-texts",
        nargs="+",
        choices=["joint", "title", "content", "none"],
        default=["joint", "title", "content"],
        help="Item text views to encode. joint preserves the previous text_embeddings.npy output.",
    )
    parser.add_argument("--query", action="store_true", default=True, help="Encode request-level query/context text from processed train/valid/test files.")
    parser.add_argument("--no-query", action="store_false", dest="query", help="Skip request-level query embeddings.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    build_text_embeddings_for_processed(
        Path(args.processed_dir),
        Path(args.qilin_dir),
        model_name=str(args.model_name),
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        pooling=str(args.pooling),
        item_texts=[name for name in args.item_texts if name != "none"],
        encode_query=bool(args.query),
        device=str(args.device),
        skip_existing=False,
        update_metadata=True,
    )


if __name__ == "__main__":
    main()
