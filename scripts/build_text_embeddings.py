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
    notes["text"] = notes["note_title"].fillna("").astype(str) + " " + notes["note_content"].fillna("").astype(str)
    return notes[["item_id", "raw_item_id", "text"]].sort_values("item_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BGE/Transformer text embeddings for processed Qilin items.")
    parser.add_argument("--qilin-dir", default=str(ROOT / "data" / "raw" / "Qilin"))
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "processed" / "qilin_full"))
    parser.add_argument("--model-name", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean", help="Use cls for BGE-style encoders; mean preserves the previous behavior.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Please install transformers first: pip install transformers") from exc

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    processed = Path(args.processed_dir)
    notes = load_notes(Path(args.qilin_dir), processed / "item_id_map.parquet")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    all_embs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(notes), args.batch_size), desc="text embeddings"):
            texts = notes["text"].iloc[start:start + args.batch_size].tolist()
            batch = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(device)
            out = model(**batch)
            emb = pool_hidden_state(out.last_hidden_state, batch["attention_mask"], args.pooling)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            all_embs.append(emb.float().cpu().numpy().astype("float32"))
    values = np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, 0), dtype="float32")
    np.save(processed / "text_embeddings.npy", values)
    np.save(processed / "text_embedding_item_ids.npy", notes["item_id"].to_numpy(dtype="int64"))
    notes[["item_id", "raw_item_id"]].to_parquet(processed / "text_embedding_items.parquet", index=False)
    (processed / "text_embedding_config.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "pooling": args.pooling,
                "max_length": args.max_length,
                "num_items": int(len(notes)),
                "embedding_shape": list(values.shape),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote text embeddings {values.shape} to {processed} with pooling={args.pooling}")


if __name__ == "__main__":
    main()

