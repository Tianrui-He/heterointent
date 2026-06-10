from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def find_image(image_root: Path, raw_item_id: int) -> Path | None:
    for suffix in [".jpg", ".jpeg", ".png", ".webp"]:
        p = image_root / f"{raw_item_id}{suffix}"
        if p.exists():
            return p
    nested = list(image_root.glob(f"**/{raw_item_id}.*"))
    return nested[0] if nested else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CLIP/SigLIP visual embeddings for processed Qilin items with local images.")
    parser.add_argument("--image-root", required=True, help="Directory containing local item images named by raw note id, for example 123.jpg.")
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "processed" / "qilin_full"))
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    try:
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise SystemExit("Please install transformers and Pillow first: pip install transformers Pillow") from exc

    processed = Path(args.processed_dir)
    image_root = Path(args.image_root)
    item_map = pd.read_parquet(processed / "item_id_map.parquet").sort_values("item_id")
    rows = []
    for row in item_map.itertuples(index=False):
        path = find_image(image_root, int(row.raw_item_id))
        if path is not None:
            rows.append({"item_id": int(row.item_id), "raw_item_id": int(row.raw_item_id), "path": str(path)})
    items = pd.DataFrame(rows)
    if items.empty:
        raise SystemExit(f"No images found under {image_root}. Check file names or pass the correct --image-root.")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    all_embs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(items), args.batch_size), desc="visual embeddings"):
            batch_rows = items.iloc[start:start + args.batch_size]
            images = []
            for path in batch_rows["path"]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt").to(device)
            if hasattr(model, "get_image_features"):
                emb = model.get_image_features(**inputs)
            else:
                out = model(**inputs)
                emb = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
            emb = torch.nn.functional.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().numpy().astype("float32"))
    values = np.concatenate(all_embs, axis=0)
    np.save(processed / "image_embeddings.npy", values)
    np.save(processed / "image_embedding_item_ids.npy", items["item_id"].to_numpy(dtype="int64"))
    items.to_parquet(processed / "image_embedding_items.parquet", index=False)
    print(f"wrote image embeddings {values.shape} for {len(items)} items to {processed}")


if __name__ == "__main__":
    main()
