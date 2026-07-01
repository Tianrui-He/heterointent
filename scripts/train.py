from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterointent.config import load_config
from heterointent.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HeteroIntent-PLE.")
    parser.add_argument("--config", default="configs/qilin_feature_opt_v2_history_compact.yaml")
    parser.add_argument("--resume", default=None, help="Path to best.pt, last.pt, or epoch_XXX.pt for continued training.")
    args = parser.parse_args()
    result = train(load_config(args.config), resume_path=args.resume)
    print(result)


if __name__ == "__main__":
    main()
