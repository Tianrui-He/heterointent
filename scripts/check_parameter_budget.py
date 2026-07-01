from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterointent.config import load_config
from heterointent.data.schema import FEATURE_PREFIXES
from heterointent.models import HeteroIntentPLE
from heterointent.utils import read_json


def _module_key(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] in {"item_encoder", "user_encoder"}:
        return ".".join(parts[:2])
    return parts[0]


def inspect_budget(config_path: Path, budget_mb: float) -> dict[str, object]:
    config = load_config(config_path)
    data_cfg = config["data"]
    processed_dir = Path(data_cfg["processed_dir"])
    metadata = read_json(processed_dir / data_cfg.get("metadata_file", "metadata.json"))
    model = HeteroIntentPLE(metadata, config)

    total_params = 0
    trainable_params = 0
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "trainable": 0})
    for name, param in model.named_parameters():
        count = int(param.numel())
        total_params += count
        if param.requires_grad:
            trainable_params += count
        key = _module_key(name)
        groups[key]["total"] += count
        if param.requires_grad:
            groups[key]["trainable"] += count

    fp32_mb = total_params * 4 / 1024 / 1024
    trainable_fp32_mb = trainable_params * 4 / 1024 / 1024
    breakdown = [
        {
            "module": module,
            "total_params": values["total"],
            "trainable_params": values["trainable"],
            "fp32_mb": values["total"] * 4 / 1024 / 1024,
        }
        for module, values in sorted(groups.items(), key=lambda item: item[1]["total"], reverse=True)
    ]
    return {
        "config": str(config_path),
        "processed_dir": str(processed_dir),
        "budget_mb": float(budget_mb),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "fp32_mb": fp32_mb,
        "trainable_fp32_mb": trainable_fp32_mb,
        "under_budget": fp32_mb <= budget_mb,
        "metadata_dims": {f"{group}_dim": int(metadata.get(f"{group}_dim", 0)) for group in FEATURE_PREFIXES},
        "breakdown": breakdown,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check HeteroIntent model parameter size against a deployment budget.")
    parser.add_argument("--config", default="configs/qilin_feature_opt_v2_history_compact.yaml")
    parser.add_argument("--budget-mb", type=float, default=800.0)
    parser.add_argument("--output", default=None, help="Optional JSON path for the budget report.")
    parser.add_argument("--fail-over-budget", action="store_true")
    args = parser.parse_args()

    report = inspect_budget(Path(args.config), budget_mb=float(args.budget_mb))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    if args.fail_over_budget and not report["under_budget"]:
        raise SystemExit(f"Model exceeds budget: {report['fp32_mb']:.2f} MB > {report['budget_mb']:.2f} MB")


if __name__ == "__main__":
    main()
