"""Check model construction and pretrained transfer before training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CONFIG = REPO_ROOT / "configs/project.yaml"


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def parse_args() -> argparse.Namespace:
    config = read_yaml(PROJECT_CONFIG)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(config["models"]["definitions"]), required=True)
    parser.add_argument("--detailed", action="store_true", help="Print detailed Ultralytics model info.")
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO

    config = read_yaml(PROJECT_CONFIG)
    args = parse_args()
    model_def = config["models"]["definitions"][args.model]

    print("Model check")
    print(f"  model: {args.model}")
    print(f"  type: {model_def['type']}")

    if model_def["type"] == "ultralytics":
        print(f"  source: {model_def['source']}")
        model = YOLO(model_def["source"])
    else:
        model_cfg = REPO_ROOT / model_def["config"]
        print(f"  config: {model_cfg}")
        print(f"  transfer_from: {model_def['transfer_from']}")
        model = YOLO(str(model_cfg))
        model.load(model_def["transfer_from"])

    print("\nModel summary")
    model.info(detailed=args.detailed)


if __name__ == "__main__":
    main()
