"""Prepare pretrained YOLO baselines and the YOLOv8s-P2 transfer model."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src.detection.model_loader import (
    ensure_baseline_pretrained_weights,
    load_yolov8_p2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-p2",
        action="store_true",
        help="Instantiate YOLOv8s-P2 and transfer compatible YOLOv8s weights.",
    )
    parser.add_argument(
        "--save-p2",
        type=Path,
        default=None,
        help="Optional path to save the initialized YOLOv8s-P2 checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = ensure_baseline_pretrained_weights()

    for name, path in weights.items():
        print(f"{name}: {path}")

    if args.check_p2 or args.save_p2:
        model = load_yolov8_p2(transfer=True)
        print("yolov8-p2: created from project P2 architecture config")
        print("yolov8-p2: compatible weights transferred from YOLOv8s")

        if args.save_p2:
            args.save_p2.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(args.save_p2))
            print(f"yolov8-p2 initialized checkpoint: {args.save_p2}")


if __name__ == "__main__":
    main()
