"""Run all VisDrone research experiments sequentially."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from _train_common import train_from_config


VISDRONE_CONFIGS = {
    "yolov8n": "configs/experiments/visdrone_yolov8n.yaml",
    "yolov8s": "configs/experiments/visdrone_yolov8s.yaml",
    "yolov8p2": "configs/experiments/visdrone_yolov8p2.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        help="VisDrone YOLO dataset YAML path, e.g. /kaggle/input/visdrone-yolo/visdrone.yaml.",
    )
    parser.add_argument(
        "--project",
        default="/kaggle/working/experiments/visdrone",
        help="Output directory for VisDrone experiment runs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=VISDRONE_CONFIGS,
        default=["yolov8n", "yolov8s", "yolov8p2"],
        help="Models to train in order.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for model_name in args.models:
        print(f"\n=== VisDrone experiment: {model_name} ===")
        train_args = argparse.Namespace(
            config=VISDRONE_CONFIGS[model_name],
            data=args.data,
            project=args.project,
            name=None,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            dry_run=args.dry_run,
        )
        train_from_config(train_args)


if __name__ == "__main__":
    main()
