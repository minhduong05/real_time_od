"""Run local real-time object detection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_od.config import read_config
from realtime_od.video import run_video


def parse_args() -> argparse.Namespace:
    config = read_config()
    inference = config["inference"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to the .pt model to use for prediction.")
    parser.add_argument("--source", default=str(inference["source"]))
    parser.add_argument("--conf", type=float, default=inference["conf"])
    parser.add_argument("--iou", type=float, default=inference["iou"])
    parser.add_argument("--imgsz", type=int, default=inference["imgsz"])
    parser.add_argument("--device", default=inference.get("device"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_video(
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
