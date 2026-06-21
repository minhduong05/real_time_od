"""Export object-detection-only results for every video in a folder."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="video")
    parser.add_argument("--output-root", default="outputs/detection")
    parser.add_argument("--weights", default="models/Intersection-Flow-5K-Yolov8n-P2/best.pt")
    parser.add_argument("--conf", default="0.35")
    parser.add_argument("--iou", default="0.7")
    parser.add_argument("--imgsz", default="640")
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-box-area-ratio", default="0.12")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    videos = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise SystemExit(f"No videos found in {input_dir}")

    for index, video_path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] Exporting detection for {video_path}")
        subprocess.run(
            [
                sys.executable,
                "app/export_detection.py",
                "--source",
                str(video_path),
                "--output-root",
                args.output_root,
                "--weights",
                args.weights,
                "--conf",
                args.conf,
                "--iou",
                args.iou,
                "--imgsz",
                args.imgsz,
                "--device",
                args.device,
                "--max-box-area-ratio",
                args.max_box_area_ratio,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
