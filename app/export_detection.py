"""Export object-detection-only results for one local video."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_od.config import read_config
from realtime_od.detection_export import run_detection_export


def safe_stem(source: str) -> str:
    path = Path(source)
    stem = path.stem if path.suffix else source
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "video"


def save_sample_frame(video_path: Path, sample_path: Path) -> bool:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return False
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
        ok, frame = capture.read()
        if not ok:
            return False
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(sample_path), frame))
    finally:
        capture.release()


def parse_args() -> argparse.Namespace:
    config = read_config()
    inference = config["inference"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="models/Intersection-Flow-5K-Yolov8n-P2/best.pt")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-root", default="outputs/detection")
    parser.add_argument("--output-dir")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=inference["iou"])
    parser.add_argument("--imgsz", type=int, default=inference["imgsz"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-box-area-ratio", type=float, default=0.12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / safe_stem(args.source)
    output_video = output_dir / "annotated.mp4"
    output_csv = output_dir / "detections.csv"
    output_summary = output_dir / "summary.json"
    sample_frame = output_dir / "sample.jpg"

    summary = run_detection_export(
        weights=args.weights,
        source=args.source,
        output_video=output_video,
        output_csv=output_csv,
        output_summary=output_summary,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        max_box_area_ratio=args.max_box_area_ratio,
    )
    sample_saved = save_sample_frame(output_video, sample_frame)

    print(f"Output folder: {output_dir}")
    print(f"Annotated video: {output_video}")
    print(f"Detection CSV: {output_csv}")
    print(f"Summary JSON: {output_summary}")
    print(f"Sample frame: {sample_frame if sample_saved else 'not saved'}")
    print(f"Frames: {summary.frames}")
    print(f"Duration seconds: {summary.duration_sec:.2f}")
    print(f"Detection rows: {summary.detection_rows}")
    if summary.class_rows:
        print("Class rows:")
        for class_name, count in summary.class_rows.items():
            print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()
