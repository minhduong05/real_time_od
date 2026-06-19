"""Run traffic detection on a video and save an annotated output."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from tqdm import tqdm

from src.detection.detector import Detector
from src.traffic.density import estimate_density
from src.traffic.statistics import count_by_class
from src.utils.video import make_video_writer, open_video
from src.visualization.draw import draw_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a trained .pt model.")
    parser.add_argument("--source", required=True, help="Video path, stream URL, or camera index.")
    parser.add_argument("--output", default="outputs/demo.mp4", help="Annotated output video path.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for quick tests.")
    return parser.parse_args()


def main():
    args = parse_args()
    detector = Detector(args.model, conf=args.conf, iou=args.iou)
    capture = open_video(args.source)

    fps = capture.get(5) or 30.0
    width = int(capture.get(3))
    height = int(capture.get(4))
    total_frames = int(capture.get(7)) or None
    if args.max_frames:
        total_frames = min(total_frames or args.max_frames, args.max_frames)

    writer = make_video_writer(args.output, fps=fps, width=width, height=height)
    names = getattr(detector.model, "names", {})

    try:
        progress = tqdm(total=total_frames, desc="Annotating video")
        frame_count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if args.max_frames is not None and frame_count >= args.max_frames:
                break

            result = detector.predict(frame, imgsz=args.imgsz, verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes is not None else []
            class_ids = result.boxes.cls.cpu().numpy().tolist() if result.boxes is not None else []
            confidences = result.boxes.conf.cpu().numpy().tolist() if result.boxes is not None else []

            density = estimate_density(boxes, roi_area=float(width * height))
            class_counts = count_by_class(class_ids, names=names)
            stats = {
                "vehicles": len(boxes),
                "density": f"{density:.2f}",
            }
            stats.update(class_counts)

            writer.write(
                draw_overlay(
                    frame,
                    boxes=boxes,
                    class_ids=class_ids,
                    confidences=confidences,
                    names=names,
                    stats=stats,
                )
            )
            frame_count += 1
            progress.update(1)
    finally:
        capture.release()
        writer.release()
        progress.close()

    print(f"Saved annotated video: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
