"""Video and webcam inference loop."""

from __future__ import annotations

from pathlib import Path

import cv2

from realtime_od.detector import Detector


def normalize_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def run_video(
    weights: str | Path,
    source: str,
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
    window_name: str = "Real-Time Object Detection",
) -> None:
    detector = Detector(weights=weights, conf=conf, iou=iou, imgsz=imgsz, device=device)
    capture = cv2.VideoCapture(normalize_source(source))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            annotated = detector.predict_frame(frame)
            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
