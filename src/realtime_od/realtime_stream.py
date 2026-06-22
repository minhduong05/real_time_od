"""Video reading, YOLO inference, and MJPEG streaming."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from realtime_od.config import resolve_project_path
from realtime_od.model_registry import get_model_spec
from realtime_od.realtime_draw import (
    draw_active_overlay,
    draw_density_zones,
    draw_detection,
    update_and_draw_counting,
)
from realtime_od.realtime_state import RealtimeState, RuntimeConfig
from realtime_od.types import Detection


def read_video_info(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return {
            "name": path.name,
            "width": width,
            "height": height,
            "fps": fps,
            "frames": frames,
            "duration_sec": frames / fps if fps else 0,
        }
    finally:
        capture.release()


def process_stream(state: RealtimeState) -> Iterable[bytes]:
    config = state.get_config()
    video_path = resolve_project_path(config.video)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    model_spec = get_model_spec(config.model_key)
    model = state.get_model(model_spec.key)
    use_tracking = config.options.get("track", False) or config.options.get("counting", False)
    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=30))
    previous_centers: dict[int, tuple[int, int]] = {}
    counted_ids: set[int] = set()
    count_by_class: Counter[str] = Counter()
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            results = (
                model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=config.conf,
                    iou=0.7,
                    imgsz=640,
                    device=0,
                    verbose=False,
                )
                if use_tracking
                else model.predict(
                    source=frame,
                    conf=config.conf,
                    iou=0.7,
                    imgsz=640,
                    device=0,
                    verbose=False,
                )
            )
            detections = extract_detections(results[0], frame_index, frame_index / fps, frame.shape, config)
            annotated = frame.copy()
            active_counts = Counter(d.class_name for d in detections)

            if config.options.get("density"):
                draw_density_zones(annotated, detections, config.density_zones)

            if config.options.get("counting") and config.count_line:
                update_and_draw_counting(
                    annotated,
                    detections,
                    config.count_line,
                    previous_centers,
                    counted_ids,
                    count_by_class,
                )

            for detection in detections:
                draw_detection(annotated, detection, trails, show_track=config.options.get("track", False))

            draw_active_overlay(annotated, active_counts, frame_index, config.options, model_spec.label)
            ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                break
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            frame_index += 1
    finally:
        capture.release()


def extract_detections(
    result: object,
    frame_index: int,
    time_sec: float,
    frame_shape: tuple[int, ...],
    config: RuntimeConfig,
) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    names = result.names if hasattr(result, "names") else {}
    xyxy_values = boxes.xyxy.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    if boxes.id is None:
        track_ids = np.arange(len(xyxy_values), dtype=int) * -1 - 1
    else:
        track_ids = boxes.id.cpu().numpy().astype(int)

    frame_height, frame_width = frame_shape[:2]
    frame_area = max(frame_width * frame_height, 1)
    detections = []
    for xyxy, track_id, class_id, confidence in zip(xyxy_values, track_ids, class_ids, confidences):
        x1, y1, x2, y2 = (float(value) for value in xyxy)
        box_area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / frame_area
        if box_area_ratio > config.max_box_area_ratio:
            continue
        class_name = str(names.get(int(class_id), f"class_{int(class_id)}"))
        detections.append(
            Detection(
                frame_index=frame_index,
                time_sec=time_sec,
                track_id=int(track_id),
                class_id=int(class_id),
                class_name=class_name,
                confidence=float(confidence),
                xyxy=(x1, y1, x2, y2),
            )
        )
    return detections
