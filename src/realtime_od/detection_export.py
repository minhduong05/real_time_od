"""Detection-only video export utilities."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from realtime_od.config import resolve_project_path
from realtime_od.types import Detection
from realtime_od.video import normalize_source


CSV_COLUMNS = [
    "frame",
    "time_sec",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
]


@dataclass(frozen=True)
class DetectionExportSummary:
    source: str
    output_video: str
    output_csv: str
    frames: int
    fps: float
    duration_sec: float
    detection_rows: int
    class_rows: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "output_video": self.output_video,
            "output_csv": self.output_csv,
            "frames": self.frames,
            "fps": round(self.fps, 4),
            "duration_sec": round(self.duration_sec, 4),
            "detection_rows": self.detection_rows,
            "class_rows": self.class_rows,
        }


def extract_detections(
    result: object,
    frame_index: int,
    time_sec: float,
    max_box_area_ratio: float,
    frame_shape: tuple[int, ...],
) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    names = result.names if hasattr(result, "names") else {}
    xyxy_values = boxes.xyxy.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()

    frame_height, frame_width = frame_shape[:2]
    frame_area = max(frame_width * frame_height, 1)
    detections = []
    for xyxy, class_id, confidence in zip(xyxy_values, class_ids, confidences):
        x1, y1, x2, y2 = (float(value) for value in xyxy)
        box_area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / frame_area
        if box_area_ratio > max_box_area_ratio:
            continue
        detections.append(
            Detection(
                frame_index=frame_index,
                time_sec=time_sec,
                track_id=-1,
                class_id=int(class_id),
                class_name=str(names.get(int(class_id), f"class_{int(class_id)}")),
                confidence=float(confidence),
                xyxy=(x1, y1, x2, y2),
            )
        )
    return detections


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = (
        (52, 152, 219),
        (46, 204, 113),
        (241, 196, 15),
        (231, 76, 60),
        (155, 89, 182),
        (26, 188, 156),
        (230, 126, 34),
        (149, 165, 166),
    )
    return palette[class_id % len(palette)]


def draw_label(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    x, y = origin
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(y, height + baseline + 4)
    cv2.rectangle(frame, (x, y - height - baseline - 5), (x + width + 6, y + 3), color, -1)
    cv2.putText(frame, text, (x + 3, y - 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_detection(frame: np.ndarray, detection: Detection) -> None:
    color = color_for_class(detection.class_id)
    x1, y1, x2, y2 = (int(value) for value in detection.xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    draw_label(frame, f"{detection.class_name} {detection.confidence:.2f}", (x1, y1 - 5), color)


def draw_overlay(frame: np.ndarray, counts: Counter[str], frame_index: int, fps: float) -> None:
    lines = [f"Frame {frame_index}", f"Video FPS {fps:.1f}"]
    lines.extend(f"{name}: {count}" for name, count in sorted(counts.items()))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 1
    line_height = 23
    width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines) + 22
    height = line_height * len(lines) + 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (12 + width, 12 + height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (23, 35 + index * line_height), font, scale, (245, 245, 245), thickness, cv2.LINE_AA)


def run_detection_export(
    weights: str | Path,
    source: str,
    output_video: str | Path,
    output_csv: str | Path,
    output_summary: str | Path,
    conf: float = 0.35,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str | None = "0",
    max_box_area_ratio: float = 0.12,
) -> DetectionExportSummary:
    weights_path = resolve_project_path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    output_video_path = resolve_project_path(output_video)
    output_csv_path = resolve_project_path(output_csv)
    output_summary_path = resolve_project_path(output_summary)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(normalize_source(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video writer: {output_video_path}")

    model = YOLO(str(weights_path))
    frame_index = 0
    detection_rows = 0
    class_rows: Counter[str] = Counter()

    try:
        with output_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            csv_writer.writeheader()

            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                results = model.predict(
                    source=frame,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    device=device,
                    verbose=False,
                )
                detections = extract_detections(
                    results[0],
                    frame_index=frame_index,
                    time_sec=frame_index / fps,
                    max_box_area_ratio=max_box_area_ratio,
                    frame_shape=frame.shape,
                )
                annotated = frame.copy()
                active_counts = Counter(detection.class_name for detection in detections)
                for detection in detections:
                    draw_detection(annotated, detection)
                    csv_writer.writerow(detection.csv_row())
                    detection_rows += 1
                    class_rows[detection.class_name] += 1
                draw_overlay(annotated, active_counts, frame_index, fps)
                writer.write(annotated)
                frame_index += 1
    finally:
        capture.release()
        writer.release()

    summary = DetectionExportSummary(
        source=str(source),
        output_video=str(output_video_path),
        output_csv=str(output_csv_path),
        frames=frame_index,
        fps=float(fps),
        duration_sec=frame_index / fps if fps else 0.0,
        detection_rows=detection_rows,
        class_rows=dict(sorted(class_rows.items())),
    )
    output_summary_path.write_text(
        json.dumps(summary.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
