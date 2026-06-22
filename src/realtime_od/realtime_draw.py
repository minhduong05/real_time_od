"""Drawing and realtime analytics overlays."""

from __future__ import annotations

from collections import Counter, deque

import cv2
import numpy as np

from realtime_od.model_registry import VEHICLE_CLASSES
from realtime_od.types import Detection


def draw_detection(
    frame: np.ndarray,
    detection: Detection,
    trails: dict[int, deque[tuple[int, int]]],
    show_track: bool,
) -> None:
    color = color_for_id(detection.track_id if detection.track_id >= 0 else detection.class_id)
    x1, y1, x2, y2 = (int(value) for value in detection.xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label_id = f" #{detection.track_id}" if show_track and detection.track_id >= 0 else ""
    label = f"{detection.class_name}{label_id} {detection.confidence:.2f}"
    draw_label(frame, label, (x1, y1 - 5), color)

    if show_track and detection.track_id >= 0:
        trails[detection.track_id].append(detection.center)
        points = np.array(trails[detection.track_id], dtype=np.int32)
        if len(points) >= 2:
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)
        cv2.circle(frame, detection.center, 3, color, -1)


def draw_density_zones(
    frame: np.ndarray,
    detections: list[Detection],
    zones: list[dict[str, int]],
) -> None:
    for index, zone in enumerate(zones, start=1):
        points = [(int(point["x"]), int(point["y"])) for point in zone.get("points", [])]
        if len(points) != 4:
            continue
        contour = np.array(points, dtype=np.int32)
        count = sum(
            1 for detection in detections
            if detection.class_name in VEHICLE_CLASSES
            and point_in_polygon(detection.center, points)
        )
        level = "LOW"
        color = (60, 190, 95)
        if count >= 4:
            level = "MED"
            color = (40, 190, 230)
        if count >= 9:
            level = "HIGH"
            color = (45, 80, 230)
        cv2.polylines(frame, [contour], isClosed=True, color=color, thickness=2)
        label_origin = points[0]
        draw_label(frame, f"Zone {index}: {level} ({count})", (label_origin[0] + 6, label_origin[1] + 24), color)


def point_in_polygon(point: tuple[int, int], polygon: list[tuple[int, int]]) -> bool:
    contour = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def update_and_draw_counting(
    frame: np.ndarray,
    detections: list[Detection],
    line: dict[str, dict[str, int]],
    previous_centers: dict[int, tuple[int, int]],
    counted_ids: set[int],
    count_by_class: Counter[str],
) -> None:
    p1 = (int(line["p1"]["x"]), int(line["p1"]["y"]))
    p2 = (int(line["p2"]["x"]), int(line["p2"]["y"]))
    cv2.line(frame, p1, p2, (0, 0, 0), 7, lineType=cv2.LINE_AA)
    cv2.line(frame, p1, p2, (0, 220, 255), 4, lineType=cv2.LINE_AA)
    cv2.circle(frame, p1, 7, (0, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, p2, 7, (0, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, p1, 5, (0, 220, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, p2, 5, (0, 220, 255), -1, lineType=cv2.LINE_AA)

    for detection in detections:
        if detection.track_id < 0:
            continue
        current = detection.center
        previous = previous_centers.get(detection.track_id)
        if previous and detection.track_id not in counted_ids and crossed_line(previous, current, p1, p2):
            counted_ids.add(detection.track_id)
            count_by_class[detection.class_name] += 1
        previous_centers[detection.track_id] = current

    total = sum(count_by_class.values())
    draw_label(frame, f"Count: {total}", (p1[0] + 8, p1[1] - 12), (0, 220, 255))
    y = 92
    for class_name, count in sorted(count_by_class.items()):
        draw_label(frame, f"{class_name}: {count}", (12, y), (0, 220, 255))
        y += 24


def crossed_line(
    previous: tuple[int, int],
    current: tuple[int, int],
    line_a: tuple[int, int],
    line_b: tuple[int, int],
) -> bool:
    prev_side = side_of_line(previous, line_a, line_b)
    curr_side = side_of_line(current, line_a, line_b)
    return prev_side != 0 and curr_side != 0 and prev_side != curr_side


def side_of_line(point: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> int:
    value = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def draw_active_overlay(
    frame: np.ndarray,
    counts: Counter[str],
    frame_index: int,
    options: dict[str, bool],
    model_label: str,
) -> None:
    enabled = ["Detection"]
    enabled.extend(name for name, is_on in options.items() if is_on)
    lines = [f"Model: {model_label}", f"Frame {frame_index}", "On: " + ", ".join(enabled)]
    lines.extend(f"{name}: {count}" for name, count in sorted(counts.items()))

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    line_height = 23
    width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines) + 22
    height = line_height * len(lines) + 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (12 + width, 12 + height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (23, 35 + index * line_height), font, scale, (245, 245, 245), thickness, cv2.LINE_AA)


def draw_label(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    x, y = origin
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(y, height + baseline + 4)
    cv2.rectangle(frame, (x, y - height - baseline - 5), (x + width + 6, y + 3), color, -1)
    cv2.putText(frame, text, (x + 3, y - 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def color_for_id(identifier: int) -> tuple[int, int, int]:
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
    return palette[identifier % len(palette)]
