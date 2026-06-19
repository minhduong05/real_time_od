"""Drawing helpers for detections, tracks, counts, and alerts."""

from __future__ import annotations

import cv2


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = [
        (56, 189, 248),
        (34, 197, 94),
        (250, 204, 21),
        (249, 115, 22),
        (244, 63, 94),
        (168, 85, 247),
        (20, 184, 166),
        (132, 204, 22),
        (251, 146, 60),
        (59, 130, 246),
    ]
    return palette[class_id % len(palette)]


def draw_overlay(frame, boxes=None, class_ids=None, confidences=None, names=None, stats=None):
    """Draw detections and summary statistics on a frame."""
    output = frame.copy()
    boxes = boxes or []
    class_ids = class_ids or []
    confidences = confidences or []
    names = names or {}

    for box, class_id, confidence in zip(boxes, class_ids, confidences):
        x1, y1, x2, y2 = [int(value) for value in box]
        color = _color_for_class(int(class_id))
        label = names.get(int(class_id), str(int(class_id)))
        text = f"{label} {float(confidence):.2f}"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_w, text_h = text_size
        cv2.rectangle(output, (x1, max(0, y1 - text_h - 8)), (x1 + text_w + 6, y1), color, -1)
        cv2.putText(
            output,
            text,
            (x1 + 3, max(12, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (15, 23, 42),
            1,
            cv2.LINE_AA,
        )

    if stats:
        y = 26
        for key, value in stats.items():
            cv2.putText(
                output,
                f"{key}: {value}",
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 28

    return output
