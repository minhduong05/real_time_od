"""Shared data structures for local traffic analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    frame_index: int
    time_sec: float
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.xyxy
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def csv_row(self) -> dict[str, int | float | str]:
        x1, y1, x2, y2 = self.xyxy
        center_x, center_y = self.center
        return {
            "frame": self.frame_index,
            "time_sec": round(self.time_sec, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 6),
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2),
            "center_x": center_x,
            "center_y": center_y,
        }
