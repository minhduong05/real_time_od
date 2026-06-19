"""YOLO detector wrapper used by the local app."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from realtime_od.config import resolve_project_path


class Detector:
    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        self.weights = resolve_project_path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"Model weights not found: {self.weights}")

        self.model = YOLO(str(self.weights))
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

    def predict_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return results[0].plot()
