"""YOLO detector wrapper."""

from __future__ import annotations

from src.detection.model_loader import load_model


class Detector:
    """Thin wrapper for loading and running YOLO models."""

    def __init__(self, model_name_or_path: str, conf: float = 0.25, iou: float = 0.7):
        self.model_name_or_path = model_name_or_path
        self.conf = conf
        self.iou = iou
        self.model = load_model(model_name_or_path)

    def predict(self, frame, **kwargs):
        """Run detection on one frame."""
        options = {"conf": self.conf, "iou": self.iou}
        options.update(kwargs)
        return self.model.predict(frame, **options)
