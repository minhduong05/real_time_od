"""VisDrone model registry used by local apps."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MODEL_KEY = "yolov8n-p2"

VEHICLE_CLASSES = {"bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    weights: str
    description: str


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "yolov8n": ModelSpec(
        key="yolov8n",
        label="YOLOv8n",
        weights="models/VisDrone/yolov8n/best.pt",
        description="Baseline nhẹ, tốc độ tốt.",
    ),
    "yolov8n-p2": ModelSpec(
        key="yolov8n-p2",
        label="YOLOv8n-P2",
        weights="models/VisDrone/yolov8n-p2/best.pt",
        description="P2 head cho object nhỏ trong ảnh.",
    ),
    "yolov8s": ModelSpec(
        key="yolov8s",
        label="YOLOv8s",
        weights="models/VisDrone/yolov8s/best.pt",
        description="Model lớn hơn để so sánh accuracy/speed.",
    ),
}


def get_model_spec(model_key: str | None) -> ModelSpec:
    return MODEL_REGISTRY.get(model_key or DEFAULT_MODEL_KEY, MODEL_REGISTRY[DEFAULT_MODEL_KEY])
