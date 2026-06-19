"""Model loading helpers for project YOLO variants."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import PROJECT_ROOT


MODEL_CONFIGS = {
    "yolov8n": PROJECT_ROOT / "configs/models/yolov8n.yaml",
    "yolov8s": PROJECT_ROOT / "configs/models/yolov8s.yaml",
    "yolov8-p2": PROJECT_ROOT / "configs/models/yolov8-p2.project.yaml",
    "yolov8n-p2": PROJECT_ROOT / "configs/models/yolov8-p2.project.yaml",
}


def _require_ultralytics():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Ultralytics is not installed. Install dependencies with "
            "`pip install -r requirements.txt` before loading models."
        ) from exc

    return YOLO


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return a dictionary."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root when needed."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_model_config(model_name: str) -> dict[str, Any]:
    """Load a named model config."""
    if model_name not in MODEL_CONFIGS:
        known = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unknown model '{model_name}'. Known models: {known}")

    config = read_yaml(MODEL_CONFIGS[model_name])
    config["_config_path"] = str(MODEL_CONFIGS[model_name])
    return config


def ensure_pretrained_weight(model_name: str) -> Path:
    """Ensure YOLOv8n/YOLOv8s pretrained weights exist under weights/pretrained."""
    config = get_model_config(model_name)
    target = resolve_project_path(config["weights"])
    if target.exists():
        return target

    YOLO = _require_ultralytics()
    target.parent.mkdir(parents=True, exist_ok=True)

    source_name = config["ultralytics_model"]
    model = YOLO(source_name)
    source = Path(getattr(model, "ckpt_path", source_name))

    if not source.exists():
        source = Path(source_name)

    if not source.exists():
        raise FileNotFoundError(
            f"Ultralytics loaded {source_name}, but the downloaded file was not found."
        )

    if source.resolve() != target.resolve():
        shutil.copy2(source, target)

    return target


def ensure_baseline_pretrained_weights() -> dict[str, Path]:
    """Download/cache the pretrained baselines used by the experiments."""
    return {
        "yolov8n": ensure_pretrained_weight("yolov8n"),
        "yolov8s": ensure_pretrained_weight("yolov8s"),
    }


def load_yolov8_p2(transfer: bool = True):
    """Create YOLOv8n-P2 and optionally transfer compatible YOLOv8n weights."""
    YOLO = _require_ultralytics()
    config = get_model_config("yolov8-p2")
    model_cfg = resolve_project_path(config["model_cfg"])
    model = YOLO(str(model_cfg))

    if transfer:
        transfer_from = resolve_project_path(config["transfer_from"])
        if not transfer_from.exists():
            transfer_from = ensure_pretrained_weight("yolov8s")
        model.load(str(transfer_from))

    return model


def load_model(model_name_or_path: str, transfer_p2: bool = True):
    """Load a project model by alias or a direct Ultralytics model path."""
    if model_name_or_path in {"yolov8-p2", "yolov8n-p2"}:
        return load_yolov8_p2(transfer=transfer_p2)

    if model_name_or_path in {"yolov8n", "yolov8s"}:
        weight = ensure_pretrained_weight(model_name_or_path)
        YOLO = _require_ultralytics()
        return YOLO(str(weight))

    YOLO = _require_ultralytics()
    return YOLO(str(resolve_project_path(model_name_or_path)))
