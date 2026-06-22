"""Runtime state and model cache for the realtime frontend."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from ultralytics import YOLO

from realtime_od.config import resolve_project_path
from realtime_od.model_registry import DEFAULT_MODEL_KEY, get_model_spec


@dataclass
class RuntimeConfig:
    video: str = ""
    model_key: str = DEFAULT_MODEL_KEY
    options: dict[str, bool] = field(default_factory=dict)
    density_zones: list[dict[str, Any]] = field(default_factory=list)
    count_line: dict[str, dict[str, int]] | None = None
    conf: float = 0.35
    max_box_area_ratio: float = 0.12


class RealtimeState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.config = RuntimeConfig()
        self.models: dict[str, YOLO] = {}

    def get_model(self, model_key: str) -> YOLO:
        model_spec = get_model_spec(model_key)
        if model_spec.key not in self.models:
            weights = resolve_project_path(model_spec.weights)
            if not weights.exists():
                raise FileNotFoundError(f"Model weights not found: {weights}")
            self.models[model_spec.key] = YOLO(str(weights))
        return self.models[model_spec.key]

    def set_config(self, payload: dict[str, Any]) -> None:
        with self.lock:
            self.config = RuntimeConfig(
                video=str(payload.get("video", "")),
                model_key=str(payload.get("model_key", DEFAULT_MODEL_KEY)),
                options=dict(payload.get("options", {})),
                density_zones=list(payload.get("density_zones", [])),
                count_line=payload.get("count_line"),
                conf=float(payload.get("conf", 0.35)),
                max_box_area_ratio=float(payload.get("max_box_area_ratio", 0.12)),
            )

    def get_config(self) -> RuntimeConfig:
        with self.lock:
            return RuntimeConfig(
                video=self.config.video,
                model_key=self.config.model_key,
                options=dict(self.config.options),
                density_zones=list(self.config.density_zones),
                count_line=self.config.count_line,
                conf=self.config.conf,
                max_box_area_ratio=self.config.max_box_area_ratio,
            )
