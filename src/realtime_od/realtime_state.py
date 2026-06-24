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
    count_lines: list[dict[str, dict[str, int]]] = field(default_factory=list)
    conf: float = 0.35
    max_box_area_ratio: float = 0.12
    enable_logging: bool = False
    stream_width: int = 1280
    jpeg_quality: int = 75


@dataclass
class PlaybackControl:
    paused: bool = False
    finish_requested: bool = False


class RealtimeState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.config = RuntimeConfig()
        self.playback = PlaybackControl()
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
        count_lines = payload.get("count_lines")
        if count_lines is None:
            count_line = payload.get("count_line")
            count_lines = [count_line] if count_line else []

        with self.lock:
            self.config = RuntimeConfig(
                video=str(payload.get("video", "")),
                model_key=str(payload.get("model_key", DEFAULT_MODEL_KEY)),
                options=dict(payload.get("options", {})),
                density_zones=list(payload.get("density_zones", [])),
                count_lines=list(count_lines),
                conf=float(payload.get("conf", 0.35)),
                max_box_area_ratio=float(payload.get("max_box_area_ratio", 0.12)),
                enable_logging=bool(payload.get("enable_logging", False)),
                stream_width=int(payload.get("stream_width", 1280)),
                jpeg_quality=int(payload.get("jpeg_quality", 75)),
            )
            self.playback = PlaybackControl()

    def get_config(self) -> RuntimeConfig:
        with self.lock:
            return RuntimeConfig(
                video=self.config.video,
                model_key=self.config.model_key,
                options=dict(self.config.options),
                density_zones=list(self.config.density_zones),
                count_lines=list(self.config.count_lines),
                conf=self.config.conf,
                max_box_area_ratio=self.config.max_box_area_ratio,
                enable_logging=self.config.enable_logging,
                stream_width=self.config.stream_width,
                jpeg_quality=self.config.jpeg_quality,
            )

    def set_playback_action(self, action: str) -> PlaybackControl:
        with self.lock:
            if action == "pause":
                self.playback.paused = True
            elif action == "resume":
                self.playback.paused = False
            elif action == "finish":
                self.playback.finish_requested = True
                self.playback.paused = False
            else:
                raise ValueError(f"Unknown playback action: {action}")
            return PlaybackControl(
                paused=self.playback.paused,
                finish_requested=self.playback.finish_requested,
            )

    def get_playback(self) -> PlaybackControl:
        with self.lock:
            return PlaybackControl(
                paused=self.playback.paused,
                finish_requested=self.playback.finish_requested,
            )
