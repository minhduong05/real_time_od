"""Per-frame realtime run logging."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from realtime_od.config import resolve_project_path
from realtime_od.model_registry import ModelSpec
from realtime_od.realtime_state import RuntimeConfig
from realtime_od.types import Detection


FRAME_COLUMNS = [
    "frame",
    "time_sec",
    "model",
    "mode",
    "current_fps",
    "source_fps",
    "detections",
    "active_counts_json",
]

DETECTION_COLUMNS = [
    "frame",
    "time_sec",
    "track_id",
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


def safe_stem(value: str) -> str:
    stem = Path(value).stem if Path(value).suffix else value
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "video"


class RealtimeRunLogger:
    def __init__(self, output_dir: Path, model_spec: ModelSpec, mode: str) -> None:
        self.output_dir = output_dir
        self.model_spec = model_spec
        self.mode = mode
        self.frame_file = (output_dir / "frame_log.csv").open("w", newline="", encoding="utf-8")
        self.detection_file = (output_dir / "detections.csv").open("w", newline="", encoding="utf-8")
        self.frame_writer = csv.DictWriter(self.frame_file, fieldnames=FRAME_COLUMNS)
        self.detection_writer = csv.DictWriter(self.detection_file, fieldnames=DETECTION_COLUMNS)
        self.frame_writer.writeheader()
        self.detection_writer.writeheader()

    @classmethod
    def create(cls, video_path: Path, config: RuntimeConfig, model_spec: ModelSpec, mode: str) -> "RealtimeRunLogger":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = resolve_project_path("outputs/logs") / safe_stem(video_path.name) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "video": str(video_path),
                    "model": asdict(model_spec),
                    "mode": mode,
                    "options": config.options,
                    "density_zones": config.density_zones,
                    "count_lines": config.count_lines,
                    "conf": config.conf,
                    "max_box_area_ratio": config.max_box_area_ratio,
                    "stream_width": config.stream_width,
                    "jpeg_quality": config.jpeg_quality,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return cls(output_dir, model_spec, mode)

    def log_frame(
        self,
        frame_index: int,
        time_sec: float,
        current_fps: float,
        source_fps: float,
        detections: list[Detection],
        active_counts: Counter[str],
    ) -> None:
        self.frame_writer.writerow(
            {
                "frame": frame_index,
                "time_sec": round(time_sec, 6),
                "model": self.model_spec.key,
                "mode": self.mode,
                "current_fps": round(current_fps, 4),
                "source_fps": round(source_fps, 4),
                "detections": len(detections),
                "active_counts_json": json.dumps(dict(sorted(active_counts.items())), ensure_ascii=False),
            }
        )
        for detection in detections:
            x1, y1, x2, y2 = detection.xyxy
            center_x, center_y = detection.center
            self.detection_writer.writerow(
                {
                    "frame": detection.frame_index,
                    "time_sec": round(detection.time_sec, 6),
                    "track_id": detection.track_id,
                    "class_id": detection.class_id,
                    "class_name": detection.class_name,
                    "confidence": round(detection.confidence, 6),
                    "x1": round(x1, 3),
                    "y1": round(y1, 3),
                    "x2": round(x2, 3),
                    "y2": round(y2, 3),
                    "center_x": center_x,
                    "center_y": center_y,
                }
            )
        self.frame_file.flush()
        self.detection_file.flush()

    def close(self) -> None:
        self.frame_file.close()
        self.detection_file.close()
