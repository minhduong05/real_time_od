"""Check that the local project layout is ready for inference."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_od.config import read_config, resolve_project_path
from realtime_od.model_registry import MODEL_REGISTRY


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def main() -> None:
    config = read_config()
    video_dir = resolve_project_path("video")
    videos = []
    if video_dir.exists():
        videos = sorted(
            path for path in video_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )

    print(f"Project: {config['project']['name']}")
    print(f"Classes: {len(config['classes'])}")
    print(f"Available train models: {', '.join(config['models']['definitions'])}")
    print("Model folders:")

    for dataset_name, dataset_config in config["datasets"].items():
        model_dir = resolve_project_path(dataset_config["model_dir"])
        print(f"  {dataset_name}: {model_dir}")

    print("VisDrone checkpoints:")
    for model in MODEL_REGISTRY.values():
        weights = resolve_project_path(model.weights)
        print(f"  {model.label}: {weights} ({'ok' if weights.exists() else 'missing'})")

    print(f"Local videos: {len(videos)} in {video_dir}")
    for video in videos[:5]:
        print(f"  {video.name}")
    if len(videos) > 5:
        print(f"  ... {len(videos) - 5} more")

    print("Generated outputs:")
    print("  outputs/detection/  batch annotated videos and CSV files")
    print("  outputs/logs/       optional realtime CSV logs")

    print("Run realtime frontend:")
    print("  python app/realtime_front.py")
    print("Export object detection for a local video, for example:")
    print("  python app/detection.py --source path/to/video.mp4 --model yolov8n-p2 --device 0")


if __name__ == "__main__":
    main()
