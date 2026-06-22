"""Check that the local project layout is ready for inference."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realtime_od.config import read_config, resolve_project_path


def main() -> None:
    config = read_config()

    print(f"Project: {config['project']['name']}")
    print(f"Classes: {len(config['classes'])}")
    print(f"Available train models: {', '.join(config['models']['definitions'])}")
    print("Model folders:")

    for dataset_name, dataset_config in config["datasets"].items():
        model_dir = resolve_project_path(dataset_config["model_dir"])
        print(f"  {dataset_name}: {model_dir}")

    print("Predict by passing the exact weight path, for example:")
    print("  python app/run.py --weights models/Top-View-Vehicle-Detection-Yolov8n-P2/best.pt --source 0")
    print("Export object detection for a local video, for example:")
    print("  python app/export_detection.py --source path/to/video.mp4 --device 0")


if __name__ == "__main__":
    main()
