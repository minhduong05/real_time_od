"""Shared training helpers for Ultralytics experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import yaml

from src.detection.model_loader import (
    ensure_pretrained_weight,
    load_yolov8_p2,
    resolve_project_path,
)


def read_experiment_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="Path to an experiment YAML file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size.")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size.")
    parser.add_argument("--device", default=None, help="Training device, e.g. 0, cpu.")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader workers.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved setup only.")
    return parser


def _resolve_training_model(config: dict[str, Any]) -> Path:
    model_name = config.get("model_name")
    model_path = resolve_project_path(config["model"])

    if model_name in {"yolov8n", "yolov8s"}:
        return ensure_pretrained_weight(model_name)

    if model_name in {"yolov8-p2", "yolov8s-p2"}:
        if not model_path.exists():
            model = load_yolov8_p2(transfer=True)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(model_path))
        return model_path

    return model_path


def train_from_config(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    config_path = resolve_project_path(args.config)
    config = read_experiment_config(config_path)

    data = resolve_project_path(config["dataset"])
    model_path = _resolve_training_model(config)
    project = resolve_project_path(config.get("project", "experiments"))
    name = config.get("name", config.get("experiment", model_path.stem))

    train_args = {
        "data": str(data),
        "epochs": args.epochs or config.get("epochs", 100),
        "imgsz": args.imgsz or config.get("imgsz", 640),
        "batch": args.batch or config.get("batch", 16),
        "project": str(project),
        "name": name,
        "exist_ok": True,
    }

    if args.device is not None:
        train_args["device"] = args.device
    if args.workers is not None:
        train_args["workers"] = args.workers

    print(f"experiment: {config.get('experiment', name)}")
    print(f"model: {model_path}")
    print(f"data: {data}")
    print(f"output: {project / name}")
    print(f"train args: {train_args}")

    if args.dry_run:
        return

    model = YOLO(str(model_path))
    model.train(**train_args)
