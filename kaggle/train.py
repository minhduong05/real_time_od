"""Kaggle-only training and tuning entrypoint."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CONFIG = REPO_ROOT / "configs/project.yaml"


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def parse_args() -> argparse.Namespace:
    config = read_yaml(PROJECT_CONFIG)
    defaults = config["kaggle"]["train"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=config["kaggle"]["data"], help="YOLO dataset YAML on Kaggle.")
    parser.add_argument(
        "--dataset-name",
        choices=list(config["datasets"]),
        default=config["kaggle"]["dataset_name"],
        help="Dataset bucket used when exporting checkpoints.",
    )
    parser.add_argument("--dataset-root", default=None, help="Optional root to generate a YOLO data YAML.")
    parser.add_argument("--train-split", default="images/train", help="Train split path under --dataset-root.")
    parser.add_argument("--val-split", default="images/val", help="Val split path under --dataset-root.")
    parser.add_argument("--test-split", default=None, help="Optional test split path under --dataset-root.")
    parser.add_argument("--model", choices=["yolov8n", "yolov8s", "yolov8n-p2"], required=True)
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--imgsz", type=int, default=defaults["imgsz"])
    parser.add_argument("--batch", type=int, default=defaults["batch"])
    parser.add_argument("--workers", type=int, default=defaults["workers"])
    parser.add_argument("--device", default=str(defaults["device"]))
    parser.add_argument("--project", default=config["kaggle"]["project"])
    parser.add_argument("--name", default=None)
    parser.add_argument("--export-dir", default=config["kaggle"]["export_dir"])
    parser.add_argument("--tune", action="store_true", help="Run Ultralytics hyperparameter tuning.")
    parser.add_argument("--iterations", type=int, default=20, help="Tuning iterations when --tune is used.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved setup without training.")
    return parser.parse_args()


def make_dataset_yaml(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if not args.dataset_root:
        return Path(args.data)

    output = Path("/kaggle/working/generated_visdrone.yaml")
    data = {
        "path": str(Path(args.dataset_root)),
        "train": args.train_split,
        "val": args.val_split,
        "names": config["classes"],
    }
    if args.test_split:
        data["test"] = args.test_split

    write_yaml(output, data)
    return output


def build_model(model_name: str, config: dict[str, Any]):
    from ultralytics import YOLO

    model_def = config["models"]["definitions"][model_name]
    if model_def["type"] == "ultralytics":
        return YOLO(model_def["source"])

    model_cfg = REPO_ROOT / model_def["config"]
    model = YOLO(str(model_cfg))
    model.load(model_def["transfer_from"])
    return model


def export_run(project: Path, run_name: str, export_dir: Path, args: argparse.Namespace) -> None:
    weights_dir = project / run_name / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    model_export_dir = export_dir / args.dataset_name / args.model

    model_export_dir.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy2(best, model_export_dir / "best.pt")
    if last.exists():
        shutil.copy2(last, model_export_dir / "last.pt")

    run_info = {
        "model": args.model,
        "dataset": args.dataset_name,
        "data": str(args.data),
        "project": str(project),
        "run": run_name,
        "best": str(model_export_dir / "best.pt") if best.exists() else None,
        "last": str(model_export_dir / "last.pt") if last.exists() else None,
    }
    write_yaml(model_export_dir / "run_info.yaml", run_info)


def main() -> None:
    config = read_yaml(PROJECT_CONFIG)
    args = parse_args()

    data_yaml = make_dataset_yaml(args, config)
    project = Path(args.project)
    export_dir = Path(args.export_dir)
    run_name = args.name or args.model

    train_args = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "project": str(project),
        "name": run_name,
        "exist_ok": True,
    }

    print("Kaggle training setup")
    print(f"  model: {args.model}")
    print(f"  dataset: {args.dataset_name}")
    print(f"  data: {data_yaml}")
    print(f"  output: {project / run_name}")
    print(f"  export: {export_dir / args.dataset_name / args.model}")
    print(f"  args: {train_args}")

    if args.dry_run:
        return

    model = build_model(args.model, config)
    if args.tune:
        model.tune(
            data=str(data_yaml),
            epochs=args.epochs,
            iterations=args.iterations,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=str(project),
            name=run_name,
            exist_ok=True,
        )
    else:
        model.train(**train_args)

    export_run(project, run_name, export_dir, args)
    print(f"Exported checkpoints to {export_dir}")


if __name__ == "__main__":
    main()
