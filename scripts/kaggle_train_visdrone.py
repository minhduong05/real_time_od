"""Kaggle-friendly VisDrone training entrypoint.

This script keeps the repository as code-only and reads the dataset directly
from /kaggle/input. It can also configure W&B before launching Ultralytics.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml
from _train_common import train_from_config


VISDRONE_CONFIGS = {
    "yolov8n": "configs/experiments/visdrone_yolov8n.yaml",
    "yolov8n-p2": "configs/experiments/visdrone_yolov8n_p2.yaml",
    "yolov8s": "configs/experiments/visdrone_yolov8s.yaml",
    "yolov8p2": "configs/experiments/visdrone_yolov8n_p2.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=None,
        help="VisDrone YOLO dataset YAML. If omitted, search /kaggle/input for visdrone.yaml.",
    )
    parser.add_argument(
        "--input-root",
        default="/kaggle/input",
        help="Root used by --data auto-discovery.",
    )
    parser.add_argument(
        "--project",
        default="/kaggle/working/experiments/visdrone",
        help="Kaggle output directory for Ultralytics runs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=VISDRONE_CONFIGS,
        default=["yolov8n"],
        help="Models to train in order.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--device-yolov8s",
        default="0,1",
        help="Device override for YOLOv8s. Defaults to both Kaggle T4 GPUs.",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Enable W&B and log runs to this W&B project.",
    )
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B team/user.")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="W&B mode used when --wandb-project is set.",
    )
    parser.add_argument(
        "--wandb-key-secret",
        default="WANDB_API_KEY",
        help="Kaggle secret/env var name that stores the W&B API key.",
    )
    return parser.parse_args()


def find_visdrone_yaml(input_root: str | Path) -> Path:
    candidates = sorted(Path(input_root).rglob("visdrone.yaml"))
    if not candidates:
        raise FileNotFoundError(
            f"No visdrone.yaml found under {input_root}. Attach the VisDrone dataset "
            "to the Kaggle notebook or pass --data explicitly."
        )

    if len(candidates) > 1:
        print("Multiple visdrone.yaml files found; using the first one:")
        for path in candidates:
            print(f"  {path}")

    return candidates[0]


def _split_path_exists(root: Path, split_value: str | list[str]) -> bool:
    if isinstance(split_value, list):
        return all((root / item).exists() for item in split_value)
    return (root / split_value).exists()


def prepare_kaggle_dataset_yaml(data_yaml: str | Path) -> Path:
    """Rewrite Kaggle input YAML when its relative root points outside /kaggle/input."""
    data_yaml = Path(data_yaml)
    with data_yaml.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    yaml_parent = data_yaml.parent
    configured_root = Path(data.get("path", ""))
    if configured_root.is_absolute():
        resolved_root = configured_root
    else:
        resolved_root = (data_yaml.parent / configured_root).resolve()

    train_ok = _split_path_exists(resolved_root, data.get("train", ""))
    val_ok = _split_path_exists(resolved_root, data.get("val", ""))
    if train_ok and val_ok:
        return data_yaml

    fallback_root = yaml_parent.resolve()
    train_ok = _split_path_exists(fallback_root, data.get("train", ""))
    val_ok = _split_path_exists(fallback_root, data.get("val", ""))
    if not (train_ok and val_ok):
        print(f"Dataset YAML root from file does not exist: {resolved_root}")
        print(f"Fallback YAML parent also does not match train/val paths: {fallback_root}")
        return data_yaml

    data["path"] = str(fallback_root)
    runtime_dir = Path("/kaggle/working/real_time_od/runtime")
    if not runtime_dir.exists():
        runtime_dir = Path("runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)

    output_yaml = runtime_dir / f"{data_yaml.stem}.kaggle.yaml"
    with output_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)

    print(f"Rewrote dataset YAML for Kaggle input paths: {output_yaml}")
    return output_yaml


def read_kaggle_secret(secret_name: str) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient
    except ModuleNotFoundError:
        return None

    try:
        return UserSecretsClient().get_secret(secret_name)
    except Exception:
        return None


def configure_wandb(args: argparse.Namespace, run_name: str) -> None:
    if not args.wandb_project:
        return

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ["WANDB_NAME"] = run_name

    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    if args.wandb_mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
        return

    try:
        import wandb
    except ModuleNotFoundError:
        print("W&B requested but wandb is not installed. Run `pip install wandb` first.")
        return

    key = os.environ.get(args.wandb_key_secret) or read_kaggle_secret(args.wandb_key_secret)
    if key:
        wandb.login(key=key)
    else:
        print(
            f"W&B API key not found in env/Kaggle secret '{args.wandb_key_secret}'. "
            "Training will start, but W&B may ask for login or run offline."
        )

    try:
        from ultralytics import settings

        settings.update({"wandb": True})
    except Exception as exc:
        print(f"Could not update Ultralytics W&B setting automatically: {exc}")


def resolve_device(args: argparse.Namespace, model_name: str) -> str | None:
    if model_name == "yolov8s" and args.device_yolov8s:
        return args.device_yolov8s
    return args.device


def main() -> None:
    args = parse_args()
    data = Path(args.data) if args.data else find_visdrone_yaml(args.input_root)
    data = prepare_kaggle_dataset_yaml(data)
    print(f"VisDrone YAML: {data}")

    for model_name in args.models:
        run_name = model_name if len(args.models) == 1 else f"{model_name}-research"
        configure_wandb(args, run_name=run_name)
        print(f"\n=== VisDrone Kaggle experiment: {model_name} ===")

        train_args = argparse.Namespace(
            config=VISDRONE_CONFIGS[model_name],
            data=str(data),
            project=args.project,
            name=run_name,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=resolve_device(args, model_name),
            workers=args.workers,
            dry_run=args.dry_run,
        )
        train_from_config(train_args)


if __name__ == "__main__":
    main()
