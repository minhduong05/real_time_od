"""Prepare Intersection-Flow-5K for YOLO training."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401

from src.utils.paths import PROJECT_ROOT


KAGGLE_DATASET = "starsw/intersection-flow-5k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/Intersection-Flow-5K")
    parser.add_argument("--out-dir", default="data/processed/intersection_flow_5k_yolo")
    parser.add_argument("--source", default=None, help="Existing dataset folder or zip file.")
    parser.add_argument("--kaggle", action="store_true", help="Download with Kaggle CLI.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def download_with_kaggle(raw_dir: Path) -> None:
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI is not installed. Run `pip install kaggle`, then place "
            "`kaggle.json` in `%USERPROFILE%\\.kaggle\\kaggle.json`."
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(raw_dir),
            "--unzip",
        ],
        check=True,
    )


def extract_zip(zip_path: Path, raw_dir: Path, force: bool = False) -> Path:
    extract_dir = raw_dir / "extracted"
    if extract_dir.exists() and force:
        shutil.rmtree(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(extract_dir)
    return extract_dir


def find_dataset_root(source: Path) -> Path:
    if (source / "images").exists() and (source / "labels").exists():
        return source

    candidates = [
        path
        for path in source.rglob("*")
        if path.is_dir() and (path / "images").exists() and (path / "labels").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find Intersection-Flow-5K images/labels folders under {source}"
        )
    return candidates[0]


def copy_yolo_dataset(source_root: Path, out_dir: Path, force: bool = False) -> None:
    if out_dir.exists() and force:
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    for folder in ["images", "labels", "annotations"]:
        source = source_root / folder
        if source.exists():
            shutil.copytree(source, out_dir / folder, dirs_exist_ok=True)

    for filename in ["classes.txt", "intersection.yaml", "test_coco.json"]:
        source = source_root / filename
        if source.exists():
            shutil.copy2(source, out_dir / filename)


def main() -> None:
    args = parse_args()
    raw_dir = PROJECT_ROOT / args.raw_dir
    out_dir = PROJECT_ROOT / args.out_dir

    if args.kaggle:
        download_with_kaggle(raw_dir)

    source = Path(args.source) if args.source else raw_dir
    if not source.is_absolute():
        source = PROJECT_ROOT / source

    if source.is_file() and source.suffix.lower() == ".zip":
        source = extract_zip(source, raw_dir, force=args.force)

    dataset_root = find_dataset_root(source)
    copy_yolo_dataset(dataset_root, out_dir, force=args.force)

    print(f"Intersection-Flow-5K raw source: {dataset_root}")
    print(f"Intersection-Flow-5K YOLO dataset: {out_dir}")


if __name__ == "__main__":
    main()
