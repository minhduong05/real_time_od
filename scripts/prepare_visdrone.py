"""Download raw VisDrone data and convert it to YOLO format."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import _bootstrap  # noqa: F401
from PIL import Image
from tqdm import tqdm

from src.utils.paths import PROJECT_ROOT


ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
ARCHIVES = {
    "VisDrone2019-DET-train": "train",
    "VisDrone2019-DET-val": "val",
    "VisDrone2019-DET-test-dev": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/VisDrone")
    parser.add_argument("--out-dir", default="data/processed/visdrone_yolo")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-extracted", action="store_true")
    return parser.parse_args()


def download_archive(url: str, target: Path) -> None:
    if target.exists():
        print(f"exists: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"download: {url}")
    urlretrieve(url, target)
    print(f"saved: {target}")


def extract_archive(archive: Path, extract_dir: Path, force: bool = False) -> None:
    expected_dir = extract_dir / archive.stem
    if expected_dir.exists() and not force:
        print(f"exists: {expected_dir}")
        return

    if expected_dir.exists() and force:
        shutil.rmtree(expected_dir)

    print(f"extract: {archive}")
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(extract_dir)


def convert_split(source_dir: Path, out_dir: Path, split: str) -> None:
    images_dir = out_dir / "images" / split
    labels_dir = out_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    source_images = source_dir / "images"
    source_labels = source_dir / "annotations"

    for image_path in tqdm(list(source_images.glob("*.jpg")), desc=f"Copying {split} images"):
        target = images_dir / image_path.name
        if not target.exists():
            shutil.copy2(image_path, target)

    for label_path in tqdm(list(source_labels.glob("*.txt")), desc=f"Converting {split} labels"):
        image_path = images_dir / label_path.with_suffix(".jpg").name
        if not image_path.exists():
            continue

        width, height = Image.open(image_path).size
        dw, dh = 1.0 / width, 1.0 / height
        lines = []

        for row in label_path.read_text(encoding="utf-8").strip().splitlines():
            values = row.split(",")
            if len(values) < 6 or values[4] != "0":
                continue

            x, y, box_w, box_h = map(int, values[:4])
            cls = int(values[5]) - 1
            x_center = (x + box_w / 2) * dw
            y_center = (y + box_h / 2) * dh
            w_norm = box_w * dw
            h_norm = box_h * dh
            lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        (labels_dir / label_path.name).write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_dir = PROJECT_ROOT / args.raw_dir
    out_dir = PROJECT_ROOT / args.out_dir
    extract_dir = raw_dir / "extracted"

    raw_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for folder_name, split in ARCHIVES.items():
        archive = raw_dir / f"{folder_name}.zip"
        if not args.skip_download:
            download_archive(f"{ASSETS_URL}/{archive.name}", archive)
        extract_archive(archive, extract_dir, force=args.force)
        convert_split(extract_dir / folder_name, out_dir, split)

    if not args.keep_extracted:
        shutil.rmtree(extract_dir, ignore_errors=True)

    print(f"VisDrone raw archives: {raw_dir}")
    print(f"VisDrone YOLO dataset: {out_dir}")


if __name__ == "__main__":
    main()
