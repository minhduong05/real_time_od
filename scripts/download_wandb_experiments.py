"""Download W&B experiment data for offline analysis.

Examples:
    python scripts/download_wandb_experiments.py
    python scripts/download_wandb_experiments.py --project test_log --out experiments/wandb_test_export
    python scripts/download_wandb_experiments.py --include-large-files
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any


DEFAULT_ENTITY = "minhmit146-hanoi-university-of-science-and-technology"
DEFAULT_PROJECTS = ["-kaggle-working-experiments-visdrone", "test_log"]
DEFAULT_OUT = Path("experiments/wandb_visdrone_export")

LARGE_SUFFIXES = {
    ".pt",
    ".pth",
    ".onnx",
    ".engine",
    ".torchscript",
    ".zip",
    ".rar",
    ".7z",
}


def json_default(value: Any) -> str:
    return str(value)


def safe_name(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in "-_.":
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("._") or "unnamed"


def should_download_file(name: str, include_large_files: bool) -> bool:
    if include_large_files:
        return True

    suffix = Path(name).suffix.lower()
    if suffix in LARGE_SUFFIXES:
        return False

    return (
        name.startswith("media/")
        or name.startswith("files/")
        or name.endswith(".json")
        or name.endswith(".csv")
        or name.endswith(".yaml")
        or name.endswith(".yml")
        or name.endswith(".txt")
    )


def should_export_run(run: Any, include_save_version_runs: bool) -> bool:
    if include_save_version_runs:
        return True

    return "save-version" not in run.name


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def export_history(run: Any, path: Path) -> None:
    rows = list(run.scan_history())
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_run(project: str, run: Any, root: Path, include_large_files: bool) -> dict[str, Any]:
    run_dir = root / safe_name(project) / f"{safe_name(run.name)}_{run.id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = dict(run.summary)
    write_json(run_dir / "config.json", dict(run.config))
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "metadata.json",
        {
            "entity": run.entity,
            "project": project,
            "name": run.name,
            "id": run.id,
            "path": run.path,
            "state": run.state,
            "group": run.group,
            "job_type": run.job_type,
            "created_at": getattr(run, "created_at", None),
            "updated_at": getattr(run, "updated_at", None),
            "url": getattr(run, "url", None),
            "tags": getattr(run, "tags", []),
        },
    )
    export_history(run, run_dir / "history.csv")

    downloaded_files = []
    skipped_files = []
    for remote_file in run.files():
        if should_download_file(remote_file.name, include_large_files):
            remote_file.download(root=str(run_dir), replace=True)
            downloaded_files.append(remote_file.name)
        else:
            skipped_files.append(remote_file.name)

    write_json(
        run_dir / "download_manifest.json",
        {
            "downloaded_files": downloaded_files,
            "skipped_files": skipped_files,
            "include_large_files": include_large_files,
        },
    )

    return {
        "project": project,
        "run_name": run.name,
        "run_id": run.id,
        "state": run.state,
        "created_at": getattr(run, "created_at", None),
        "group": run.group,
        "url": getattr(run, "url", None),
        "metric_mAP50": summary.get("metrics/mAP50(B)", summary.get("test/mAP50")),
        "metric_mAP50_95": summary.get("metrics/mAP50-95(B)", summary.get("test/mAP50-95")),
        "downloaded_file_count": len(downloaded_files),
        "skipped_file_count": len(skipped_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument(
        "--project",
        action="append",
        default=None,
        help="W&B project name. Repeat this option to export multiple projects.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument(
        "--include-large-files",
        action="store_true",
        help="Also download large artifacts such as checkpoints and archives.",
    )
    parser.add_argument(
        "--include-save-version-runs",
        action="store_true",
        help="Also export runs whose names contain 'save-version'.",
    )
    args = parser.parse_args()

    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit(
            "Missing WANDB_API_KEY. Set it first, for example: "
            "$env:WANDB_API_KEY='your_key_here'"
        )

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("Missing dependency. Install it with: pip install wandb") from exc

    projects = args.project or DEFAULT_PROJECTS
    args.out.mkdir(parents=True, exist_ok=True)

    wandb.login(key=os.environ["WANDB_API_KEY"])
    api = wandb.Api()
    overview = []

    for project in projects:
        print(f"Exporting {args.entity}/{project}")
        for run in api.runs(f"{args.entity}/{project}"):
            if not should_export_run(run, args.include_save_version_runs):
                print(f"  skipped {run.name} ({run.id})")
                continue

            row = export_run(project, run, args.out, args.include_large_files)
            overview.append(row)
            print(f"  exported {run.name} ({run.id})")

    write_json(args.out / "runs_overview.json", overview)
    with (args.out / "runs_overview.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in overview for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(overview)

    if not args.no_zip:
        archive = shutil.make_archive(str(args.out), "zip", root_dir=args.out)
        print(f"Created archive: {archive}")

    print(f"Exported {len(overview)} runs to {args.out}")


if __name__ == "__main__":
    main()
