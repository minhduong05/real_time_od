"""Generate composite test_log comparison panels from exported W&B run media."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path("experiments/wandb_visdrone_export/test_log")
OUT_ROOT = Path("experiments/comparison_panels/test_log")

RUN_DIRS = {
    "yolov8s": ROOT / "yolov8s_pugbqriz",
    "yolov8n-p2": ROOT / "yolov8n-p2_xscvxmtm",
    "yolov8n": ROOT / "yolov8n_if29oy0m",
}

MODEL_COLORS = {
    "yolov8s": (129, 88, 214),
    "yolov8n-p2": (71, 157, 91),
    "yolov8n": (244, 63, 77),
}

PLOT_NAMES = [
    "confusion_matrix_normalized",
    "confusion_matrix",
    "BoxR_curve",
    "BoxP_curve",
    "BoxPR_curve",
    "BoxF1_curve",
]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.ImageFont) -> int:
    left, _, right, _ = draw.textbbox((0, 0), text, font=text_font)
    return right - left


def paste_contained(canvas: Image.Image, src_path: Path, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img.thumbnail((right - left, bottom - top), Image.Resampling.LANCZOS)
        x = left + (right - left - img.width) // 2
        y = top + (bottom - top - img.height) // 2
        canvas.paste(img, (x, y))


def first_plot_path(run_dir: Path, plot_name: str) -> Path:
    matches = sorted((run_dir / "media/images/plots").glob(f"{plot_name}_*.png"))
    if not matches:
        raise FileNotFoundError(f"Missing plot {plot_name} in {run_dir}")
    return matches[0]


def make_three_model_panel(title: str, image_paths: dict[str, Path], out_path: Path) -> None:
    width, height = 2400, 1050
    margin, gap = 90, 50
    col_w = (width - margin * 2 - gap * 2) // 3

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(44, bold=True)
    label_font = load_font(34)

    draw.text(((width - text_width(draw, title, title_font)) // 2, 55), title, fill=(20, 31, 45), font=title_font)
    for index, model in enumerate(["yolov8s", "yolov8n-p2", "yolov8n"]):
        left = margin + index * (col_w + gap)
        right = left + col_w
        draw.text(
            ((left + right - text_width(draw, model, label_font)) // 2, 155),
            model,
            fill=MODEL_COLORS[model],
            font=label_font,
        )
        paste_contained(canvas, image_paths[model], (left, 210, right, height - 70))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def fingerprint(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("L").resize((64, 64), Image.Resampling.BILINEAR)


def distance(left: Image.Image, right: Image.Image) -> float:
    diff = ImageChops.difference(left, right)
    hist = diff.histogram()
    return sum(value * ((idx % 256) ** 2) for idx, value in enumerate(hist)) / (64 * 64)


def match_samples() -> list[dict[str, Path]]:
    paths = {
        model: sorted((run_dir / "media/images/test").glob("sample_predictions_*.png"))
        for model, run_dir in RUN_DIRS.items()
    }
    fps = {model: [(path, fingerprint(path)) for path in model_paths] for model, model_paths in paths.items()}
    rows: list[dict[str, Path]] = []
    used = {model: set() for model in paths}

    for base_path, base_fp in fps["yolov8s"]:
        row = {"yolov8s": base_path}
        used["yolov8s"].add(base_path)
        for model in ["yolov8n-p2", "yolov8n"]:
            candidates = [(path, fp) for path, fp in fps[model] if path not in used[model]]
            best_path, _ = min(candidates, key=lambda item: distance(base_fp, item[1]))
            row[model] = best_path
            used[model].add(best_path)
        rows.append(row)

    return rows


def make_overview(paths: list[Path], out_path: Path) -> None:
    card_w, card_h = 1180, 760
    gap, margin = 55, 55
    canvas = Image.new("RGB", (margin * 2 + card_w * 3 + gap * 2, margin * 2 + card_h * 2 + gap), "white")
    for index, path in enumerate(paths):
        row, col = divmod(index, 3)
        left = margin + col * (card_w + gap)
        top = margin + row * (card_h + gap)
        paste_contained(canvas, path, (left, top, left + card_w, top + card_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    generated_plots = []
    for plot_name in PLOT_NAMES:
        paths = {model: first_plot_path(run_dir, plot_name) for model, run_dir in RUN_DIRS.items()}
        out_path = OUT_ROOT / "plots" / f"W&B_{plot_name}_comparison.png"
        make_three_model_panel(f"plots/{plot_name}", paths, out_path)
        generated_plots.append(out_path)

    make_overview(generated_plots, OUT_ROOT / "plots" / "W&B_plots_overview.png")

    for index, paths in enumerate(match_samples()):
        out_path = OUT_ROOT / "test" / f"W&B_sample_predictions_index_{index:02d}.png"
        make_three_model_panel(f"test/sample_predictions index {index}", paths, out_path)

    print(f"Generated comparison panels in {OUT_ROOT}")


if __name__ == "__main__":
    main()
