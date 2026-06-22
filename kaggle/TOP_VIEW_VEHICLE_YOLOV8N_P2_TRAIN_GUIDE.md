# Top-View Vehicle Detection: fine-tune YOLOv8n-P2 trên Kaggle

Guide này tạo notebook Kaggle mới để fine-tune kiến trúc **YOLOv8n-P2** trên dataset:

```text
Top-View Vehicle Detection Image Dataset
Kaggle slug: top-view-vehicle-detection-image-dataset
Dataset root: /kaggle/input/top-view-vehicle-detection-image-dataset/Vehicle_Detection_Image_Dataset
Class: Vehicle
Train: 536 images
Valid: 90 images
Image size: 640x640
Format: YOLO
```

Mục tiêu mới của project là highway/top-view traffic videos, nên dataset này là workflow chính hiện tại.

## A. Chuẩn bị Kaggle Notebook

1. Tạo notebook Kaggle mới.
2. Settings -> Accelerator: chọn **GPU T4** hoặc **GPU T4 x2**.
3. Bật Internet.
4. Add Input dataset:

```text
top-view-vehicle-detection-image-dataset
```

5. Add-ons -> Secrets: thêm và bật:

```text
WANDB_API_KEY
```

6. Tạo W&B project:

```text
top-view-vehicle-yolov8n-p2
```

## Cell 1 -- Markdown

```markdown
# Fine-tuning YOLOv8n-P2 on Top-View Vehicle Detection

- Dataset: Top-View Vehicle Detection Image Dataset
- Class: Vehicle
- Model: YOLOv8n-P2, transfer learning from yolov8n.pt
- Image size: 640
- Tracker: Weights & Biases
- Goal: highway/top-view vehicle detection for realtime traffic density estimation
```

## Cell 2 -- Clone hoặc mở repository

Nếu notebook clone từ GitHub:

```python
!git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY>.git /kaggle/working/real_time_od
%cd /kaggle/working/real_time_od

!git status --short
!find configs kaggle -maxdepth 3 -type f | sort
```

Nếu bạn upload repo bằng Kaggle Dataset, chỉ cần `%cd` vào thư mục repo tương ứng.

Đúng khi có:

```text
configs/models/yolov8n-p2.yaml
kaggle/train.py
kaggle/check_model.py
```

## Cell 3 -- Cài package và kiểm tra GPU

Không dùng nâng cấp toàn bộ môi trường. Kaggle đã có PyTorch/CUDA.

```python
%pip install -q wandb
%pip install -q ultralytics --no-deps

import os
from pathlib import Path

import torch
import ultralytics
import wandb

print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

assert torch.cuda.is_available(), "Hãy bật GPU trong Kaggle Settings."

for device_id in range(torch.cuda.device_count()):
    print(f"GPU {device_id}:", torch.cuda.get_device_name(device_id))
    print(torch.zeros(1, device=f"cuda:{device_id}"))
```

Đúng khi CUDA available là `True` và có ít nhất một GPU T4.

## Cell 4 -- W&B login

```python
import os
from kaggle_secrets import UserSecretsClient

WANDB_ENTITY = "minhmit146-hanoi-university-of-science-and-technology"
WANDB_PROJECT = "top-view-vehicle-yolov8n-p2"
WANDB_RUN_NAME = "yolov8n-p2_top-view-vehicle"
WANDB_GROUP = "top-view-vehicle-primary-run"
WANDB_TAGS = "top-view-vehicle,yolov8n-p2,vehicle-detection,highway,primary-run"

os.environ["WANDB_ENTITY"] = WANDB_ENTITY
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_NAME"] = WANDB_RUN_NAME
os.environ["WANDB_TAGS"] = WANDB_TAGS

wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
!yolo settings wandb=True
```

Đúng khi login thành công. Khi train bắt đầu, W&B sẽ tạo run trong project `top-view-vehicle-yolov8n-p2`.

## Cell 5 -- Tìm và kiểm tra dataset

Cell này không phụ thuộc quá chặt vào folder slug ngoài cùng; nó tìm file `data.yaml`.

```python
from pathlib import Path
from PIL import Image
import yaml

INPUT_ROOT = Path("/kaggle/input")

yaml_candidates = [
    path for path in INPUT_ROOT.rglob("data.yaml")
    if "Vehicle_Detection_Image_Dataset" in str(path)
]
assert len(yaml_candidates) == 1, f"Không tìm thấy đúng data.yaml: {yaml_candidates}"

SOURCE_YAML = yaml_candidates[0]
DATASET_ROOT = SOURCE_YAML.parent

print("Dataset root:", DATASET_ROOT)
print("Source YAML:", SOURCE_YAML)
print(SOURCE_YAML.read_text())

with SOURCE_YAML.open("r", encoding="utf-8") as file:
    source_data = yaml.safe_load(file)

names = source_data.get("names")
if isinstance(names, dict):
    class_names = [names[index] for index in sorted(names)]
else:
    class_names = list(names)

print("Classes:", class_names)
assert class_names == ["Vehicle"], class_names
assert int(source_data.get("nc", len(class_names))) == 1

TRAIN_IMAGES_DIR = DATASET_ROOT / "train" / "images"
TRAIN_LABELS_DIR = DATASET_ROOT / "train" / "labels"
VALID_IMAGES_DIR = DATASET_ROOT / "valid" / "images"
VALID_LABELS_DIR = DATASET_ROOT / "valid" / "labels"

for path in [TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, VALID_IMAGES_DIR, VALID_LABELS_DIR]:
    assert path.is_dir(), f"Thiếu folder: {path}"

extensions = {".jpg", ".jpeg", ".png", ".bmp"}
train_images = sorted(path for path in TRAIN_IMAGES_DIR.iterdir() if path.suffix.lower() in extensions)
valid_images = sorted(path for path in VALID_IMAGES_DIR.iterdir() if path.suffix.lower() in extensions)
train_labels = sorted(TRAIN_LABELS_DIR.glob("*.txt"))
valid_labels = sorted(VALID_LABELS_DIR.glob("*.txt"))

print("Train images:", len(train_images))
print("Valid images:", len(valid_images))
print("Train labels:", len(train_labels))
print("Valid labels:", len(valid_labels))

assert len(train_images) == 536
assert len(valid_images) == 90

train_sizes = {Image.open(path).size for path in train_images[:50]}
valid_sizes = {Image.open(path).size for path in valid_images[:50]}
print("Train image sizes sample:", train_sizes)
print("Valid image sizes sample:", valid_sizes)
assert train_sizes == {(640, 640)}
assert valid_sizes == {(640, 640)}
```

Ghi chú: số label có thể nhỏ hơn số ảnh nếu dataset có ảnh không có object. Điều đó hợp lệ trong YOLO.

## Cell 6 -- Kiểm tra label YOLO format

```python
from collections import Counter

def valid_yolo_line(line: str):
    fields = line.split()
    if len(fields) != 5:
        return None
    try:
        class_id = int(fields[0])
        coordinates = [float(value) for value in fields[1:]]
    except ValueError:
        return None
    if class_id != 0:
        return None
    if not all(0.0 <= value <= 1.0 for value in coordinates):
        return None
    return class_id

invalid_rows = []
class_counts = Counter()
valid_object_count = 0

for split, labels_dir in [("train", TRAIN_LABELS_DIR), ("valid", VALID_LABELS_DIR)]:
    for label_path in sorted(labels_dir.glob("*.txt")):
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            class_id = valid_yolo_line(line)
            if class_id is None:
                invalid_rows.append((split, label_path.name, line_number, line))
                continue
            class_counts["Vehicle"] += 1
            valid_object_count += 1

print("Valid objects:", valid_object_count)
print("Class counts:", class_counts)
print("Invalid rows:", len(invalid_rows))
assert valid_object_count > 0
assert not invalid_rows, invalid_rows[:10]
```

Đúng khi không có invalid rows và object count > 0.

## Cell 7 -- Tạo YAML an toàn cho Kaggle

Không dùng trực tiếp `data.yaml` từ input nếu path bên trong là relative khó đoán. Tạo YAML mới với absolute path.

```python
TOP_VIEW_YAML = Path("/kaggle/working/top_view_vehicle.yaml")
top_view_data = {
    "path": str(DATASET_ROOT),
    "train": "train/images",
    "val": "valid/images",
    "nc": 1,
    "names": {0: "Vehicle"},
}

with TOP_VIEW_YAML.open("w", encoding="utf-8") as file:
    yaml.safe_dump(top_view_data, file, sort_keys=False)

print(TOP_VIEW_YAML.read_text())
```

## Cell 8 -- Kiểm tra YOLOv8n-P2 transfer

```python
!python kaggle/check_model.py --model yolov8n-p2
```

Đúng khi log có:

```text
model: yolov8n-p2
transfer_from: yolov8n.pt
```

Ultralytics sẽ tự override `nc` từ model YAML placeholder sang `nc=1` theo dataset YAML khi train.

## Cell 9 -- Dry-run trước khi train

```python
!python kaggle/train.py \
  --dataset-name Top-View-Vehicle-Detection \
  --data /kaggle/working/top_view_vehicle.yaml \
  --model yolov8n-p2 \
  --epochs 100 \
  --imgsz 640 \
  --batch 32 \
  --workers 4 \
  --device 0 \
  --optimizer AdamW \
  --lr0 0.0001 \
  --lrf 0.1 \
  --weight-decay 0.0005 \
  --warmup-epochs 3 \
  --patience 50 \
  --mosaic 1.0 \
  --close-mosaic 10 \
  --fliplr 0.5 \
  --hsv-h 0.015 \
  --hsv-s 0.7 \
  --hsv-v 0.4 \
  --mixup 0.0 \
  --seed 0 \
  --save-period 25 \
  --project /kaggle/working/experiments/top-view-vehicle \
  --name yolov8n-p2 \
  --export-dir /kaggle/working/export \
  --dry-run
```

Đúng khi output có:

```text
dataset: Top-View-Vehicle-Detection
dataset nc: 1
dataset names: ['Vehicle']
output: /kaggle/working/experiments/top-view-vehicle/yolov8n-p2
export: /kaggle/working/export/Top-View-Vehicle-Detection/yolov8n-p2
```

Nếu chọn GPU T4 x2, có thể đổi:

```text
--device 0,1
--batch 64
```

Nhưng primary run nên dùng `device 0`, `batch 32` cho đơn giản và dễ tái lập vì dataset nhỏ.

## Cell 10 -- Train YOLOv8n-P2

```python
!python kaggle/train.py \
  --dataset-name Top-View-Vehicle-Detection \
  --data /kaggle/working/top_view_vehicle.yaml \
  --model yolov8n-p2 \
  --epochs 100 \
  --imgsz 640 \
  --batch 32 \
  --workers 4 \
  --device 0 \
  --optimizer AdamW \
  --lr0 0.0001 \
  --lrf 0.1 \
  --weight-decay 0.0005 \
  --warmup-epochs 3 \
  --patience 50 \
  --mosaic 1.0 \
  --close-mosaic 10 \
  --fliplr 0.5 \
  --hsv-h 0.015 \
  --hsv-s 0.7 \
  --hsv-v 0.4 \
  --mixup 0.0 \
  --seed 0 \
  --save-period 25 \
  --project /kaggle/working/experiments/top-view-vehicle \
  --name yolov8n-p2 \
  --export-dir /kaggle/working/export
```

Kỳ vọng:

1. W&B tạo run `yolov8n-p2_top-view-vehicle`.
2. Log train/box_loss, train/cls_loss, train/dfl_loss giảm dần.
3. Validation mAP50 và mAP50-95 tăng dần.
4. Early stopping có thể dừng trước 100 epoch nếu không cải thiện.
5. Cuối train có `best.pt` và `last.pt`.

## Cell 11 -- Kiểm tra output sau train

```python
from pathlib import Path

EXPORT_DIR = Path("/kaggle/working/export/Top-View-Vehicle-Detection/yolov8n-p2")
RUN_DIR = Path("/kaggle/working/experiments/top-view-vehicle/yolov8n-p2")

required_files = [
    EXPORT_DIR / "best.pt",
    EXPORT_DIR / "last.pt",
    EXPORT_DIR / "run_info.yaml",
    RUN_DIR / "results.csv",
    RUN_DIR / "results.png",
    RUN_DIR / "confusion_matrix.png",
    RUN_DIR / "confusion_matrix_normalized.png",
]

curve_candidates = [
    RUN_DIR / "BoxPR_curve.png",
    RUN_DIR / "PR_curve.png",
    RUN_DIR / "BoxP_curve.png",
    RUN_DIR / "P_curve.png",
    RUN_DIR / "BoxR_curve.png",
    RUN_DIR / "R_curve.png",
]

for path in required_files:
    print(("OK  " if path.exists() else "MISS"), path)

for path in curve_candidates:
    if path.exists():
        print("OK  ", path)

assert (EXPORT_DIR / "best.pt").exists(), "Không có best.pt"
assert (RUN_DIR / "results.csv").exists(), "Không có results.csv"

print("Best checkpoint:", EXPORT_DIR / "best.pt")
print("W&B project:", f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")
```

## Cell 12 -- Đọc metric tốt nhất từ results.csv

```python
import pandas as pd

results_csv = RUN_DIR / "results.csv"
df = pd.read_csv(results_csv)
df.columns = df.columns.str.strip()

map_col = "metrics/mAP50-95(B)"
best_row = df.loc[df[map_col].idxmax()]

display(best_row.to_frame("best").T)

print("Best epoch:", int(best_row["epoch"]))
print("Best mAP50:", float(best_row["metrics/mAP50(B)"]))
print("Best mAP50-95:", float(best_row["metrics/mAP50-95(B)"]))
print("Precision:", float(best_row["metrics/precision(B)"]))
print("Recall:", float(best_row["metrics/recall(B)"]))
```

Kết quả tham khảo từ notebook gốc dùng YOLOv8n thường có mAP50 rất cao. Với YOLOv8n-P2, kỳ vọng hợp lý:

```text
mAP50 cao, ideally > 0.90
mAP50-95 tốt, ideally > 0.65
Precision/Recall cân bằng
```

Không dùng các ngưỡng này như cam kết tuyệt đối; hãy so với validation curve và sample predictions.

## Cell 13 -- Inference nhanh trên validation images

```python
import random
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

best_model = YOLO(str(EXPORT_DIR / "best.pt"))

random.seed(0)
sample_images = random.sample(valid_images, k=min(9, len(valid_images)))

fig, axes = plt.subplots(3, 3, figsize=(18, 18))
for ax, image_path in zip(axes.ravel(), sample_images):
    result = best_model.predict(source=str(image_path), imgsz=640, conf=0.35, verbose=False)[0]
    annotated = cv2.cvtColor(result.plot(line_width=1), cv2.COLOR_BGR2RGB)
    ax.imshow(annotated)
    ax.set_title(image_path.name, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.show()
```

Đúng khi xe trong ảnh top-view được detect rõ, ít false positive nền đường.

## Cell 14 -- Inference trên sample video của dataset

```python
import shutil

source_video = DATASET_ROOT / "sample_video.mp4"
working_video = Path("/kaggle/working/top_view_sample_video.mp4")

assert source_video.exists(), source_video
shutil.copy2(source_video, working_video)

video_results = best_model.predict(
    source=str(working_video),
    imgsz=640,
    conf=0.35,
    save=True,
    project="/kaggle/working/top_view_video_predictions",
    name="yolov8n-p2",
    exist_ok=True,
)
```

Nếu Ultralytics xuất `.avi`, có thể convert sang mp4:

```python
!find /kaggle/working/top_view_video_predictions -type f | sort
!ffmpeg -y -loglevel warning -i /kaggle/working/top_view_video_predictions/yolov8n-p2/top_view_sample_video.avi /kaggle/working/top_view_sample_video_pred.mp4
```

## Cell 15 -- Tạo ZIP export để tải về local

```python
import shutil

REPORT_DIR = Path("/kaggle/working/top_view_vehicle_train_export")
shutil.rmtree(REPORT_DIR, ignore_errors=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

for source in [
    EXPORT_DIR / "best.pt",
    EXPORT_DIR / "last.pt",
    EXPORT_DIR / "run_info.yaml",
    RUN_DIR / "results.csv",
    RUN_DIR / "results.png",
    RUN_DIR / "confusion_matrix.png",
    RUN_DIR / "confusion_matrix_normalized.png",
]:
    if source.exists():
        shutil.copy2(source, REPORT_DIR / source.name)

for pattern in ("*curve*.png", "labels.jpg", "train_batch*.jpg", "val_batch*.jpg"):
    for source in RUN_DIR.glob(pattern):
        shutil.copy2(source, REPORT_DIR / source.name)

video_pred_dir = Path("/kaggle/working/top_view_video_predictions")
if video_pred_dir.exists():
    for source in video_pred_dir.rglob("*"):
        if source.is_file():
            target = REPORT_DIR / "video_predictions" / source.relative_to(video_pred_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

zip_path = shutil.make_archive(
    "/kaggle/working/Top-View-Vehicle-Detection_yolov8n-p2_export",
    "zip",
    root_dir=REPORT_DIR,
)

print("ZIP:", zip_path)
print("W&B:", f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")
```

Tải file:

```text
/kaggle/working/Top-View-Vehicle-Detection_yolov8n-p2_export.zip
```

Sau khi tải về local, đặt:

```text
models/Top-View-Vehicle-Detection-Yolov8n-P2/best.pt
models/Top-View-Vehicle-Detection-Yolov8n-P2/last.pt
models/Top-View-Vehicle-Detection-Yolov8n-P2/run_info.yaml
```

## Checklist W&B sau train

Trong W&B project `top-view-vehicle-yolov8n-p2`, kiểm tra:

1. Config có dataset path, `model=yolov8n-p2`, `imgsz=640`, `batch=32`, `epochs=100`, `lr0=0.0001`, `patience=50`.
2. History có train loss, validation loss, precision, recall, mAP50, mAP50-95.
3. Media có confusion matrix, PR/P/R/F1 curves, train/val sample images.
4. Files/artifacts có best checkpoint hoặc ít nhất Kaggle ZIP chứa `best.pt`.
5. Run name đúng: `yolov8n-p2_top-view-vehicle`.

## Ghi chú báo cáo

Dataset này chỉ có một class `Vehicle`, nên frontend tập trung vào phát hiện xe tổng quát thay vì phân biệt từng loại phương tiện. Đổi lại, model phù hợp hơn với top-view/highway videos và bài toán traffic density.
