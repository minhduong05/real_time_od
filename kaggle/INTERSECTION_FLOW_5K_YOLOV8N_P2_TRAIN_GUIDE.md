# Intersection-Flow-5K: fine-tune YOLOv8n-P2 trên Kaggle

Notebook này fine-tune kiến trúc **YOLOv8n-P2** trên Intersection-Flow-5K, log train/validation lên W&B project mới và export checkpoint tốt nhất.

Mô hình này sẽ là detector của phần ứng dụng traffic monitoring:

~~~text
Camera giao lộ
  -> YOLOv8n-P2 detection
  -> tracking
  -> đếm theo class/làn
  -> mật độ và cảnh báo ùn tắc
~~~

Dataset có 8 class, theo đúng thứ tự:

~~~text
vehicle
bus
bicycle
pedestrian
engine
truck
tricycle
obstacle
~~~

Cấu trúc dataset cần có:

~~~text
Intersection-Flow-5K/
  images/train, images/val, images/test
  labels/train, labels/val, labels/test
  intersection.yaml
  classes.txt
~~~

## 0. Thiết lập giao diện Kaggle

Tạo notebook mới, sau đó:

1. Trong Settings -> Accelerator, chọn **GPU T4 x2**. Code chỉ dùng GPU đầu tiên, device 0.
2. Bật Internet để clone repository, tải pretrained yolov8n.pt và gửi log W&B.
3. Trong Add-ons -> Secrets, thêm WANDB_API_KEY rồi bật quyền truy cập secret cho notebook.
4. Bấm Add Input và thêm Kaggle Dataset Intersection-Flow-5K.
5. Tạo W&B project mới, ví dụ intersection-flow-5k-yolov8n-p2. W&B cũng có thể tự tạo project khi train bắt đầu.

Không dùng GPU P100: PyTorch mới trong Kaggle có thể không hỗ trợ kiến trúc P100.

## Cell 1 -- Markdown

~~~markdown
# Fine-tuning YOLOv8n-P2 on Intersection-Flow-5K

- Task: traffic-object detection
- Model: YOLOv8n-P2, transfer learning from yolov8n.pt
- Dataset: Intersection-Flow-5K, 8 classes
- Validation: chạy sau từng epoch
- Tracking: Weights & Biases
~~~

## Cell 2 -- Clone repository

Thay URL GitHub bằng repository của bạn. Nếu repo đã được thêm bằng Kaggle Dataset thì không clone; chỉ cd tới thư mục repo đó.

~~~python
!git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY>.git /kaggle/working/real_time_od
%cd /kaggle/working/real_time_od

!git status --short
!find configs kaggle -maxdepth 3 -type f | sort
~~~

**Đúng khi:** có các file configs/models/yolov8n-p2.yaml, kaggle/train.py và kaggle/check_model.py.

## Cell 3 -- Cài dependency an toàn, kiểm tra GPU

Không dùng pip với cờ nâng cấp toàn bộ package. Việc đó có thể nâng PyTorch/CUDA và làm GPU Kaggle mất tương thích.

~~~python
%pip install -q wandb
%pip install -q -r kaggle/requirements.txt

import torch
import ultralytics
import wandb

print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

assert torch.cuda.is_available(), "Hãy bật GPU T4 x2 trong Kaggle Settings."
assert torch.cuda.get_device_capability(0) >= (7, 0), (
    "GPU không tương thích PyTorch hiện tại. Chọn GPU T4 x2 thay cho P100."
)

test_tensor = torch.zeros(1, device="cuda")
print("CUDA tensor test:", test_tensor)
~~~

**Đúng khi:** CUDA available là True, GPU là Tesla T4 và tensor test chạy thành công.

## Cell 4 -- Đăng nhập và cấu hình W&B

Đổi W&B project nếu bạn muốn tên khác. Entity là workspace hiện tại của bạn.

~~~python
import os
from kaggle_secrets import UserSecretsClient

WANDB_ENTITY = "minhmit146-hanoi-university-of-science-and-technology"
WANDB_PROJECT = "intersection-flow-5k-yolov8n-p2"
WANDB_RUN_NAME = "yolov8n-p2_intersection-flow-5k"

os.environ["WANDB_ENTITY"] = WANDB_ENTITY
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_NAME"] = WANDB_RUN_NAME

wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
!yolo settings wandb=True
~~~

**Đúng khi:** W&B login thành công. Khi train bắt đầu, terminal sẽ in URL project và run.

## Cell 5 -- Tìm, kiểm tra dataset và class

Cell tự tìm file intersection.yaml nên không phụ thuộc vào tên Kaggle slug. Nó kiểm tra split và thứ tự class trước khi train.

~~~python
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
yaml_candidates = list(INPUT_ROOT.rglob("intersection.yaml"))
assert len(yaml_candidates) == 1, (
    f"Phải có đúng một intersection.yaml, tìm thấy: {yaml_candidates}"
)

SOURCE_YAML = yaml_candidates[0]
DATASET_ROOT = SOURCE_YAML.parent
CLASSES_PATH = DATASET_ROOT / "classes.txt"

assert CLASSES_PATH.is_file(), f"Thiếu classes.txt: {CLASSES_PATH}"
CLASS_NAMES = [
    line.strip()
    for line in CLASSES_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

EXPECTED_CLASSES = [
    "vehicle", "bus", "bicycle", "pedestrian",
    "engine", "truck", "tricycle", "obstacle",
]

print("Dataset root:", DATASET_ROOT)
print("Source YAML:", SOURCE_YAML)
print("Classes:", CLASS_NAMES)
assert CLASS_NAMES == EXPECTED_CLASSES, (
    "Thứ tự/tên class khác dự kiến. Kiểm tra classes.txt trước khi train."
)

for split in ("train", "val", "test"):
    images_dir = DATASET_ROOT / "images" / split
    labels_dir = DATASET_ROOT / "labels" / split
    assert images_dir.is_dir(), f"Thiếu images/{split}"
    assert labels_dir.is_dir(), f"Thiếu labels/{split}"

    image_count = len([
        path for path in images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])
    label_count = len(list(labels_dir.glob("*.txt")))
    print(f"{split:5s}: {image_count} images | {label_count} label files")
    assert image_count > 0, f"Split {split} không có ảnh"
~~~

Trong ảnh cấu trúc dataset còn có thư mục annotations. Thư mục đó không dùng ở đây: Ultralytics train trực tiếp từ labels, vì labels đã là YOLO format.

## Cell 5b -- Xác thực toàn bộ labels là YOLO format

Cell này đọc toàn bộ file label trước khi train. Nó kiểm tra mỗi object có đúng năm giá trị: class_id, x_center, y_center, width, height; class_id thuộc 0 đến 7 và bốn tọa độ đã chuẩn hóa trong đoạn 0 đến 1.

~~~python
from collections import Counter

class_counts = Counter()
invalid_rows = []
total_objects = 0

for split in ("train", "val", "test"):
    labels_dir = DATASET_ROOT / "labels" / split
    for label_path in sorted(labels_dir.glob("*.txt")):
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue

            fields = line.split()
            if len(fields) != 5:
                invalid_rows.append((str(label_path), line_number, line))
                continue

            try:
                class_id = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
            except ValueError:
                invalid_rows.append((str(label_path), line_number, line))
                continue

            if class_id not in range(len(CLASS_NAMES)) or not all(
                0.0 <= value <= 1.0 for value in coordinates
            ):
                invalid_rows.append((str(label_path), line_number, line))
                continue

            class_counts[CLASS_NAMES[class_id]] += 1
            total_objects += 1

print("Tổng số object:", total_objects)
print("Số object theo class:")
for class_name in CLASS_NAMES:
    print(f"  {class_name:12s} {class_counts[class_name]}")

assert not invalid_rows, (
    f"Có {len(invalid_rows)} label row không đúng YOLO format. "
    f"Ví dụ: {invalid_rows[:5]}"
)
assert total_objects > 0, "Không đọc được object nào từ labels."
~~~

**Đúng khi:** không có AssertionError; mọi class đều được in số lượng object. Nếu có class có số lượng 0 thì không train ngay, cần kiểm tra dữ liệu hoặc classes.txt.

## Cell 6 -- Tạo YAML an toàn cho Kaggle

Không dùng trực tiếp intersection.yaml từ dataset, vì path có thể không đúng khi mount vào Kaggle. YAML dưới đây dùng path tuyệt đối và đúng thứ tự 8 class.

~~~python
import yaml

intersection_yaml = {
    "path": str(DATASET_ROOT),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {index: name for index, name in enumerate(CLASS_NAMES)},
}

DATA_YAML = Path("/kaggle/working/intersection_flow_5k.yaml")
with DATA_YAML.open("w", encoding="utf-8") as file:
    yaml.safe_dump(intersection_yaml, file, sort_keys=False, allow_unicode=True)

print(DATA_YAML.read_text())
assert len(intersection_yaml["names"]) == 8
~~~

## Cell 7 -- Kiểm tra YOLOv8n-P2 và pretrained transfer

Config YOLOv8n-P2 trong repo hiện có nc là 10 để dùng với VisDrone. Khi train, Ultralytics sẽ tự đổi thành nc là 8 theo data YAML này. Đó là hành vi đúng.

~~~python
!python kaggle/check_model.py --model yolov8n-p2
~~~

**Đúng khi:**

- log có model yolov8n-p2;
- log có transfer_from yolov8n.pt;
- pretrained weights được transfer vào kiến trúc P2;
- không có lỗi tải model.

## Cell 8 -- Dry-run kiểm tra argument

Dry-run chưa train model. Nó chỉ xác nhận dataset, model name, output path và export path.

~~~python
!python kaggle/train.py \
  --dataset-name Intersection-Flow-5K \
  --data /kaggle/working/intersection_flow_5k.yaml \
  --model yolov8n-p2 \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --workers 4 \
  --device 0 \
  --project /kaggle/working/experiments/intersection-flow-5k \
  --name yolov8n-p2 \
  --export-dir /kaggle/working/export \
  --dry-run
~~~

**Đúng khi:** output có:

~~~text
model: yolov8n-p2
dataset: Intersection-Flow-5K
data: /kaggle/working/intersection_flow_5k.yaml
output: /kaggle/working/experiments/intersection-flow-5k/yolov8n-p2
export: /kaggle/working/export/Intersection-Flow-5K/yolov8n-p2
~~~

Nếu sai path, dừng tại đây và sửa trước khi train.

## Cell 9 -- Fine-tune 100 epochs

Đây là cell train thật. Validation chạy sau mọi epoch, vì vậy W&B có train loss, val loss, Precision, Recall, mAP50 và mAP50-95 theo epoch.

~~~python
!python kaggle/train.py \
  --dataset-name Intersection-Flow-5K \
  --data /kaggle/working/intersection_flow_5k.yaml \
  --model yolov8n-p2 \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --workers 4 \
  --device 0 \
  --project /kaggle/working/experiments/intersection-flow-5k \
  --name yolov8n-p2 \
  --export-dir /kaggle/working/export
~~~

Nếu out-of-memory, đổi batch 16 thành batch 8 và chạy lại Cell 9 từ đầu. Không đổi cấu hình giữa chừng.

**Đúng khi trong log:**

1. Có dòng model YAML nc 10 bị ghi đè thành nc 8. Đây không phải lỗi.
2. Mỗi epoch có train loss và validation metric.
3. W&B in URL run của project intersection-flow-5k-yolov8n-p2.
4. Cuối train có checkpoint best.pt và last.pt được export.

## Cell 10 -- Kiểm tra checkpoint, kết quả và W&B

~~~python
EXPORT_DIR = Path("/kaggle/working/export/Intersection-Flow-5K/yolov8n-p2")
RUN_DIR = Path("/kaggle/working/experiments/intersection-flow-5k/yolov8n-p2")

required_files = [
    EXPORT_DIR / "best.pt",
    EXPORT_DIR / "last.pt",
    EXPORT_DIR / "run_info.yaml",
    RUN_DIR / "results.csv",
    RUN_DIR / "results.png",
    RUN_DIR / "PR_curve.png",
    RUN_DIR / "confusion_matrix.png",
    RUN_DIR / "confusion_matrix_normalized.png",
]

for path in required_files:
    print(("OK  " if path.exists() else "MISS"), path)

assert (EXPORT_DIR / "best.pt").exists(), "Không có best.pt"
assert (RUN_DIR / "results.csv").exists(), "Không có results.csv"

print("Best checkpoint:", EXPORT_DIR / "best.pt")
print("W&B project:", f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")
~~~

## Cell 11 -- Lưu CSV và zip checkpoint để tải về

~~~python
import shutil

REPORT_DIR = Path("/kaggle/working/intersection_flow_5k_train_export")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

shutil.copy2(RUN_DIR / "results.csv", REPORT_DIR / "results.csv")
shutil.copy2(RUN_DIR / "results.png", REPORT_DIR / "results.png")
shutil.copy2(EXPORT_DIR / "best.pt", REPORT_DIR / "best.pt")
shutil.copy2(EXPORT_DIR / "last.pt", REPORT_DIR / "last.pt")
shutil.copy2(EXPORT_DIR / "run_info.yaml", REPORT_DIR / "run_info.yaml")

for pattern in ("*curve*.png", "confusion_matrix*.png", "labels.jpg"):
    for source in RUN_DIR.glob(pattern):
        shutil.copy2(source, REPORT_DIR / source.name)

zip_path = shutil.make_archive(
    "/kaggle/working/Intersection-Flow-5K_yolov8n-p2_export",
    "zip",
    root_dir=REPORT_DIR,
)

print("File ZIP để tải trong Kaggle Output:", zip_path)
~~~

Sau Cell 11, bấm **Save Version**. Tải file:

~~~text
/kaggle/working/Intersection-Flow-5K_yolov8n-p2_export.zip
~~~

## Kiểm tra W&B sau train

Trong project intersection-flow-5k-yolov8n-p2, kiểm tra:

1. train box_loss, cls_loss và dfl_loss giảm dần.
2. val loss không tăng kéo dài ở cuối train.
3. Precision, Recall, mAP50, mAP50-95 được log theo epoch.
4. PR curve, F1 curve, P curve, R curve và confusion matrix xuất hiện trong media/files.
5. Epoch có mAP50-95 tốt nhất tương ứng checkpoint best.pt.

Sau fine-tune, dùng best.pt để test trên split test đúng một lần. Tách test và ứng dụng tracking/counting sang notebook khác để báo cáo rõ train-val-test.
