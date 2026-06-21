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

Theo README chính thức, mọi ảnh có độ phân giải 1920x1080; object xa có thể chỉ 15x15 pixel, cảnh có mật độ cao và che khuất nặng. Repository chính thức minh họa train YOLOv8 với imgsz 640. Primary run trong guide này giữ imgsz 640 để bám baseline của tác giả, dễ tái lập và phù hợp với ràng buộc real-time. Kiến trúc P2 là phần cải tiến giúp giữ đặc trưng độ phân giải cao cho object nhỏ sau khi resize.

## Cấu hình primary run được chọn

Report tham khảo cho thấy YOLO với Mosaic, flip, HSV augmentation, warmup và cosine schedule là workflow phù hợp. Tuy nhiên các số mAP trong bảng report được ghi là minh họa, nên không dùng chúng như bằng chứng rằng một cấu hình đã tối ưu. Primary run dưới đây là cấu hình có cơ sở kỹ thuật cho YOLOv8n-P2, còn best.pt phải được chọn bằng mAP50-95 trên validation.

~~~text
Fine-tune toàn bộ network: không freeze layer
Tối đa epochs:                200
Early-stopping patience:       30
Image size:                   640
Batch global, T4 x2:           32
Optimizer:                   AdamW
Learning rate ban đầu:       0.001
Learning-rate cuối:          lr0 x 0.01
Scheduler:                   cosine
Weight decay:                0.0005
Warmup:                      3 epochs
Mosaic:                      1.0, tắt ở 15 epoch cuối
Horizontal flip:             0.5
HSV hue/saturation/value:    0.015 / 0.7 / 0.4
MixUp:                       0.0
Seed:                        42
~~~

Lý do: imgsz 640 là baseline chính thức từ dataset; P2 là thay đổi kiến trúc có chủ đích để tăng độ nhạy với object nhỏ/xa mà không tăng input resolution. AdamW với lr 0.001 phù hợp cho transfer learning; Mosaic hỗ trợ cảnh đông, còn close_mosaic giúp model ổn định trên ảnh tự nhiên ở giai đoạn cuối. Không thêm Albumentations thủ công, vì augmentation chuẩn của Ultralytics đã đủ và tránh vô tình làm sai bounding box.

imgsz 960 là một ablation có thể chạy sau này nếu cần chứng minh input resolution cao hơn giúp class nhỏ, nhưng không gọi nó là tốt hơn nếu chưa so sánh trên validation. Không dùng 960 cho primary run hiện tại.

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

1. Trong Settings -> Accelerator, chọn **GPU T4 x2**. Primary run dùng cả hai GPU qua device 0,1.
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
# Kaggle đã có PyTorch, OpenCV, NumPy và PyYAML. Không dùng
# pip install -r kaggle/requirements.txt ở đây vì resolver có thể nâng
# dependency CUDA/Numba không liên quan tới YOLO.
%pip install -q wandb
%pip install -q ultralytics --no-deps

import torch
import ultralytics
import wandb

print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("CUDA available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "Hãy bật GPU T4 x2 trong Kaggle Settings."
assert torch.cuda.device_count() >= 2, "Cần chọn GPU T4 x2 trong Kaggle Settings."

for device_id in range(2):
    print(f"GPU {device_id}:", torch.cuda.get_device_name(device_id))
    assert torch.cuda.get_device_capability(device_id) >= (7, 0), (
        "GPU không tương thích PyTorch hiện tại. Chọn GPU T4 x2 thay cho P100."
    )
    test_tensor = torch.zeros(1, device=f"cuda:{device_id}")
    print(f"CUDA tensor test GPU {device_id}:", test_tensor)
~~~

**Đúng khi:** CUDA available là True, có hai GPU Tesla T4 và tensor test chạy thành công trên GPU 0 lẫn GPU 1.

## Cell 4 -- Đăng nhập và cấu hình W&B

Đổi W&B project nếu bạn muốn tên khác. Entity là workspace hiện tại của bạn.

~~~python
import os
from kaggle_secrets import UserSecretsClient

WANDB_ENTITY = "minhmit146-hanoi-university-of-science-and-technology"
WANDB_PROJECT = "intersection-flow-5k-yolov8n-p2"
WANDB_RUN_NAME = "yolov8n-p2_intersection-flow-5k"
WANDB_TAGS = "intersection-flow-5k,yolov8n-p2,traffic-monitoring,primary-run"

os.environ["WANDB_ENTITY"] = WANDB_ENTITY
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_NAME"] = WANDB_RUN_NAME
os.environ["WANDB_TAGS"] = WANDB_TAGS

wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
!yolo settings wandb=True
~~~

**Đúng khi:** W&B login thành công. Khi train bắt đầu, terminal sẽ in URL project và run.

W&B sẽ lưu config của toàn bộ argument train, tags, history loss/metric/LR theo epoch, GPU system metrics và media do Ultralytics tạo.

Khi device là 0,1, Ultralytics tự chạy DDP. W&B chỉ tạo một run chính từ rank 0, không tạo hai run trùng lặp.

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

EXPECTED_IMAGE_COUNTS = {"train": 5483, "val": 722, "test": 723}

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
    assert image_count == EXPECTED_IMAGE_COUNTS[split], (
        f"Split {split} cần có {EXPECTED_IMAGE_COUNTS[split]} ảnh theo README, "
        f"nhưng hiện có {image_count}. Kiểm tra lại Kaggle Dataset."
    )
~~~

Trong ảnh cấu trúc dataset còn có thư mục annotations. Thư mục đó không dùng ở đây: Ultralytics train trực tiếp từ labels, vì labels đã là YOLO format.

## Cell 5b -- Xác thực toàn bộ labels là YOLO format

Cell này đọc toàn bộ file label trước khi train. Nó kiểm tra mỗi object có đúng năm giá trị: class_id, x_center, y_center, width, height; class_id thuộc 0 đến 7 và bốn tọa độ đã chuẩn hóa trong đoạn 0 đến 1.

Nếu chỉ có một số rất nhỏ row lỗi, cell lưu manifest để truy vết. Input dataset gốc là read-only; Ultralytics sẽ tự bỏ các ảnh chứa label lỗi trong lúc quét dữ liệu. Với dataset này chỉ có 20 ảnh lỗi, tỷ lệ rất nhỏ và không ảnh hưởng đáng kể đến primary run.

~~~python
from collections import Counter

import pandas as pd

# Luôn lấy source từ Kaggle Input, kể cả khi cell được chạy lại.
SOURCE_DATASET_ROOT = SOURCE_YAML.parent

def valid_yolo_line(line: str):
    fields = line.split()
    if len(fields) != 5:
        return None
    try:
        class_id = int(fields[0])
        coordinates = [float(value) for value in fields[1:]]
    except ValueError:
        return None
    if class_id not in range(len(CLASS_NAMES)):
        return None
    if not all(0.0 <= value <= 1.0 for value in coordinates):
        return None
    return class_id

class_counts = Counter()
invalid_rows = []
valid_object_count = 0

for split in ("train", "val", "test"):
    labels_dir = SOURCE_DATASET_ROOT / "labels" / split
    for label_path in sorted(labels_dir.glob("*.txt")):
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            class_id = valid_yolo_line(line)
            if class_id is None:
                invalid_rows.append({
                    "split": split,
                    "label_file": label_path.name,
                    "line_number": line_number,
                    "raw_label": line,
                })
                continue
            class_counts[CLASS_NAMES[class_id]] += 1
            valid_object_count += 1

raw_object_count = valid_object_count + len(invalid_rows)
invalid_fraction = len(invalid_rows) / raw_object_count
print("Tổng object hợp lệ:", valid_object_count)
print("Số row label lỗi:", len(invalid_rows), f"({invalid_fraction:.6%})")
print("Số object theo class:")
for class_name in CLASS_NAMES:
    print(f"  {class_name:12s} {class_counts[class_name]}")

assert valid_object_count > 0, "Không đọc được object hợp lệ nào từ labels."
assert len(invalid_rows) <= 100 and invalid_fraction < 0.001, (
    "Có quá nhiều label lỗi; dừng lại để kiểm tra dataset thay vì tự làm sạch."
)

invalid_manifest = Path("/kaggle/working/intersection_flow_5k_invalid_label_rows.csv")
pd.DataFrame(invalid_rows).to_csv(invalid_manifest, index=False)

# Dùng input dataset gốc. Ultralytics tự bỏ toàn bộ ảnh có label corrupt.
# Không dùng symlink labels vì Ultralytics resolve image symlink về source path.
DATASET_ROOT = SOURCE_DATASET_ROOT
print("Training dataset root:", DATASET_ROOT)
print("Manifest các row lỗi:", invalid_manifest)
~~~

**Đúng khi:** không có AssertionError; mọi class đều được in số lượng object và manifest ghi rõ row lỗi. Với bản dataset đang dùng, có 20 row lỗi trên hơn 406 nghìn object. Ultralytics sẽ tự bỏ 20 ảnh train chứa các row này; tỷ lệ 20/5483 ảnh rất nhỏ, đồng thời validation/test không có label corrupt.

## Cell 6 -- Tạo YAML an toàn cho Kaggle

Không dùng trực tiếp intersection.yaml từ dataset, vì path có thể không đúng khi mount vào Kaggle. YAML dưới đây dùng path tuyệt đối và đúng thứ tự 8 class.

~~~python
import yaml

intersection_yaml = {
    "path": str(DATASET_ROOT),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": len(CLASS_NAMES),
    "names": {index: name for index, name in enumerate(CLASS_NAMES)},
}

DATA_YAML = Path("/kaggle/working/intersection_flow_5k.yaml")
with DATA_YAML.open("w", encoding="utf-8") as file:
    yaml.safe_dump(intersection_yaml, file, sort_keys=False, allow_unicode=True)

print(DATA_YAML.read_text())
assert len(intersection_yaml["names"]) == 8
assert intersection_yaml["nc"] == 8
~~~

## Cell 7 -- Kiểm tra YOLOv8n-P2 và pretrained transfer

YOLOv8n-P2 là kiến trúc P2 dùng lại được cho nhiều dataset, không phải checkpoint đã fine-tune trên VisDrone. Lệnh này tạo kiến trúc từ file YAML rồi transfer pretrained weights từ `yolov8n.pt` của COCO. Config có nc là 10 như một placeholder tương thích với lần train VisDrone trước; khi train Intersection-Flow-5K, Ultralytics tự đổi thành nc là 8 theo data YAML. Đó là hành vi đúng.

~~~python
!python kaggle/check_model.py --model yolov8n-p2
~~~

**Đúng khi:**

- log có model yolov8n-p2;
- log có transfer_from yolov8n.pt, không phải models/VisDrone/.../best.pt;
- pretrained weights được transfer vào kiến trúc P2;
- không có lỗi tải model.

## Cell 8 -- Dry-run kiểm tra argument

Dry-run chưa train model. Nó chỉ xác nhận dataset, model name, output path và export path.

~~~python
!python kaggle/train.py \
  --dataset-name Intersection-Flow-5K \
  --data /kaggle/working/intersection_flow_5k.yaml \
  --model yolov8n-p2 \
  --epochs 200 \
  --imgsz 640 \
  --batch 32 \
  --workers 4 \
  --device 0,1 \
  --optimizer AdamW \
  --lr0 0.001 \
  --lrf 0.01 \
  --weight-decay 0.0005 \
  --warmup-epochs 3 \
  --cos-lr \
  --patience 30 \
  --mosaic 1.0 \
  --close-mosaic 15 \
  --fliplr 0.5 \
  --hsv-h 0.015 \
  --hsv-s 0.7 \
  --hsv-v 0.4 \
  --mixup 0.0 \
  --seed 42 \
  --save-period 25 \
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
dataset nc: 8
dataset names: ['vehicle', 'bus', 'bicycle', 'pedestrian', 'engine', 'truck', 'tricycle', 'obstacle']
output: /kaggle/working/experiments/intersection-flow-5k/yolov8n-p2
export: /kaggle/working/export/Intersection-Flow-5K/yolov8n-p2
~~~

Nếu sai path, dừng tại đây và sửa trước khi train.

## Cell 9 -- Fine-tune toàn bộ YOLOv8n-P2, tối đa 200 epochs

Đây là cell train thật. Không truyền freeze, nên toàn bộ backbone và P2 head đều được fine-tune. Validation chạy sau mọi epoch, vì vậy W&B có train loss, val loss, Precision, Recall, mAP50 và mAP50-95 theo epoch.

imgsz 640 bám ví dụ train chính thức của Intersection-Flow-5K và giúp kết quả tái lập, đồng thời hợp với mục tiêu real-time. Batch 32 là batch toàn cục cho hai T4, tương đương 16 ảnh mỗi GPU. Nếu thiếu VRAM, chỉ hạ batch toàn cục trước; giữ imgsz 640 cho primary run.

~~~python
!python kaggle/train.py \
  --dataset-name Intersection-Flow-5K \
  --data /kaggle/working/intersection_flow_5k.yaml \
  --model yolov8n-p2 \
  --epochs 200 \
  --imgsz 640 \
  --batch 32 \
  --workers 4 \
  --device 0,1 \
  --optimizer AdamW \
  --lr0 0.001 \
  --lrf 0.01 \
  --weight-decay 0.0005 \
  --warmup-epochs 3 \
  --cos-lr \
  --patience 30 \
  --mosaic 1.0 \
  --close-mosaic 15 \
  --fliplr 0.5 \
  --hsv-h 0.015 \
  --hsv-s 0.7 \
  --hsv-v 0.4 \
  --mixup 0.0 \
  --seed 42 \
  --save-period 25 \
  --project /kaggle/working/experiments/intersection-flow-5k \
  --name yolov8n-p2 \
  --export-dir /kaggle/working/export
~~~

Nếu out-of-memory, đổi batch 32 thành batch 16 và chạy lại Cell 9 từ đầu. Không đổi cấu hình giữa chừng.

**Đúng khi trong log:**

1. Dry-run trước đó đã in dataset nc: 8. Khi train thật, log phải có dòng model YAML nc 10 bị ghi đè thành nc 8. Đây không phải lỗi.
2. Mỗi epoch có train loss và validation metric.
3. W&B in URL run của project intersection-flow-5k-yolov8n-p2.
4. Early stopping có thể kết thúc trước epoch 200 nếu mAP validation không cải thiện trong 30 epoch; đây là hành vi đúng.
5. Cuối train có checkpoint best.pt và last.pt được export.

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
    RUN_DIR / "confusion_matrix.png",
    RUN_DIR / "confusion_matrix_normalized.png",
]

# Ultralytics versions use either BoxPR_curve.png or PR_curve.png.
pr_curve_candidates = [RUN_DIR / "BoxPR_curve.png", RUN_DIR / "PR_curve.png"]
pr_curve = next((path for path in pr_curve_candidates if path.exists()), None)

for path in required_files:
    print(("OK  " if path.exists() else "MISS"), path)
print(("OK  " if pr_curve else "MISS"), pr_curve or "BoxPR_curve.png / PR_curve.png")

assert (EXPORT_DIR / "best.pt").exists(), "Không có best.pt"
assert (RUN_DIR / "results.csv").exists(), "Không có results.csv"
assert pr_curve is not None, "Không có Precision-Recall curve"

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

Sau khi tải ZIP về local, giải nén và đặt checkpoint vào:

~~~text
models/Intersection-Flow-5K-Yolov8n-P2/best.pt
models/Intersection-Flow-5K-Yolov8n-P2/last.pt
models/Intersection-Flow-5K-Yolov8n-P2/run_info.yaml
~~~

## Kiểm tra W&B sau train

Trong project intersection-flow-5k-yolov8n-p2, kiểm tra:

Trong project intersection-flow-5k-yolov8n-p2, mở đúng run có tag primary-run và kiểm tra các nhóm sau:

1. **Config**: dataset path, class names, model yolov8n-p2, epochs 200, imgsz 640, batch, AdamW, lr0, lrf, Mosaic, close_mosaic, seed và save_period 25 đều xuất hiện. Đây là bằng chứng tái lập thí nghiệm.
2. **History theo epoch**: train box_loss, cls_loss, dfl_loss; val box_loss, cls_loss, dfl_loss; metrics Precision, Recall, mAP50 và mAP50-95. Train loss nên giảm, còn model tốt nhất được quyết định bởi validation mAP50-95.
3. **Learning rate**: các panel lr/pg0, lr/pg1, lr/pg2 phải cho thấy warmup rồi giảm theo cosine schedule.
4. **System**: GPU utilization, GPU memory, CPU/RAM, thời gian mỗi epoch. Các chỉ số này cần để phân tích trade-off accuracy và tài nguyên.
5. **Media**: results.png, PR curve, F1 curve, Precision curve, Recall curve, confusion matrix, confusion matrix normalized, labels và ảnh train/val sample.
6. **Files/checkpoints**: Kaggle output phải có results.csv, best.pt, last.pt và checkpoint mốc epoch 25/50/75... do save_period 25. best.pt phải tương ứng epoch có validation mAP50-95 tốt nhất. W&B thường hiển thị model artifact cuối; file ZIP ở Cell 11 là bản lưu chắc chắn để tải tất cả checkpoint về.

Sau fine-tune, dùng best.pt để test trên split test đúng một lần. Tách test và ứng dụng tracking/counting sang notebook khác để báo cáo rõ train-val-test.

## Ghi chú từ README chính thức

- Dùng thư mục labels cho YOLO training; annotations là PASCAL VOC XML, còn test_coco.json chỉ hữu ích khi cần COCO evaluation.
- Không chạy test trong lúc chọn epoch hoặc chỉnh hyperparameter. Train dùng train, chọn best.pt dựa trên val, sau đó test đúng một lần trên images/test.
- Khi viết báo cáo, trích dẫn dataset: Zhao, Y. và Wang, Z., FlowDet: Overcoming Perspective and Scale Challenges in Real-Time End-to-End Traffic Detection, PRCV 2025.
- Dataset có license CC BY-NC-SA 4.0; sử dụng trong phạm vi học tập/nghiên cứu phi thương mại là phù hợp, nhưng ghi rõ nguồn dataset trong báo cáo.
