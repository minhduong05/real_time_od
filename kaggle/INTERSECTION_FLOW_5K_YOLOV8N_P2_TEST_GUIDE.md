# Test cuối: YOLOv8n-P2 trên Intersection-Flow-5K

Notebook này đánh giá đúng một lần checkpoint tốt nhất trên split test 723 ảnh. Nó không fine-tune, không thay đổi weights và log toàn bộ kết quả lên W&B.

## A. Chuẩn bị weights

Từ ZIP train hoặc thư mục local, lấy file best.pt. Với cấu trúc Kaggle Input hiện tại, có thể giữ nguyên tên:

~~~text
best.pt
~~~

Nếu checkpoint đã được tải về local repo, dùng file sau làm nguồn:

~~~text
models/Intersection-Flow-5K-Yolov8n-P2/best.pt
~~~

Trên Kaggle, tạo Dataset mới chỉ chứa file này, ví dụ tên dataset là intersection-flow-5k-yolov8n-p2-weights.

Trong notebook test mới, Add Input hai dataset:

1. Intersection-Flow-5K.
2. Dataset weights mới tạo.

## B. Kaggle Settings

1. Chọn GPU T4 hoặc GPU T4 x2. Test chỉ dùng GPU 0.
2. Bật Internet.
3. Add-ons -> Secrets: bật WANDB_API_KEY.
4. Dùng W&B project đã theo dõi train: intersection-flow-5k-yolov8n-p2.

## Cell 1 -- Markdown

~~~markdown
# Final test: YOLOv8n-P2 on Intersection-Flow-5K

- Checkpoint: best.pt from the completed fine-tuning run
- Split: test, 723 images
- Metrics: Precision, Recall, mAP50, mAP50-95, AP by class, speed
- Tracker: Weights & Biases
- Rule: this notebook does not train or tune the model.
~~~

## Cell 2 -- Package và GPU

~~~python
%pip install -q wandb
%pip install -q ultralytics --no-deps

import torch
import ultralytics
import wandb

print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

assert torch.cuda.is_available(), "Hãy bật GPU trong Kaggle Settings."
print(torch.zeros(1, device="cuda:0"))
~~~

## Cell 3 -- W&B login

~~~python
import os
from kaggle_secrets import UserSecretsClient

WANDB_ENTITY = "minhmit146-hanoi-university-of-science-and-technology"
WANDB_PROJECT = "intersection-flow-5k-yolov8n-p2"
WANDB_RUN_NAME = "test_yolov8n-p2_intersection-flow-5k"
WANDB_GROUP = "intersection-flow-5k-final-test"

os.environ["WANDB_ENTITY"] = WANDB_ENTITY
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_NAME"] = WANDB_RUN_NAME
os.environ["WANDB_TAGS"] = "intersection-flow-5k,yolov8n-p2,test,final-evaluation"

wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
!yolo settings wandb=True
~~~

## Cell 4 -- Kiểm tra weights, test split và classes

~~~python
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")

weight_matches = sorted(INPUT_ROOT.rglob("best.pt"))
weight_matches = [
    path for path in weight_matches
    if "weights" in str(path).lower()
]
assert len(weight_matches) == 1, f"Checkpoint không đúng: {weight_matches}"
BEST_WEIGHTS = weight_matches[0]

yaml_candidates = list(INPUT_ROOT.rglob("intersection.yaml"))
assert len(yaml_candidates) == 1, f"Dataset không đúng: {yaml_candidates}"
DATASET_ROOT = yaml_candidates[0].parent

CLASS_NAMES = [
    line.strip()
    for line in (DATASET_ROOT / "classes.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
EXPECTED_CLASSES = [
    "vehicle", "bus", "bicycle", "pedestrian",
    "engine", "truck", "tricycle", "obstacle",
]
assert CLASS_NAMES == EXPECTED_CLASSES, CLASS_NAMES

TEST_IMAGES_DIR = DATASET_ROOT / "images" / "test"
TEST_LABELS_DIR = DATASET_ROOT / "labels" / "test"
extensions = {".jpg", ".jpeg", ".png", ".bmp"}
test_images = sorted(path for path in TEST_IMAGES_DIR.iterdir()
                     if path.suffix.lower() in extensions)
test_labels = sorted(TEST_LABELS_DIR.glob("*.txt"))

print("Weights:", BEST_WEIGHTS)
print("Test images:", len(test_images))
print("Test labels:", len(test_labels))
print("Classes:", CLASS_NAMES)

assert len(test_images) == 723
assert len(test_labels) == 723
~~~

## Cell 5 -- Kiểm tra labels test và tạo YAML

~~~python
import yaml

invalid_rows = []
for label_path in test_labels:
    for number, line in enumerate(label_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
            valid = len(fields) == 5 and class_id in range(8) and all(
                0 <= value <= 1 for value in coordinates
            )
        except (ValueError, IndexError):
            valid = False
        if not valid:
            invalid_rows.append((label_path.name, number, line))

assert not invalid_rows, f"Test label lỗi: {invalid_rows[:5]}"

test_data = {
    "path": str(DATASET_ROOT),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 8,
    "names": {index: name for index, name in enumerate(CLASS_NAMES)},
}
DATA_YAML = Path("/kaggle/working/intersection_flow_5k_test.yaml")
with DATA_YAML.open("w", encoding="utf-8") as file:
    yaml.safe_dump(test_data, file, sort_keys=False)

print(DATA_YAML.read_text())
~~~

## Cell 6 -- Chạy final test và log W&B

~~~python
import hashlib
import random
import shutil

import numpy as np
import pandas as pd
from ultralytics import YOLO

IMGSZ = 640
BATCH = 16  # One GPU is used for test; batch size does not change mAP.
DEVICE = 0
IOU = 0.7
VIS_CONF = 0.25
RESULTS_ROOT = Path("/kaggle/working/intersection_flow_5k_test_results")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

def short_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:12]

random.seed(42)
sample_images = random.sample(test_images, k=min(8, len(test_images)))

run = wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    group=WANDB_GROUP,
    job_type="test",
    name=WANDB_RUN_NAME,
    tags=["intersection-flow-5k", "yolov8n-p2", "test", "final-evaluation"],
    config={
        "model": "YOLOv8n-P2",
        "weights_path": str(BEST_WEIGHTS),
        "weights_sha256_short": short_sha256(BEST_WEIGHTS),
        "dataset": "Intersection-Flow-5K",
        "split": "test",
        "imgsz": IMGSZ,
        "batch": BATCH,
        "device": DEVICE,
        "iou": IOU,
        "source_train_run": "yolov8n-p2_intersection-flow-5k",
    },
    reinit="finish_previous",
)

try:
    model = YOLO(str(BEST_WEIGHTS))
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        iou=IOU,
        plots=True,
        project=str(RESULTS_ROOT),
        name="yolov8n-p2_test",
        exist_ok=True,
        verbose=True,
    )

    box = metrics.box
    summary = {
        "test/precision": float(box.mp),
        "test/recall": float(box.mr),
        "test/mAP50": float(box.map50),
        "test/mAP50-95": float(box.map),
    }
    summary.update({
        f"speed/{name}_ms_per_image": float(value)
        for name, value in metrics.speed.items()
    })
    run.log(summary)

    rows = []
    for index, class_id in enumerate(np.asarray(box.ap_class_index, dtype=int)):
        rows.append([
            int(class_id),
            CLASS_NAMES[int(class_id)],
            float(np.asarray(box.p)[index]),
            float(np.asarray(box.r)[index]),
            float(np.asarray(box.all_ap)[index, 0]),
            float(np.asarray(box.all_ap)[index].mean()),
        ])

    per_class_df = pd.DataFrame(
        rows,
        columns=["class_id", "class_name", "precision", "recall", "AP50", "AP50-95"],
    )
    per_class_path = Path("/kaggle/working/intersection_flow_5k_test_per_class_metrics.csv")
    summary_path = Path("/kaggle/working/intersection_flow_5k_test_summary.csv")
    per_class_df.to_csv(per_class_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    run.log({"test/per_class_metrics": wandb.Table(dataframe=per_class_df)})

    for image_path in sorted(Path(metrics.save_dir).glob("*.png")):
        run.log({f"plots/{image_path.stem}": wandb.Image(str(image_path))})

    predictions = model.predict(
        source=[str(path) for path in sample_images],
        imgsz=IMGSZ,
        conf=VIS_CONF,
        iou=IOU,
        device=DEVICE,
        verbose=False,
    )
    run.log({
        "test/sample_predictions": [
            wandb.Image(result.plot()[:, :, ::-1], caption=Path(result.path).name)
            for result in predictions
        ]
    })

    display(pd.DataFrame([summary]))
    display(per_class_df)
finally:
    run.finish()
~~~

Đúng khi log quét đường dẫn images/test và labels/test, in ra metric test, đồng thời W&B có một run test riêng.

## Kết quả mong muốn

Các dấu hiệu bắt buộc để biết notebook đang test đúng dữ liệu và đúng model:

~~~text
Weights: /kaggle/input/intersection-flow-5k-yolov8n-p2-weights/best.pt
Test images: 723
Test labels: 723
Classes: ['vehicle', 'bus', 'bicycle', 'pedestrian', 'engine', 'truck', 'tricycle', 'obstacle']
~~~

Trong output của `model.val`, cần thấy Ultralytics chạy trên split test, không phải train hoặc val. Các metric chính cần ghi lại là:

~~~text
test/precision
test/recall
test/mAP50
test/mAP50-95
speed/preprocess_ms_per_image
speed/inference_ms_per_image
speed/postprocess_ms_per_image
~~~

Cách đọc nhanh để biết kết quả tốt:

1. `test/mAP50-95` là chỉ số chính để báo cáo chất lượng tổng quát. Nó càng cao càng tốt và nên gần với validation mAP50-95 của run train tốt nhất. Nếu thấp hơn validation quá nhiều, ví dụ tụt hơn khoảng 5-10 điểm phần trăm, cần kiểm tra lại weights, split test hoặc khả năng overfit.
2. `test/mAP50` thường cao hơn `test/mAP50-95`. Nếu `mAP50` ổn nhưng `mAP50-95` thấp, model bắt được object nhưng box chưa khít.
3. `precision` cao nhưng `recall` thấp nghĩa là model ít báo nhầm nhưng bỏ sót nhiều object. Với traffic monitoring, bỏ sót xe/người nhiều là vấn đề cần xem lại threshold hoặc model.
4. `recall` cao nhưng `precision` thấp nghĩa là model bắt được nhiều object nhưng có nhiều false positive. Khi đưa vào tracking/counting, trường hợp này dễ làm đếm thừa.
5. Per-class AP không nên chỉ tốt ở `vehicle` rồi rất thấp ở các class nhỏ như `bicycle`, `pedestrian`, `tricycle`, `obstacle`. Nếu các class nhỏ quá thấp, cần ghi rõ trong báo cáo vì đây là điểm yếu quan trọng của bài toán giao thông.
6. Sample predictions trên W&B phải có box nằm đúng vật thể, không lệch hàng loạt và không phát hiện tràn lan trên nền đường. Đây là bước kiểm tra trực quan bắt buộc, nhất là khi metric có vẻ đẹp bất thường.

Các dấu hiệu bất thường cần dừng lại kiểm tra:

1. `Test images` hoặc `Test labels` khác 723.
2. `Classes` sai thứ tự hoặc thiếu class.
3. `Weights` không nằm trong dataset `intersection-flow-5k-yolov8n-p2-weights`.
4. `mAP50` gần 0 hoặc sample predictions gần như không có box.
5. W&B không có run tên `test_yolov8n-p2_intersection-flow-5k` trong project `intersection-flow-5k-yolov8n-p2`.

## Kết quả test đã ghi nhận

Run test đã được log lên W&B:

~~~text
Project: intersection-flow-5k-yolov8n-p2
Run: test_yolov8n-p2_intersection-flow-5k
URL: https://wandb.ai/minhmit146-hanoi-university-of-science-and-technology/intersection-flow-5k-yolov8n-p2/runs/uiff4u94
~~~

Tổng quan trên 723 ảnh test, 40,065 object:

~~~text
Precision: 0.77578
Recall:    0.57138
mAP50:     0.64391
mAP50-95:  0.42549
Speed:     0.68 ms preprocess, 4.62 ms inference, 2.99 ms postprocess / image
~~~

Kết quả này cho thấy model có precision khá tốt nhưng recall còn trung bình. Nghĩa là khi model dự đoán thì tương đối đáng tin, nhưng vẫn bỏ sót một phần object, đặc biệt ở các class nhỏ hoặc khó.

Per-class AP:

~~~text
class        precision  recall   AP50    AP50-95
vehicle      0.87766   0.82246  0.88974  0.66825
bus          0.85176   0.74944  0.80164  0.62269
bicycle      0.78583   0.62570  0.72307  0.39832
pedestrian   0.69400   0.28792  0.39212  0.16994
engine       0.62163   0.36250  0.44207  0.30156
truck        0.76687   0.54039  0.61359  0.42121
tricycle     0.75779   0.67265  0.71375  0.53686
obstacle     0.85068   0.50999  0.57530  0.28513
~~~

Nhận xét để đưa vào báo cáo:

1. `vehicle` và `bus` là hai class tốt nhất, phù hợp vì số lượng mẫu nhiều và kích thước object thường rõ hơn.
2. `tricycle` cũng khá ổn so với số lượng mẫu ít, với AP50-95 khoảng 0.537.
3. `pedestrian` là class yếu nhất: recall khoảng 0.288 và AP50-95 khoảng 0.170. Model bỏ sót nhiều người đi bộ, có thể do object nhỏ, xa, bị che khuất hoặc mật độ giao thông cao.
4. `engine` có ít mẫu test nhất, chỉ 80 instances, nên metric dễ dao động và chưa thật chắc để kết luận mạnh.
5. `obstacle` có nhiều instance nhưng AP50-95 thấp hơn AP50 khá nhiều, cho thấy model có phát hiện được nhưng bounding box chưa khít hoặc object khó định vị.
6. Tốc độ inference khoảng 4.62 ms/image trên Tesla T4 là nhanh, phù hợp mục tiêu real-time nếu pipeline tracking/counting không quá nặng.

## Cell 7 -- Tạo ZIP kết quả test

~~~python
EXPORT_DIR = Path("/kaggle/working/intersection_flow_5k_final_test_export")
shutil.rmtree(EXPORT_DIR, ignore_errors=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

shutil.copy2(
    "/kaggle/working/intersection_flow_5k_test_summary.csv",
    EXPORT_DIR / "test_summary.csv",
)
shutil.copy2(
    "/kaggle/working/intersection_flow_5k_test_per_class_metrics.csv",
    EXPORT_DIR / "test_per_class_metrics.csv",
)
shutil.copy2(BEST_WEIGHTS, EXPORT_DIR / "best.pt")

for source in RESULTS_ROOT.rglob("*"):
    if source.is_file():
        target = EXPORT_DIR / "plots" / source.relative_to(RESULTS_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

zip_path = shutil.make_archive(
    "/kaggle/working/Intersection-Flow-5K_yolov8n-p2_test_export",
    "zip",
    root_dir=EXPORT_DIR,
)

print("Test ZIP:", zip_path)
print("W&B project:", f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")
~~~

Bấm Save Version, sau đó tải file:

~~~text
/kaggle/working/Intersection-Flow-5K_yolov8n-p2_test_export.zip
~~~

Gửi ZIP này để phân tích test và chốt ngưỡng confidence cho ứng dụng. Sau test không dùng metric test để fine-tune thêm.
