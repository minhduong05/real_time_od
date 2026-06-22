# VisDrone: đánh giá 3 mô hình đã fine-tune trên Kaggle

Tài liệu này đánh giá ba checkpoint hiện có trong repository.

~~~
models/VisDrone/yolov8n/best.pt
models/VisDrone/yolov8n-p2/best.pt
models/VisDrone/yolov8s/best.pt
~~~

Chỉ dùng **VisDrone2019-DET-test-dev** cho phần so sánh cuối cùng. Split này có nhãn công khai nên tính được mAP và AP theo từng class. Không dùng VisDrone2019-DET-test-challenge trong notebook này: nó không có nhãn công khai, chỉ dành cho việc tạo submission gửi server chính thức hoặc trực quan hóa dự đoán.

Kết quả mong đợi là ba W&B run có thể so sánh trong project test_log. Mỗi run log metric tổng, bảng AP theo class, các biểu đồ đánh giá và cùng tám ảnh dự đoán mẫu.

## A. Đưa ba weights local lên một Kaggle Dataset

Làm việc này **một lần** trước khi chạy notebook đánh giá.

1. Trên máy local, tạo một thư mục tạm, ví dụ `kaggle_weights`, rồi **copy** ba file `best.pt` vào đó. Không đổi tên file gốc trong repository.
2. Đổi tên ba bản copy để mỗi file có tên riêng:

~~~
yolov8n_best.pt
yolov8n-p2_best.pt
yolov8s_best.pt
~~~

3. Trên Kaggle, mở **Datasets -> New Dataset** rồi upload ba file vừa đổi tên. Không cần upload `last.pt`.
4. Đặt tên dataset rõ ràng, ví dụ visdrone-finetuned-weights, rồi tạo Dataset.
5. Trong notebook benchmark, bấm **Add Input** và thêm cả hai dataset:
   - VisDrone Dataset hiện tại, chứa ảnh và nhãn;
   - visdrone-finetuned-weights, chứa ba checkpoint.

Sau khi thêm dataset weights, cấu trúc có thể như sau. Tên thư mục ngoài cùng có thể khác.

~~~
/kaggle/input/visdrone-finetuned-weights/
  yolov8n_best.pt
  yolov8n-p2_best.pt
  yolov8s_best.pt
~~~

Không copy checkpoint vào /kaggle/working. Kaggle Dataset là read-only, tái sử dụng được và đảm bảo mọi lần benchmark dùng đúng cùng một weights. Nút **Upload** trong notebook chỉ phù hợp cho một file tạm thời, không phù hợp cho benchmark cuối cùng này.

## B. Thiết lập notebook Kaggle

Tạo notebook mới rồi thiết lập trước khi thêm cell:

1. **Settings -> Accelerator -> GPU T4 x2**. Code chỉ dùng `device=0`, nên thực tế một GPU T4 là đủ. Không chọn P100: PyTorch mới trên Kaggle không còn hỗ trợ kiến trúc P100 (sm_60).
2. **Settings -> Internet -> On**, để W&B gửi log trực tuyến.
3. **Add-ons -> Secrets**: thêm WANDB_API_KEY, dán API key W&B và bật secret này cho notebook.
4. Thêm hai input dataset ở phần A.

Chạy các cell dưới đây theo đúng thứ tự.

## Cell 1 -- Markdown

~~~markdown
# Đánh giá cuối: VisDrone2019-DET-test-dev

Mô hình: YOLOv8n, YOLOv8n-P2, YOLOv8s (đã fine-tune)

- Tập đánh giá: VisDrone2019-DET-test-dev
- Metric: Precision, Recall, mAP50, mAP50-95, AP theo từng class
- Theo dõi: W&B project test_log
- Quy tắc: mọi model dùng chính xác cùng cấu hình đánh giá.
~~~

## Cell 2 -- Cài thư viện an toàn và kiểm tra GPU

~~~python
# Không dùng -U ở đây: nó có thể nâng cấp PyTorch/CUDA và làm GPU Kaggle mất tương thích.
# --no-deps giữ nguyên PyTorch có sẵn trong Kaggle.
%pip install -q wandb
%pip install -q ultralytics --no-deps

import torch
import ultralytics
import wandb

print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

assert torch.cuda.is_available(), "Hãy bật GPU trong Kaggle Settings rồi restart session."
assert torch.cuda.get_device_capability(0) >= (7, 0), (
    "GPU này không tương thích PyTorch hiện tại. Hãy đổi Accelerator sang GPU T4 x2."
)

# Kiểm tra thật sự có thể chạy một tensor trên GPU.
test_tensor = torch.zeros(1, device="cuda")
print("CUDA tensor test:", test_tensor)
~~~

**Đúng khi:** CUDA available in ra True, GPU là Tesla T4 (hoặc GPU có compute capability từ 7.0) và dòng CUDA tensor test in ra thành công. Nếu thấy P100/sm_60, dừng lại và đổi Accelerator sang GPU T4 x2.

## Cell 3 -- Đăng nhập W&B qua Kaggle Secret

~~~python
import os
from kaggle_secrets import UserSecretsClient

WANDB_API_KEY = UserSecretsClient().get_secret("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

wandb.login(key=WANDB_API_KEY)
~~~

**Đúng khi:** W&B báo đăng nhập thành công. Không bao giờ in hoặc dán API key trực tiếp vào code notebook.

## Cell 4 -- Tìm và kiểm tra dữ liệu, checkpoint

Cell này tự tìm path nên không phụ thuộc vào tên slug của Kaggle Dataset. Nó kiểm tra test-dev có labels và có đúng ba file weights đã đổi tên cần dùng.

~~~python
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")

def find_one_dir(name: str) -> Path:
    matches = [path for path in INPUT_ROOT.rglob(name) if path.is_dir()]
    assert len(matches) == 1, f"Phải có đúng một thư mục '{name}', nhưng tìm thấy: {matches}"
    return matches[0]

TEST_DEV_DIR = find_one_dir("VisDrone2019-DET-test-dev")
TEST_CHALLENGE_DIR = find_one_dir("VisDrone2019-DET-test-challenge")

test_images_dir = TEST_DEV_DIR / "images"
test_labels_dir = TEST_DEV_DIR / "labels"
assert test_images_dir.is_dir(), f"Thiếu thư mục ảnh: {test_images_dir}"
assert test_labels_dir.is_dir(), f"Thiếu thư mục labels: {test_labels_dir}"
assert (TEST_CHALLENGE_DIR / "images").is_dir(), "Cấu trúc test-challenge không đúng"
assert not (TEST_CHALLENGE_DIR / "labels").exists(), (
    "test-challenge không được có nhãn công khai; hãy kiểm tra lại dataset đã chọn."
)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
test_images = sorted(path for path in test_images_dir.iterdir()
                     if path.suffix.lower() in image_extensions)
test_labels = sorted(test_labels_dir.glob("*.txt"))

print("Số ảnh test-dev:", len(test_images))
print("Số file label test-dev:", len(test_labels))
print("Số ảnh test-challenge:", len(list((TEST_CHALLENGE_DIR / "images").glob("*"))))

MODELS = {}
WEIGHT_FILENAMES = {
    "yolov8n": "yolov8n_best.pt",
    "yolov8n-p2": "yolov8n-p2_best.pt",
    "yolov8s": "yolov8s_best.pt",
}

for model_name, filename in WEIGHT_FILENAMES.items():
    matches = list(INPUT_ROOT.rglob(filename))
    assert len(matches) == 1, (
        f"Phải có đúng một file {filename} cho {model_name}; tìm thấy: {matches}. "
        "Hãy kiểm tra lại Kaggle Dataset chứa weights đã được Add Input."
    )
    MODELS[model_name] = matches[0]

print("\nCheckpoint sẽ dùng:")
for name, path in MODELS.items():
    print(f"  {name:10s} {path} ({path.stat().st_size / 1024 / 1024:.2f} MiB)")

assert len(test_images) > 0, "Không tìm thấy ảnh test-dev"
assert len(test_labels) > 0, "Không tìm thấy labels test-dev"
assert set(MODELS) == {"yolov8n", "yolov8n-p2", "yolov8s"}
~~~

**Đúng khi:** cell in ra đủ ba checkpoint và không checkpoint nào có path /kaggle/working. Số file ảnh và label không bắt buộc bằng nhau, vì ảnh không có object có thể không có file label.

## Cell 5 -- Tạo file YAML chỉ dùng để đánh giá test-dev

Thứ tự class giống configs/project.yaml của project này. Ultralytics tự tìm label bằng cách thay /images/ bằng /labels/.

~~~python
import yaml

DATA_ROOT = TEST_DEV_DIR.parent

VISDRONE_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}

test_yaml = {
    "path": str(DATA_ROOT),
    # Ultralytics yêu cầu train và val tồn tại trong mọi detection YAML,
    # dù lệnh bên dưới chỉ đánh giá split="test".
    "train": "VisDrone2019-DET-train/images",
    "val": "VisDrone2019-DET-val/images",
    "test": "VisDrone2019-DET-test-dev/images",
    "names": VISDRONE_NAMES,
}

DATA_YAML = Path("/kaggle/working/visdrone_test_dev.yaml")
with DATA_YAML.open("w", encoding="utf-8") as file:
    yaml.safe_dump(test_yaml, file, sort_keys=False, allow_unicode=True)

print(DATA_YAML.read_text())
yaml_text = DATA_YAML.read_text()
assert "VisDrone2019-DET-train/images" in yaml_text
assert "VisDrone2019-DET-val/images" in yaml_text
assert "VisDrone2019-DET-test-dev/images" in yaml_text
~~~

**Đúng khi:** YAML có cả train, val và test. Hai key train/val chỉ thỏa điều kiện của Ultralytics; lệnh benchmark vẫn dùng `split="test"`, nên dữ liệu thực sự được đánh giá là VisDrone2019-DET-test-dev/images.

## Cell 6 -- Thiết lập cấu hình benchmark chung

Không được đổi cấu hình riêng cho một model. Nếu GPU out of memory, đổi BATCH thành 8 rồi chạy lại **cả ba model** với batch mới đó.

~~~python
WANDB_ENTITY = "minhmit146-hanoi-university-of-science-and-technology"
WANDB_PROJECT = "test_log"
GROUP_NAME = "visdrone-test-dev-final-benchmark"

IMGSZ = 640
BATCH = 16
DEVICE = 0
IOU = 0.7
VIS_CONF = 0.25  # chỉ dùng để trực quan hóa, không ảnh hưởng mAP

RESULTS_ROOT = Path("/kaggle/working/benchmark_results")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

print({
    "dataset": "VisDrone2019-DET-test-dev",
    "models": list(MODELS),
    "imgsz": IMGSZ,
    "batch": BATCH,
    "device": DEVICE,
    "iou": IOU,
})
~~~

## Cell 7 -- Khai báo hàm đánh giá

Hàm này tạo một W&B run riêng cho mỗi model, log metric tổng, AP theo class, biểu đồ do Ultralytics tạo và tám ảnh dự đoán giống nhau.

~~~python
import hashlib
import random

import numpy as np
import pandas as pd
from ultralytics import YOLO

def short_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:12]

random.seed(42)
SAMPLE_IMAGES = random.sample(test_images, k=min(8, len(test_images)))

def evaluate_one_model(model_name: str, weights: Path) -> dict:
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=GROUP_NAME,
        job_type="test",
        name=model_name,
        config={
            "model_name": model_name,
            "weights_path": str(weights),
            "weights_sha256_short": short_sha256(weights),
            "dataset": "VisDrone2019-DET-test-dev",
            "imgsz": IMGSZ,
            "batch": BATCH,
            "device": DEVICE,
            "iou": IOU,
        },
        tags=["visdrone", "test-dev", "final-benchmark"],
        # W&B bản mới dùng finish_previous thay cho giá trị finish cũ.
        reinit="finish_previous",
    )

    try:
        model = YOLO(str(weights))
        metrics = model.val(
            data=str(DATA_YAML),
            split="test",
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            iou=IOU,
            plots=True,
            project=str(RESULTS_ROOT),
            name=model_name,
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
            f"speed/{key}_ms_per_image": float(value)
            for key, value in metrics.speed.items()
        })
        run.log(summary)

        class_ids = np.asarray(box.ap_class_index, dtype=int)
        all_ap = np.asarray(box.all_ap)
        precisions = np.asarray(box.p)
        recalls = np.asarray(box.r)

        per_class_rows = []
        for index, class_id in enumerate(class_ids):
            per_class_rows.append([
                int(class_id),
                VISDRONE_NAMES[int(class_id)],
                float(precisions[index]),
                float(recalls[index]),
                float(all_ap[index, 0]),
                float(all_ap[index].mean()),
            ])

        run.log({
            "test/per_class_metrics": wandb.Table(
                columns=[
                    "class_id", "class_name", "precision", "recall",
                    "AP50", "AP50-95",
                ],
                data=per_class_rows,
            )
        })

        for image_path in sorted(Path(metrics.save_dir).glob("*.png")):
            run.log({f"plots/{image_path.stem}": wandb.Image(str(image_path))})

        predictions = model.predict(
            source=[str(path) for path in SAMPLE_IMAGES],
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

        return {
            "model": model_name,
            "Precision": float(box.mp),
            "Recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
            "Inference ms/image": float(metrics.speed["inference"]),
        }
    finally:
        run.finish()
~~~

## Cell 8 -- Chạy đánh giá cả ba model

~~~python
benchmark_rows = []

for model_name, weight_path in MODELS.items():
    print(f"\n{'=' * 72}\nĐang đánh giá: {model_name}\n{'=' * 72}")
    benchmark_rows.append(evaluate_one_model(model_name, weight_path))

comparison_df = (
    pd.DataFrame(benchmark_rows)
    .sort_values("mAP50-95", ascending=False)
    .reset_index(drop=True)
)

comparison_df
~~~

**Đúng khi, với từng model:** Ultralytics cho biết đang quét VisDrone2019-DET-test-dev, in Precision/Recall/mAP và W&B in ra link của run. Nếu log cho thấy nó quét val, hãy dừng: YAML hoặc split đang sai.

## Cell 9 -- Kiểm tra và lưu bảng kết quả cuối

~~~python
assert len(comparison_df) == 3, "Benchmark phải có đúng ba dòng."
assert set(comparison_df["model"]) == set(MODELS), "Thiếu model hoặc bị lặp model."

metric_columns = ["Precision", "Recall", "mAP50", "mAP50-95"]
assert not comparison_df[metric_columns].isna().any().any(), "Có metric bị thiếu."
assert comparison_df[metric_columns].apply(
    lambda column: column.between(0, 1).all()
).all(), "Mọi metric phải thuộc đoạn [0, 1]."

comparison_path = Path("/kaggle/working/visdrone_test_dev_comparison.csv")
comparison_df.to_csv(comparison_path, index=False)

display(comparison_df)
print(f"Đã lưu bảng so sánh: {comparison_path}")
print("W&B group:", GROUP_NAME)
~~~

**Đúng khi:** có bảng ba dòng sắp xếp theo mAP50-95 và file CSV xuất hiện ở Kaggle Output. Bấm **Save Version** để lưu kết quả vĩnh viễn.

## C. Kiểm tra lại trên W&B

Mở W&B project test_log, lọc theo group:

~~~
visdrone-test-dev-final-benchmark
~~~

Phải có đúng ba test runs:

~~~
yolov8n
yolov8n-p2
yolov8s
~~~

Trong từng run, kiểm tra:

1. Summary có test/mAP50 và test/mAP50-95.
2. test/per_class_metrics có AP của từng class.
3. Có plots/PR_curve và confusion matrix.
4. test/sample_predictions có cùng tên ảnh cho cả ba model.

Dùng mAP50-95 làm metric xếp hạng chính. Dùng mAP50, AP theo class và tốc độ inference để giải thích sự khác nhau. Không fine-tune tiếp dựa vào test-dev vì đây là tập so sánh cuối cùng.

## D. Tùy chọn: test-challenge

Sau khi chọn model tốt nhất từ test-dev, dùng test-challenge để tạo submission gửi server đánh giá chính thức của VisDrone. Không thể tự tính local mAP, AP theo class, Precision, Recall hoặc confusion matrix trên split này vì nhãn bị ẩn.

## E. Export toàn bộ kết quả để viết báo cáo hoặc gửi phân tích

Chạy các cell này sau khi cả ba model đã hoàn thành. Chúng không chạy lại model và không làm thay đổi metric.

### Cell 10 -- Tải bảng AP từng class từ W&B về dạng CSV

Cell này tải table `test/per_class_metrics` đã log trong ba W&B runs. Nó bỏ qua các run lỗi/không có metric test, nếu có.

~~~python
import json

EXPORT_DIR = Path("/kaggle/working/visdrone_report_export")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

api = wandb.Api()
remote_runs = [
    run for run in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    if run.group == GROUP_NAME
    and run.name in MODELS
    and "test/mAP50-95" in run.summary
]

assert len(remote_runs) == 3, (
    f"Cần đúng 3 W&B test runs hoàn chỉnh, nhưng tìm thấy {len(remote_runs)}: "
    f"{[(run.name, run.id) for run in remote_runs]}"
)

all_per_class = []
for remote_run in remote_runs:
    table_files = [
        file for file in remote_run.files()
        if "per_class_metrics" in file.name and file.name.endswith(".table.json")
    ]
    assert table_files, f"Không tìm thấy bảng AP theo class cho run {remote_run.name}"

    # Mỗi run chỉ có một bảng; lấy bảng mới nhất nếu W&B có nhiều bản ghi.
    table_file = table_files[-1]
    run_export_dir = EXPORT_DIR / remote_run.name
    table_file.download(root=str(run_export_dir), replace=True)
    local_table_path = run_export_dir / table_file.name

    payload = json.loads(local_table_path.read_text())
    per_class_df = pd.DataFrame(payload["data"], columns=payload["columns"])
    per_class_df.insert(0, "model", remote_run.name)
    per_class_df.to_csv(
        EXPORT_DIR / f"{remote_run.name}_per_class_metrics.csv", index=False
    )
    all_per_class.append(per_class_df)

per_class_comparison_df = pd.concat(all_per_class, ignore_index=True)
per_class_path = EXPORT_DIR / "visdrone_test_dev_per_class_comparison.csv"
per_class_comparison_df.to_csv(per_class_path, index=False)

display(per_class_comparison_df)
print("Đã lưu:", per_class_path)
~~~

### Cell 11 -- Gom bảng và biểu đồ thành một ZIP

~~~python
import shutil

# Bảng tổng đã tạo ở Cell 9.
shutil.copy2(comparison_path, EXPORT_DIR / comparison_path.name)

# Các biểu đồ Ultralytics: PR/F1/P/R curves, confusion matrix và ảnh validation.
shutil.copytree(
    RESULTS_ROOT,
    EXPORT_DIR / "benchmark_results",
    dirs_exist_ok=True,
)

zip_base = "/kaggle/working/visdrone_test_dev_report_export"
zip_path = shutil.make_archive(zip_base, "zip", root_dir=EXPORT_DIR)

print("File ZIP để tải về:", zip_path)
print("\nNội dung export:")
for path in sorted(EXPORT_DIR.rglob("*")):
    if path.is_file():
        print(path.relative_to(EXPORT_DIR))
~~~

Sau đó mở phần **Output** ở cạnh phải Kaggle, tải file sau về máy:

~~~text
/kaggle/working/visdrone_test_dev_report_export.zip
~~~

Để mình phân tích, chỉ cần gửi file ZIP này. Nếu chỉ cần phân tích định lượng, gửi tối thiểu hai file CSV sau cũng đủ:

~~~text
visdrone_test_dev_comparison.csv
visdrone_test_dev_per_class_comparison.csv
~~~

## F. Export lại từ W&B khi notebook cũ đã tắt

Không cần chạy lại train hoặc test. W&B đã lưu log, bảng và media trên cloud. Tạo một notebook Kaggle nhỏ mới, bật **Internet**, thêm secret WANDB_API_KEY, và chạy các cell sau. Không cần GPU, không cần thêm dataset VisDrone hay weights.

### Cell F1 -- Đăng nhập W&B

~~~python
%pip install -q wandb pandas

import json
import shutil
from pathlib import Path

import pandas as pd
import wandb
from kaggle_secrets import UserSecretsClient

wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
~~~

### Cell F2 -- Liệt kê các run sẽ export

Cell này lấy cả train/val từ project train cũ và test từ project test_log. Run test lỗi không có mAP sẽ tự bị bỏ qua.

~~~python
ENTITY = "minhmit146-hanoi-university-of-science-and-technology"
TRAIN_PROJECT = "-kaggle-working-experiments-visdrone"
TEST_PROJECT = "test_log"
MODEL_NAMES = {"yolov8n", "yolov8n-p2", "yolov8s"}
TEST_GROUP = "visdrone-test-dev-final-benchmark"

api = wandb.Api()
selected_runs = []

for project in (TRAIN_PROJECT, TEST_PROJECT):
    for run in api.runs(f"{ENTITY}/{project}"):
        if run.name not in MODEL_NAMES:
            continue

        summary = dict(run.summary)
        if project == TEST_PROJECT:
            if run.group != TEST_GROUP or "test/mAP50-95" not in summary:
                continue

        selected_runs.append((project, run))

run_overview = pd.DataFrame([
    {
        "project": project,
        "run_name": run.name,
        "run_id": run.id,
        "state": run.state,
        "created_at": run.created_at,
        "mAP50": dict(run.summary).get("test/mAP50", dict(run.summary).get("metrics/mAP50(B)")),
        "mAP50-95": dict(run.summary).get("test/mAP50-95", dict(run.summary).get("metrics/mAP50-95(B)")),
    }
    for project, run in selected_runs
])

display(run_overview)
assert len([run for project, run in selected_runs if project == TEST_PROJECT]) == 3, (
    "Cần có đúng ba test runs hoàn chỉnh."
)
~~~

Kiểm tra bảng này trước khi sang cell tiếp theo. Có thể có nhiều run train cùng tên nếu đã train lại nhiều lần; cell sẽ export tất cả chúng để không làm mất dữ liệu.

### Cell F3 -- Tải lịch sử metric, bảng và toàn bộ media về ZIP

~~~python
EXPORT_ROOT = Path("/kaggle/working/wandb_visdrone_export")
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

for project, run in selected_runs:
    run_dir = EXPORT_ROOT / project / f"{run.name}_{run.id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Lưu config và summary: metric train/val/test cuối cùng.
    (run_dir / "config.json").write_text(
        json.dumps(dict(run.config), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(dict(run.summary), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # Train/val history theo epoch; test thường chỉ có một hàng metric.
    history = run.history(samples=100_000)
    history.to_csv(run_dir / "history.csv", index=False)

    # Tải plots, images và W&B tables; không tải checkpoint lớn.
    for remote_file in run.files():
        if remote_file.name.startswith("media/"):
            remote_file.download(root=str(run_dir), replace=True)

    print("Đã export:", project, run.name, run.id)

zip_path = shutil.make_archive(
    "/kaggle/working/wandb_visdrone_export",
    "zip",
    root_dir=EXPORT_ROOT,
)
print("Tải file này ở Kaggle Output:", zip_path)
~~~

File ZIP có:

- history.csv: metric train/val theo epoch và metric test;
- summary.json: metric cuối cùng của mỗi run;
- media/images: biểu đồ loss, mAP, PR/F1/P/R curve, confusion matrix và ảnh prediction;
- media/table: bảng AP theo từng class của test.

Sau khi chạy xong, bấm **Save Version** cho notebook export nhỏ này, rồi tải:

~~~text
/kaggle/working/wandb_visdrone_export.zip
~~~

Gửi ZIP này để được phân tích đầy đủ. Nếu file lớn, gửi riêng các file history.csv, summary.json, test_per_class_metrics table JSON và các ảnh PR/confusion matrix.
