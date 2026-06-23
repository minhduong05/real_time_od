# Project 2: Real-Time Object Detection With VisDrone YOLOv8 (YOLOv8n, YOLOv8-P2, YOLOv8s)

**Người thực hiện:** Trần Tuấn Minh - 20230051

Repository này được xây dựng cho bài toán phát hiện đối tượng giao thông từ video, tập trung vào huấn luyện, đánh giá và triển khai thử nghiệm các mô hình YOLOv8 đã fine-tune trên bộ dữ liệu VisDrone.

Các mục tiêu chính:

1. Fine-tune và benchmark ba kiến trúc YOLOv8 trên bộ dữ liệu VisDrone.
2. Xây dựng giao diện realtime để trực quan hóa kết quả phát hiện, tracking, ước lượng mật độ và đếm phương tiện trên video local.
3. Xuất kết quả suy luận dạng annotated video, CSV và summary JSON nhằm phục vụ phân tích định lượng, đối chiếu thực nghiệm và tổng hợp kết quả.

Project hiện chỉ tập trung vào 3 checkpoint VisDrone:

```text
models/VisDrone/yolov8n/best.pt
models/VisDrone/yolov8n-p2/best.pt
models/VisDrone/yolov8s/best.pt
```

Các class VisDrone:

```text
pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor
```

## Cấu Trúc Repo

```text
app/
  realtime_front.py          Entry point chạy frontend realtime local
  detection.py               CLI export object-detection-only cho video/folder

src/realtime_od/
  realtime_app.py            Flask routes/API cho frontend
  realtime_template.py       HTML/CSS/JS giao diện realtime
  realtime_state.py          Runtime config, pause/resume/finish state, cache YOLO model
  realtime_stream.py         Đọc video, chạy YOLO predict/track, stream MJPEG
  realtime_draw.py           Vẽ bbox, track trail, density zone, counting line, overlay
  realtime_logger.py         Ghi CSV log từng frame khi bật Log CSV
  detection_export.py        Logic batch export annotated.mp4/detections.csv
  model_registry.py          Khai báo 3 model VisDrone dùng chung cho frontend/export
  config.py                  Đọc configs/project.yaml và resolve path
  types.py                   Dataclass Detection dùng chung

configs/
  project.yaml               Class, model, inference và Kaggle defaults
  models/yolov8n-p2.yaml     Kiến trúc YOLOv8n-P2 custom

kaggle/
  train.py                   Script train/tune trên Kaggle
  check_model.py             Kiểm tra model config/pretrained transfer
  requirements.txt           Dependencies tối thiểu cho Kaggle

models/VisDrone/
  yolov8n/                   Đặt best.pt/last.pt của YOLOv8n
  yolov8n-p2/                Đặt best.pt/last.pt của YOLOv8n-P2
  yolov8s/                   Đặt best.pt/last.pt của YOLOv8s

experiments/                 Lưu số liệu W&B/benchmark tải về để phân tích báo cáo
video/                       Video local để frontend/batch export đọc
outputs/                     Kết quả sinh ra khi chạy app, không commit git
docs/                        Hướng dẫn chi tiết theo từng workflow
scripts/verify_setup.py      Kiểm tra nhanh layout local
```

Repo cho phép commit các file `models/VisDrone/*/best.pt` để người clone có thể chạy demo ngay. Các file video, output, dữ liệu tải về trong `experiments/`, và checkpoint phụ như `last.pt` vẫn bị `.gitignore`.

## Cài Đặt Local

```powershell
conda activate real_time_od
pip install -r requirements.txt
python scripts/verify_setup.py
```

`verify_setup.py` sẽ in ra class, model folder và trạng thái tồn tại của 3 checkpoint VisDrone.

## Chạy Frontend Realtime

```powershell
conda activate real_time_od
python app/realtime_front.py
```

Mở browser:

```text
http://127.0.0.1:7860
```

Luồng xử lý realtime:

```text
video frame gốc
  -> YOLOv8n / YOLOv8n-P2 / YOLOv8s
  -> vẽ bbox/class/confidence
  -> tùy chọn Track / Density / Counting
  -> resize stream nếu chọn Stream width 960/1280
  -> nén JPEG theo JPEG quality
  -> gửi MJPEG lên browser
```

Object Detection luôn bật. Các option còn lại chỉ chạy khi bật trong giao diện:

```text
Track                    ByteTrack, track ID, motion trail
Traffic Density          Vẽ polygon 4 điểm cho từng vùng
Vehicle Counting         Vẽ line 2 điểm để đếm xe đi qua
Log CSV                  Ghi outputs/logs/<video>/<timestamp>/
Stream width             960 / 1280 / Original cho frame gửi lên browser
JPEG quality             65-90, mặc định 75
```

Nên demo bằng video 720p hoặc 1080p, 24-30 FPS, camera tĩnh hoặc gần tĩnh. Nếu stream giật, thử tắt `Log CSV`, chọn `Stream width = 960`, `JPEG quality = 70-75`.

## Batch Export Object Detection

Batch export chỉ chạy object detection, không chạy Track/Density/Counting. Dùng khi cần video minh họa và CSV cho báo cáo.

Một video:

```powershell
python app/detection.py `
  --source video/video2.mp4 `
  --model yolov8n-p2 `
  --device 0 `
  --conf 0.35 `
  --imgsz 640
```

Toàn bộ thư mục `video/`:

```powershell
python app/detection.py `
  --input-dir video `
  --output-root outputs/detection `
  --model yolov8n-p2 `
  --device 0
```

Mỗi video có output riêng:

```text
outputs/detection/<ten-video>/
  annotated.mp4
  detections.csv
  summary.json
  sample.jpg
```

## Kaggle Train Và Benchmark

Các workflow Kaggle nằm trong `kaggle/`, hướng dẫn chi tiết nằm trong `docs/`:

```text
docs/VISDRONE_RUN_GUIDE.md             Fine-tune YOLOv8n, YOLOv8n-P2, YOLOv8s
docs/VISDRONE_TEST_BENCHMARK_GUIDE.md  Benchmark 3 model và export W&B/report data
docs/KAGGLE_README.md                  Tóm tắt các file Kaggle
```

Sau khi train xong trên Kaggle, tải checkpoint về đúng vị trí:

```text
models/VisDrone/yolov8n/best.pt
models/VisDrone/yolov8n-p2/best.pt
models/VisDrone/yolov8s/best.pt
```

Các file số liệu tải từ W&B/Kaggle để viết báo cáo nên để vào:

```text
experiments/
```

Ví dụ:

```text
experiments/wandb_visdrone_export.zip
experiments/visdrone_test_dev_report_export.zip
experiments/visdrone_test_dev_comparison.csv
```

## Nên Đọc Docs Theo Thứ Tự

```text
docs/FRONTEND_ARCHITECTURE.md          Hiểu các file frontend/backend import nhau thế nào
docs/FRONTEND_GUIDE.md                 Cách chạy và dùng giao diện realtime
docs/LOCAL_DETECTION_EXPORT_GUIDE.md   Cách export annotated video/CSV
docs/MODELS_README.md                  Cách đặt checkpoint model local
docs/EXPERIMENTS_README.md             Cách lưu dữ liệu thực nghiệm tải từ W&B/Kaggle
docs/VISDRONE_RUN_GUIDE.md             Cách fine-tune trên Kaggle
docs/VISDRONE_TEST_BENCHMARK_GUIDE.md  Cách benchmark và tải số liệu từ W&B
```
