# Real-Time Object Detection With VisDrone

Project này tập trung vào phân tích lý thuyết, thực nghiệm và demo realtime cho 3 model đã fine-tune trên VisDrone:

```text
models/VisDrone/yolov8n/best.pt
models/VisDrone/yolov8n-p2/best.pt
models/VisDrone/yolov8s/best.pt
```

VisDrone có 10 class:

```text
pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor
```

## Layout

```text
app/                         # entrypoint frontend và batch export
configs/                     # cấu hình project/model
kaggle/                      # guide train/benchmark VisDrone
models/VisDrone/             # 3 checkpoint fine-tuned
scripts/                     # kiểm tra setup local
src/realtime_od/             # backend realtime và inference utils
video/                       # video local để demo
```

## Frontend Realtime

```powershell
conda activate real_time_od
python app/realtime_front.py
```

Mở:

```text
http://127.0.0.1:7860
```

Frontend cho phép:

```text
chọn video trong thư mục video/
chọn 1 trong 3 model VisDrone
Object Detection luôn bật
bật/tắt Track bằng ByteTrack
vẽ vùng Traffic Density Estimation
vẽ line Vehicle Counting
```

## Batch Export

Batch export chỉ dùng cho Object Detection để lấy video/CSV phục vụ báo cáo:

```powershell
python app/detection.py `
  --source "video/path_to_video.mp4" `
  --model yolov8n-p2 `
  --device 0
```

Output:

```text
outputs/detection/<video-name>/
  annotated.mp4
  detections.csv
  summary.json
  sample.jpg
```

## Kaggle

Các file quan trọng:

```text
docs/VISDRONE_RUN_GUIDE.md             train/fine-tune 3 model VisDrone
docs/VISDRONE_TEST_BENCHMARK_GUIDE.md  benchmark 3 model trên test-dev
kaggle/train.py                          script train dùng trên Kaggle
```

## Local Check

```powershell
conda activate real_time_od
python scripts/verify_setup.py
```
