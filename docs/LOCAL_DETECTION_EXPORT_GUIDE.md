# Local Detection Export Guide

Guide này dùng để xuất kết quả **Object Detection only** cho video local. Đây là workflow phục vụ báo cáo: tạo video đã vẽ bbox, CSV detection theo frame, summary JSON và sample image.

Tracking, Traffic Density và Vehicle Counting không chạy trong batch export. Các phần đó chỉ chạy trong frontend realtime.

## Chạy Một Video

```powershell
conda activate real_time_od
pip install -r requirements.txt

python app/detection.py `
  --source video/video2.mp4 `
  --model yolov8n-p2 `
  --device 0 `
  --conf 0.2 `
  --iou 0.7 `
  --imgsz 640 `
  --max-box-area-ratio 0.12
```

Chọn model bằng một trong ba key:

```text
yolov8n
yolov8n-p2
yolov8s
```

## Chạy Toàn Bộ Thư Mục Video

```powershell
python app/detection.py `
  --input-dir video `
  --output-root outputs/detection `
  --model yolov8n-p2 `
  --device 0 `
  --conf 0.2 `
  --iou 0.7 `
  --imgsz 640 `
  --max-box-area-ratio 0.12
```

## Output

Mỗi video sẽ có một folder riêng:

```text
outputs/detection/<ten-video>/
  annotated.mp4
  detections.csv
  summary.json
  sample.jpg
```

Ý nghĩa:

```text
annotated.mp4   Video gốc đã vẽ bbox/class/confidence
detections.csv  Mỗi dòng là một detection trong một frame
summary.json    Tổng số frame, FPS, duration, số detection theo class
sample.jpg      Một frame mẫu lấy từ annotated.mp4
```

`detections.csv` có format:

```text
frame,time_sec,class_id,class_name,confidence,x1,y1,x2,y2,center_x,center_y
```

Không có `track_id` vì đây là export detection-only.

## Khi Nào Dùng Batch Export?

Dùng khi cần:

```text
video minh họa object detection
CSV detection theo từng frame
summary số lượng detection theo class
sample frame đưa vào báo cáo
```

Không dùng cho:

```text
Track ID
motion trail
Vehicle Counting
Traffic Density Estimation
```

Các chức năng đó nằm trong frontend:

```powershell
python app/realtime_front.py
```

