# Local Detection Export Guide

Guide này dùng để xuất kết quả **Object Detection only** cho video local.

Tracking, counting line và density realtime được xử lý trong HTTP frontend, không chạy trong batch export này.

## Chạy một video

```powershell
conda activate real_time_od
pip install -r requirements.txt

python app/export_detection.py `
  --source "video/YTSave_YouTube_4K-camera-example-for-Traffic-Monitoring_Media_cJatWBDNabE_001_720p.mp4" `
  --device 0 `
  --conf 0.35 `
  --iou 0.7 `
  --imgsz 640 `
  --max-box-area-ratio 0.12
```

Output mặc định:

```text
outputs/detection/<ten-video>/
  annotated.mp4
  detections.csv
  summary.json
  sample.jpg
```

`detections.csv` có format:

```text
frame,time_sec,class_id,class_name,confidence,x1,y1,x2,y2,center_x,center_y
```

Không có `track_id` trong CSV này vì đây là export detection-only.

## Chạy tất cả video

```powershell
python app/export_detections.py `
  --input-dir video `
  --output-root outputs/detection `
  --device 0 `
  --conf 0.35 `
  --iou 0.7 `
  --imgsz 640 `
  --max-box-area-ratio 0.12
```

## Khi nào dùng batch export này?

Dùng khi cần:

```text
video minh họa object detection
CSV detection theo từng frame
summary số lượng detection theo class
sample frame để đưa vào báo cáo
```

Không dùng batch export này cho:

```text
Track ID
motion trail
Vehicle Counting
Traffic Density Estimation realtime
```

Các chức năng đó nằm trong frontend:

```powershell
python app/realtime_front.py
```
