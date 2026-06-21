# Frontend Architecture

Tài liệu này giải thích các file liên quan đến frontend realtime và các script batch còn lại trong repo.

## Luồng chạy frontend realtime

Chạy lệnh:

```powershell
conda activate real_time_od
python app/realtime_front.py
```

Luồng import:

```text
app/realtime_front.py
  -> src/realtime_od/realtime_app.py
      -> src/realtime_od/config.py
      -> src/realtime_od/types.py
      -> models/Intersection-Flow-5K-Yolov8n-P2/best.pt
      -> video/*.mp4
```

## File chính của frontend

### `app/realtime_front.py`

File entrypoint để start web server Flask.

Nó chỉ làm các việc:

```text
thêm src/ vào PYTHONPATH
import create_app từ realtime_od.realtime_app
đọc host/port/debug từ command line
app.run(...)
```

Nó không xử lý frame, không chạy YOLO trực tiếp.

### `src/realtime_od/realtime_app.py`

Đây là file xử lý chính của frontend realtime.

Nó chứa:

```text
HTML/CSS/JavaScript của giao diện
Flask routes
load danh sách video trong video/
load frame đầu để người dùng vẽ vùng/line
nhận config từ frontend
chạy YOLOv8n-P2 predict/track theo từng frame
stream kết quả realtime về browser bằng MJPEG
logic Traffic Density Estimation
logic Vehicle Counting
logic Track overlay
```

Các route quan trọng:

```text
GET  /              giao diện browser
GET  /api/videos    danh sách video trong video/
GET  /api/frame     frame đầu của video để vẽ
GET  /api/video-info metadata video
POST /api/config    nhận lựa chọn option/vùng/line từ frontend
GET  /video_feed    stream realtime frame đã xử lý
```

Object Detection luôn chạy. Các option còn lại chỉ chạy khi bật trên giao diện:

```text
Traffic Density Estimation
Vehicle Counting
Track
```

## File được frontend realtime import

### `src/realtime_od/config.py`

Frontend dùng:

```python
PROJECT_ROOT
resolve_project_path
```

Mục đích:

```text
resolve đường dẫn video/model tương đối từ root project
tìm models/Intersection-Flow-5K-Yolov8n-P2/best.pt
tìm video/*.mp4
```

### `src/realtime_od/types.py`

Frontend realtime import class:

```python
Detection
```

Class này dùng làm cấu trúc dữ liệu chung cho một detection, có thể kèm `track_id` khi HTTP frontend bật tracking:

```text
frame_index
time_sec
track_id
class_id
class_name
confidence
xyxy
center
```

## File batch export object detection, không phải frontend realtime

### `app/export_detection.py`

Chạy một video bằng object detection only và xuất file:

```text
annotated.mp4
detections.csv
summary.json
sample.jpg
```

Nó dùng:

```text
src/realtime_od/detection_export.py
```

Không có tracking trong batch export này.

### `app/export_detections.py`

Chạy toàn bộ video trong thư mục `video/`.

Nó gọi lại:

```text
app/export_detection.py
```

Dùng khi muốn batch export object detection cho tất cả video.

### `src/realtime_od/detection_export.py`

Module xử lý batch object detection:

```text
cv2.VideoCapture
YOLO.predict
draw bbox/class/confidence
write annotated.mp4
write detections.csv
write summary.json
```

Module này không chạy ByteTrack.

## File inference cũ

### `app/run.py`

Script preview detection cũ bằng OpenCV window.

Luồng:

```text
app/run.py
  -> src/realtime_od/video.py
      -> src/realtime_od/detector.py
```

Nó không liên quan trực tiếp đến frontend realtime. Có thể giữ lại vì đơn giản, hữu ích để test nhanh webcam/video bằng detection thuần.

### `src/realtime_od/video.py`

Vòng lặp OpenCV cũ:

```text
cv2.VideoCapture
detector.predict_frame
cv2.imshow
```

Không được frontend realtime dùng.

### `src/realtime_od/detector.py`

Wrapper YOLO predict-frame đơn giản, dùng bởi `app/run.py`.

Không được frontend realtime dùng.

## File hướng dẫn

```text
FRONTEND_GUIDE.md          cách chạy frontend realtime
FRONTEND_ARCHITECTURE.md   file này, giải thích quan hệ các file
video/README.md            hướng dẫn với video local trong video/
LOCAL_DETECTION_EXPORT_GUIDE.md hướng dẫn batch object detection/export
README.md                  tổng quan repo
```

## Nên giữ file nào cho báo cáo?

Nên giữ:

```text
app/realtime_front.py
src/realtime_od/realtime_app.py
FRONTEND_GUIDE.md
FRONTEND_ARCHITECTURE.md
video/README.md
app/export_detection.py
app/export_detections.py
src/realtime_od/detection_export.py
src/realtime_od/types.py
LOCAL_DETECTION_EXPORT_GUIDE.md
kaggle/*.md
models/README.md
```

Lý do: frontend để demo realtime tracking/counting/density, batch scripts để xuất video/CSV object detection minh chứng, Kaggle guides để tái lập train/test.

Có thể bỏ sau này nếu muốn repo gọn hơn:

```text
app/run.py
src/realtime_od/video.py
src/realtime_od/detector.py
```

Nhưng hiện tại nên giữ vì chúng là baseline inference đơn giản và không ảnh hưởng frontend.
