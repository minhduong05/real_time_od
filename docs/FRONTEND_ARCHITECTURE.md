# Frontend Architecture

Tài liệu này giải thích cấu trúc frontend realtime hiện tại sau khi đã tách module.

## Luồng chạy

```powershell
conda activate real_time_od
python app/realtime_front.py
```

Luồng import chính:

```text
app/realtime_front.py
  -> src/realtime_od/realtime_app.py
      -> src/realtime_od/realtime_template.py
      -> src/realtime_od/realtime_state.py
      -> src/realtime_od/realtime_stream.py
          -> src/realtime_od/realtime_draw.py
          -> src/realtime_od/types.py
      -> src/realtime_od/model_registry.py
      -> src/realtime_od/config.py
```

Checkpoint được quản lý tập trung ở:

```text
src/realtime_od/model_registry.py
```

với 3 model:

```text
models/VisDrone/yolov8n/best.pt
models/VisDrone/yolov8n-p2/best.pt
models/VisDrone/yolov8s/best.pt
```

## Frontend Realtime

### `app/realtime_front.py`

Entrypoint để start Flask server. File này chỉ thêm `src/` vào `PYTHONPATH`, import `create_app`, đọc `host/port/debug`, rồi gọi `app.run(...)`.

### `src/realtime_od/realtime_app.py`

Flask app factory và route definitions.

Nó chứa:

```text
GET  /              render giao diện
GET  /api/videos    danh sách video trong video/
GET  /api/models    danh sách 3 model VisDrone
GET  /api/frame     frame đầu để vẽ vùng/line
GET  /api/video-info metadata video
POST /api/config    nhận lựa chọn từ frontend
GET  /video_feed    MJPEG stream
```

File này không xử lý YOLO trực tiếp.

### `src/realtime_od/realtime_template.py`

Chứa HTML/CSS/JavaScript của giao diện.

Giao diện cho phép:

```text
chọn video
chọn model
bật/tắt Traffic Density Estimation
bật/tắt Vehicle Counting
bật/tắt Track
vẽ polygon 4 điểm
vẽ counting line 2 điểm
```

### `src/realtime_od/realtime_state.py`

Quản lý runtime state và cache model.

```text
RuntimeConfig: video, model_key, options, zones, line, conf
RealtimeState: lưu config hiện tại và cache YOLO model theo model_key
```

### `src/realtime_od/realtime_stream.py`

Xử lý luồng video realtime:

```text
đọc frame bằng OpenCV
chọn model từ RealtimeState
YOLO.predict hoặc YOLO.track
convert result thành Detection
gọi realtime_draw để vẽ overlay
encode JPG
yield MJPEG frames về browser
```

### `src/realtime_od/realtime_draw.py`

Chứa toàn bộ logic vẽ và analytics realtime:

```text
bounding box/class/confidence
track ID và trail
density zone
counting line
active count overlay
```

### `src/realtime_od/model_registry.py`

Nguồn cấu hình duy nhất cho 3 model VisDrone:

```text
key
label
weights
description
```

Frontend và batch export đều dùng file này.

## Batch Export

### `app/detection.py`

CLI batch export Object Detection. File này chạy được cả một video hoặc toàn bộ folder video.

```text
annotated.mp4
detections.csv
summary.json
sample.jpg
```

Có thể chọn model bằng:

```powershell
python app/detection.py --source video/path.mp4 --model yolov8n-p2 --device 0
```

Chạy toàn bộ thư mục `video/`:

```powershell
python app/detection.py --input-dir video --model yolov8n-p2 --device 0
```

### `src/realtime_od/detection_export.py`

Module xử lý batch object detection. Module này không chạy ByteTrack, không chạy counting/density realtime.

## File Còn Lại

```text
src/realtime_od/config.py       resolve project path, đọc config YAML
src/realtime_od/types.py        dataclass Detection dùng chung
configs/project.yaml            class/model/dataset config
configs/models/yolov8n-p2.yaml  kiến trúc YOLOv8n-P2
```

## File Đã Bỏ

Các file OpenCV-window inference cũ đã được xóa vì không còn dùng trong hướng hiện tại:

```text
app/run.py
src/realtime_od/video.py
src/realtime_od/detector.py
```
