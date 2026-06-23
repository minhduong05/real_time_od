# Frontend Architecture

Tài liệu này giải thích các file của frontend realtime và backend local. Luồng chính là Flask trả giao diện HTML, JavaScript gửi config, backend đọc video từng frame rồi trả MJPEG stream.

## Import Graph

```text
app/realtime_front.py
  -> src/realtime_od/realtime_app.py
      -> src/realtime_od/realtime_template.py
      -> src/realtime_od/realtime_state.py
      -> src/realtime_od/realtime_stream.py
          -> src/realtime_od/realtime_draw.py
          -> src/realtime_od/realtime_logger.py
          -> src/realtime_od/types.py
      -> src/realtime_od/model_registry.py
      -> src/realtime_od/config.py
```

`app/realtime_front.py` chỉ là entrypoint. Phần xử lý chính nằm trong `src/realtime_od/`.

## Luồng Realtime

```text
Browser
  -> GET /
  -> chọn video/model/options
  -> POST /api/config
  -> GET /video_feed

Backend
  -> OpenCV đọc frame gốc
  -> YOLO.predict hoặc YOLO.track
  -> extract Detection dataclass
  -> vẽ overlay bằng realtime_draw.py
  -> ghi CSV nếu bật Log CSV
  -> resize frame stream nếu cần
  -> encode JPEG
  -> yield MJPEG frame về browser
```

## Các Route Flask

```text
GET  /               Render giao diện
GET  /api/videos     Liệt kê video trong video/
GET  /api/models     Liệt kê 3 model VisDrone và trạng thái checkpoint
GET  /api/video-info Metadata video: width, height, fps, frames, duration
GET  /api/frame      Frame đầu để vẽ density zone/counting line
POST /api/config     Nhận config khi bấm Start
POST /api/playback   Pause/resume/finish stream đang chạy
GET  /video_feed     MJPEG stream sau xử lý
```

## Vai Trò Từng File

### `app/realtime_front.py`

Entry point chạy server:

```powershell
python app/realtime_front.py
```

File này thêm `src/` vào `sys.path`, import `create_app`, đọc `--host`, `--port`, `--debug`, rồi gọi `app.run(...)`.

### `src/realtime_od/realtime_app.py`

Tạo Flask app và khai báo API route. File này không chạy YOLO trực tiếp, mà gọi:

```text
RealtimeState                  lưu config và model cache
read_video_info                đọc metadata video
process_stream                 sinh MJPEG stream
INDEX_HTML                     template frontend
MODEL_REGISTRY                 danh sách 3 model
```

### `src/realtime_od/realtime_template.py`

Chứa HTML/CSS/JavaScript của giao diện. Các việc chính:

```text
load danh sách video và model
load frame đầu để vẽ vùng/line
vẽ density polygon 4 điểm
vẽ counting line 2 điểm
gửi config khi bấm Start
gửi pause/resume/finish
hiển thị stream MJPEG
```

Đây là frontend duy nhất của app hiện tại.

### `src/realtime_od/realtime_state.py`

Quản lý state dùng chung giữa các request:

```text
RuntimeConfig
  video, model_key, options
  density_zones, count_line
  conf, max_box_area_ratio
  enable_logging
  stream_width, jpeg_quality

PlaybackControl
  paused, finish_requested

RealtimeState
  config hiện tại
  cache YOLO model theo model_key
```

Model được cache để không phải load lại checkpoint sau mỗi request.

### `src/realtime_od/realtime_stream.py`

File xử lý realtime nặng nhất:

```text
OpenCV VideoCapture
YOLO.predict nếu không cần tracking
YOLO.track + ByteTrack nếu bật Track hoặc Counting
extract bbox/class/confidence/track_id
gọi realtime_draw.py để vẽ
gọi realtime_logger.py nếu bật Log CSV
tính FPS hiện tại / FPS gốc
resize stream frame theo Stream width
encode JPEG theo JPEG quality
yield multipart MJPEG
```

Lưu ý: `Stream width` chỉ resize frame sau khi đã xử lý và vẽ overlay. YOLO vẫn chạy trên frame gốc của video.

### `src/realtime_od/realtime_draw.py`

Chứa toàn bộ logic vẽ:

```text
bounding box
class + confidence
track ID + trail
density zone LOW/MED/HIGH
counting line và count theo class
overlay model/frame/FPS/options/active counts
```

`Traffic Density` và `Vehicle Counting` chỉ tính nhóm class phương tiện trong `VEHICLE_CLASSES`.

### `src/realtime_od/realtime_logger.py`

Chỉ chạy khi bật `Log CSV` trên frontend. Output:

```text
outputs/logs/<ten-video>/<timestamp>/
  run_config.json
  frame_log.csv
  detections.csv
```

Tắt `Log CSV` sẽ giảm I/O và thường giúp stream mượt hơn.

### `src/realtime_od/model_registry.py`

Nguồn cấu hình duy nhất cho 3 checkpoint dùng trong app:

```text
yolov8n      -> models/VisDrone/yolov8n/best.pt
yolov8n-p2   -> models/VisDrone/yolov8n-p2/best.pt
yolov8s      -> models/VisDrone/yolov8s/best.pt
```

Frontend và batch export đều dùng file này.

## Batch Export Liên Quan Thế Nào?

Batch export không dùng `realtime_app.py`, `realtime_template.py` hay `realtime_stream.py`.

Luồng batch export:

```text
app/detection.py
  -> src/realtime_od/detection_export.py
      -> src/realtime_od/types.py
      -> src/realtime_od/config.py
```

Batch export chỉ tạo object detection output:

```text
annotated.mp4
detections.csv
summary.json
sample.jpg
```

Tracking, density và counting chỉ nằm trong frontend realtime.

