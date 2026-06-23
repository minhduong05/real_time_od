# Realtime Frontend Guide

Frontend realtime dùng để xem trực tiếp kết quả xử lý video bằng 3 model VisDrone đã fine-tune.

## Chạy App

Từ root project:

```powershell
conda activate real_time_od
pip install -r requirements.txt
python app/realtime_front.py
```

Mở:

```text
http://127.0.0.1:7860
```

Nếu muốn đổi port:

```powershell
python app/realtime_front.py --host 127.0.0.1 --port 7861
```

## Chuẩn Bị Trước Khi Chạy

Đặt checkpoint:

```text
models/VisDrone/yolov8n/best.pt
models/VisDrone/yolov8n-p2/best.pt
models/VisDrone/yolov8s/best.pt
```

Đặt video vào:

```text
video/
```

App nhận các định dạng:

```text
.mp4, .avi, .mov, .mkv, .webm
```

## Pipeline

```text
OpenCV đọc frame gốc từ video
  -> YOLO xử lý frame gốc
  -> vẽ bbox/class/confidence
  -> chạy thêm Track/Density/Counting nếu được bật
  -> resize frame hiển thị nếu chọn Stream width 960/1280
  -> encode JPEG theo JPEG quality
  -> gửi MJPEG stream lên browser
```

Có hai kích thước cần phân biệt:

```text
Frame model xử lý: frame gốc của video
Frame browser nhận: frame sau khi resize/nén để stream
```

Vì vậy chọn `Stream width = 960` không làm YOLO detect kém hơn trong code hiện tại. Nó chỉ làm ảnh gửi lên browser nhẹ hơn.

## Các Bước Dùng Giao Diện

1. Chọn video trong dropdown.
2. Chọn model: `YOLOv8n`, `YOLOv8n-P2`, hoặc `YOLOv8s`.
3. Bấm `Load frame đầu` nếu muốn đổi video hoặc vẽ lại vùng.
4. Chọn option cần xử lý thêm.
5. Nếu bật `Traffic Density Estimation`, chọn số vùng, vẽ 4 điểm cho từng vùng rồi bấm `OK vùng`.
6. Nếu bật `Vehicle Counting`, vẽ 2 điểm cho line rồi bấm `OK thanh`.
7. Bấm `Start`.
8. Dùng `Stop/Continue` để dừng tạm frame hiện tại rồi chạy tiếp.
9. Dùng `Finish` để kết thúc stream và quay về frame đầu.

Object Detection luôn chạy. Các option khác chỉ chạy khi bật checkbox.

## Ý Nghĩa Option

```text
Traffic Density Estimation
  Đếm số phương tiện trong từng polygon 4 điểm.
  LOW: 0-3
  MED: 4-8
  HIGH: >= 9

Vehicle Counting
  Dùng ByteTrack nội bộ và line 2 điểm để đếm object đi qua line.
  Mỗi track ID chỉ được đếm một lần.

Track
  Bật ByteTrack, hiển thị track ID và trail.

Log CSV
  Ghi run_config.json, frame_log.csv, detections.csv vào outputs/logs/.
  Mặc định tắt để giảm I/O và tăng độ mượt.

Stream width
  Original: giữ nguyên kích thước video khi gửi lên browser.
  1280: giới hạn chiều rộng stream tối đa 1280.
  960: giới hạn chiều rộng stream tối đa 960 để nhẹ hơn.

JPEG quality
  Chất lượng nén JPEG của từng frame gửi lên browser.
  90 nét hơn nhưng nặng hơn.
  75 cân bằng.
  65 nhẹ hơn nhưng ảnh có thể xấu hơn.
```

## Output Log Realtime

Khi bật `Log CSV`, mỗi lần `Start` tạo folder:

```text
outputs/logs/<ten-video>/<timestamp>/
```

Bên trong:

```text
run_config.json   Config video/model/options/stream của lần chạy
frame_log.csv     Mỗi frame: thời gian, FPS, số detection, active counts
detections.csv    Mỗi detection: class, confidence, bbox, center, track_id
```

Nếu đang test FPS, nên tắt `Log CSV`.

## Video Nên Dùng

Khuyến nghị cho demo local:

```text
Resolution: 720p hoặc 1080p
FPS: 24-30
Độ dài: 20-90 giây khi demo
Camera: tĩnh hoặc ít rung
Góc nhìn: trên cao, highway, đường phố, giao lộ
Ánh sáng: ban ngày hoặc đủ sáng
```

Gợi ý setting:

```text
Mượt nhất:       720p, Stream width 960, JPEG quality 70-75, Log CSV off
Cân bằng:        720p/1080p, Stream width 1280, JPEG quality 75, Log CSV off
Nét để quan sát: Original hoặc 1280, JPEG quality 85-90
```

Tránh dùng trực tiếp video 4K cho realtime local. Nếu có video 4K, nên tải/cắt bản 720p hoặc 1080p.

