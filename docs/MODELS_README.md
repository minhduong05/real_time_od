# Model Storage

Repo dùng 3 checkpoint VisDrone đã fine-tune:

```text
models/VisDrone/yolov8n/best.pt
models/VisDrone/yolov8n-p2/best.pt
models/VisDrone/yolov8s/best.pt
```

Có thể đặt thêm `last.pt` cùng thư mục để lưu checkpoint cuối, nhưng app local mặc định chỉ đọc `best.pt`.

## Commit Checkpoint Nào?

Repo cho phép commit các checkpoint chính:

```text
models/VisDrone/yolov8n/best.pt
models/VisDrone/yolov8n-p2/best.pt
models/VisDrone/yolov8s/best.pt
```

Các checkpoint phụ như `last.pt` vẫn bị ignore để repo không phình thêm. Repo vẫn giữ `.gitkeep` trong từng folder model để cấu trúc thư mục không bị mất nếu chưa có checkpoint.

## Model Registry

Frontend và batch export không tự scan toàn bộ `models/`. Chúng đọc danh sách model từ:

```text
src/realtime_od/model_registry.py
```

Hiện có 3 key:

```text
yolov8n
yolov8n-p2
yolov8s
```

Chạy kiểm tra checkpoint:

```powershell
python scripts/verify_setup.py
```

Nếu một checkpoint thiếu, frontend vẫn hiện model đó nhưng đánh dấu missing/disabled.

## Dùng Model Trong Frontend

```powershell
python app/realtime_front.py
```

Sau đó chọn model trong dropdown.

## Dùng Model Khi Batch Export

```powershell
python app/detection.py --source video/video2.mp4 --model yolov8n-p2 --device 0
```
