# Model Storage

Checkpoint chính của hướng hiện tại:

```text
models/Top-View-Vehicle-Detection-Yolov8n-P2/
  best.pt
  last.pt
  run_info.yaml
```

Sau khi train trên Kaggle, tải ZIP export về rồi đặt các file vào đúng thư mục trên.

Các checkpoint `.pt` được ignore bởi git.

Ví dụ chạy local sau khi đã có `best.pt`:

```powershell
python app/realtime_front.py
```

Batch export object detection:

```powershell
python app/export_detection.py --source video/path_to_video.mp4 --device 0
```
