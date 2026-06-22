# Model Storage

Repo này dùng 3 checkpoint VisDrone đã fine-tune:

```text
models/VisDrone/yolov8n/best.pt
models/VisDrone/yolov8n-p2/best.pt
models/VisDrone/yolov8s/best.pt
```

Các file `.pt` được ignore bởi git, nhưng vẫn cần tồn tại local để chạy frontend/batch export.

Chạy frontend:

```powershell
python app/realtime_front.py
```

Trong frontend có dropdown để chọn 1 trong 3 model.

Batch export một video với model cụ thể:

```powershell
python app/detection.py --source video/path_to_video.mp4 --model yolov8n-p2 --device 0
```
