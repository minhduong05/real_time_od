# Real-Time Object Detection

This repo is split into two clean workflows:

- Kaggle: train, tune, and export checkpoints with GPU.
- Local: run inference/app with a checkpoint downloaded from Kaggle.

No dataset or `.pt` checkpoint is committed to git.

## Layout

```text
app/                         # local inference CLI
configs/                     # shared project/model/dataset settings
kaggle/                      # Kaggle-only train/tune/export script
models/
  VisDrone/                  # downloaded VisDrone checkpoints
  Top-View-Vehicle-Detection-Yolov8n-P2/
                              # downloaded Top-View Vehicle YOLOv8n-P2 checkpoint
scripts/                     # local utility checks
src/realtime_od/             # local inference package
```

Use this checkpoint naming convention for new exports:

```text
models/<Dataset>/<Model>/best.pt
models/<Dataset>/<Model>/last.pt
```

The current target fine-tuned Top-View Vehicle YOLOv8n-P2 model should be stored at:

```text
models/Top-View-Vehicle-Detection-Yolov8n-P2/best.pt
models/Top-View-Vehicle-Detection-Yolov8n-P2/last.pt
models/Top-View-Vehicle-Detection-Yolov8n-P2/run_info.yaml
```

Examples:

```text
models/VisDrone/yolov8n/best.pt
models/VisDrone/yolov8n-p2/best.pt
models/Top-View-Vehicle-Detection-Yolov8n-P2/best.pt
```

## Kaggle Train

Install dependencies:

```bash
pip install -r kaggle/requirements.txt
```

Train YOLOv8n-P2 on Top-View Vehicle Detection:

```bash
python kaggle/train.py \
  --dataset-name Top-View-Vehicle-Detection \
  --data /kaggle/working/top_view_vehicle.yaml \
  --model yolov8n-p2 \
  --epochs 100 \
  --imgsz 640 \
  --batch 32 \
  --device 0
```

The export will be written to:

```text
/kaggle/working/export/Top-View-Vehicle-Detection/yolov8n-p2/best.pt
/kaggle/working/export/Top-View-Vehicle-Detection/yolov8n-p2/last.pt
/kaggle/working/export/Top-View-Vehicle-Detection/yolov8n-p2/run_info.yaml
```

Zip it on Kaggle:

```bash
zip -r /kaggle/working/Top-View-Vehicle-Detection_yolov8n-p2_export.zip /kaggle/working/export/Top-View-Vehicle-Detection/yolov8n-p2
```

Download the zip, then place the files under:

```text
models/Top-View-Vehicle-Detection-Yolov8n-P2/
```

## Local Inference

Install local dependencies:

```bash
pip install -r requirements.txt
```

Run with an explicit checkpoint:

```bash
python app/run.py --weights models/Top-View-Vehicle-Detection-Yolov8n-P2/best.pt --source 0
```

Run a video file:

```bash
python app/run.py --weights models/Top-View-Vehicle-Detection-Yolov8n-P2/best.pt --source path/to/video.mp4
```

## Traffic App

Realtime browser UI:

```bash
python app/realtime_front.py
```

Open:

```text
http://127.0.0.1:7860
```

Export object detection for a video with the fine-tuned Top-View Vehicle YOLOv8n-P2 checkpoint:

```bash
python app/export_detection.py --source video/path_to_video.mp4 --device 0
```

The batch export is object-detection-only. By default, each source video gets its own output folder:

```text
outputs/detection/<video-name>/
  annotated.mp4
  detections.csv
  summary.json
  sample.jpg
```

The annotated video includes bounding boxes, class names, confidence scores, and active object counts by class. Tracking is handled only in the realtime HTTP frontend.

The CSV contains one row per detection per frame:

```text
frame,time_sec,class_id,class_name,confidence,x1,y1,x2,y2,center_x,center_y
```

For the current local demo video and recommended settings, see:

```text
LOCAL_DETECTION_EXPORT_GUIDE.md
FRONTEND_GUIDE.md
FRONTEND_ARCHITECTURE.md
video/README.md
```

Check setup:

```bash
python scripts/verify_setup.py
```
