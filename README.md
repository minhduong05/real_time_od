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
  Intersection-Flow-5K/       # downloaded Intersection-Flow-5K checkpoints
scripts/                     # local utility checks
src/realtime_od/             # local inference package
```

Use this checkpoint naming convention:

```text
models/<Dataset>/<Model>/best.pt
models/<Dataset>/<Model>/last.pt
```

Examples:

```text
models/VisDrone/yolov8n/best.pt
models/VisDrone/yolov8n-p2/best.pt
models/Intersection-Flow-5K/yolov8n/best.pt
```

## Kaggle Train

Install dependencies:

```bash
pip install -r kaggle/requirements.txt
```

Train YOLOv8n-P2 on VisDrone:

```bash
python kaggle/train.py \
  --dataset-name VisDrone \
  --data /kaggle/working/visdrone.yaml \
  --model yolov8n-p2 \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0,1
```

The export will be written to:

```text
/kaggle/working/export/VisDrone/yolov8n-p2/best.pt
/kaggle/working/export/VisDrone/yolov8n-p2/last.pt
/kaggle/working/export/VisDrone/yolov8n-p2/run_info.yaml
```

Zip it on Kaggle:

```bash
zip -r /kaggle/working/VisDrone_yolov8n-p2_export.zip /kaggle/working/export/VisDrone/yolov8n-p2
```

Download the zip, then place the files under:

```text
models/VisDrone/yolov8n-p2/
```

## Local Inference

Install local dependencies:

```bash
pip install -r requirements.txt
```

Run with an explicit checkpoint:

```bash
python app/run.py --weights models/VisDrone/yolov8n-p2/best.pt --source 0
```

Run a video file:

```bash
python app/run.py --weights models/VisDrone/yolov8n-p2/best.pt --source path/to/video.mp4
```

Check setup:

```bash
python scripts/verify_setup.py
```
