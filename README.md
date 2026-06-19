# YOLOv8-P2 Small Vehicle Detection and Traffic Monitoring

This project studies YOLOv8-P2 for small vehicle detection and applies it to real-time traffic surveillance.

## Research Track

- Dataset: VisDrone
- Models: YOLOv8n, YOLOv8s, YOLOv8-P2
- Metrics: Precision, Recall, mAP50, mAP50-95, FPS

## Application Track

- Dataset: Intersection-Flow-5K and real traffic videos
- Models: YOLOv8s, YOLOv8-P2
- Features: vehicle detection, vehicle counting, traffic density estimation, congestion warning

## Ultralytics Reference

The archive `ultralytics-main.zip` is kept at the repository root for reference.
Useful files inside it:

- `ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml`
- `ultralytics-main/ultralytics/cfg/models/v8/yolov8.yaml`
- `ultralytics-main/ultralytics/cfg/models/v8/yolov8-p2.yaml`

## Pretrained Models

YOLOv8n and YOLOv8s use official pretrained weights. YOLOv8-P2 does not have an
official pretrained checkpoint, so this project initializes YOLOv8s-P2 from the
P2 architecture and partially transfers compatible weights from YOLOv8s.

Prepare all initial weights:

```bash
python scripts/prepare_models.py --check-p2 --save-p2 weights/pretrained/yolov8s-p2-init.pt
```

Expected local files:

- `weights/pretrained/yolov8n.pt`
- `weights/pretrained/yolov8s.pt`
- `weights/pretrained/yolov8s-p2-init.pt`

## Training

VisDrone research experiments:

```bash
python scripts/train_visdrone.py --config configs/experiments/visdrone_yolov8n.yaml
python scripts/train_visdrone.py --config configs/experiments/visdrone_yolov8s.yaml
python scripts/train_visdrone.py --config configs/experiments/visdrone_yolov8p2.yaml
```

Traffic application experiments:

```bash
python scripts/train_traffic.py --config configs/experiments/traffic_yolov8s.yaml
python scripts/train_traffic.py --config configs/experiments/traffic_yolov8p2.yaml
```

Use `--dry-run` to check paths without starting training.
