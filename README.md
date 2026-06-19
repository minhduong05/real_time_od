# YOLOv8-P2 Small Vehicle Detection and Traffic Monitoring

This project studies YOLOv8-P2 for small vehicle detection and applies it to real-time traffic surveillance.

## Research Track

- Dataset: VisDrone
- Models: YOLOv8n, YOLOv8n-P2, YOLOv8s
- Metrics: Precision, Recall, mAP50, mAP50-95, FPS

## Application Track

- Dataset: Intersection-Flow-5K and real traffic videos
- Models: YOLOv8n-P2 and YOLOv8s
- Features: vehicle detection, vehicle counting, traffic density estimation, congestion warning

## Dataset Preparation

The Kaggle workflow uses datasets attached under `/kaggle/input`, so the
repository does not store dataset folders. For local experiments only, VisDrone
can be downloaded directly from Ultralytics assets and converted to YOLO format:

```bash
python scripts/prepare_visdrone.py
```

Intersection-Flow-5K is hosted on Kaggle as `starsw/intersection-flow-5k`.
After configuring Kaggle credentials, run:

```bash
python scripts/prepare_intersection_flow.py --kaggle
```

If the dataset is downloaded manually, place the zip or extracted folder under
`data/raw/Intersection-Flow-5K`, then run:

```bash
python scripts/prepare_intersection_flow.py
```

## Pretrained Models

YOLOv8n and YOLOv8s use official pretrained weights. YOLOv8n-P2 does not have
an official pretrained checkpoint, so this project initializes YOLOv8n-P2 from
the P2 architecture and partially transfers compatible weights from YOLOv8n.

Prepare all initial weights:

```bash
python scripts/prepare_models.py --check-p2 --save-p2 weights/pretrained/yolov8n-p2-init.pt
```

Expected local files:

- `weights/pretrained/yolov8n.pt`
- `weights/pretrained/yolov8s.pt`
- `weights/pretrained/yolov8n-p2-init.pt`

## Training

The preferred workflow is to fine-tune on Kaggle. See:

```text
docs/kaggle_workflow.md
```

For Kaggle notebooks that attach this GitHub repository and a VisDrone dataset,
use the Kaggle entrypoint. It auto-discovers `visdrone.yaml` under
`/kaggle/input`, writes outputs under `/kaggle/working`, and can enable W&B:

```bash
python scripts/kaggle_train_visdrone.py \
  --models yolov8n \
  --project /kaggle/working/experiments/visdrone \
  --epochs 20 \
  --batch 8 \
  --workers 2 \
  --device 0 \
  --wandb-project real-time-od-visdrone
```

VisDrone research experiments:

```bash
python scripts/train_visdrone_research.py \
  --data /kaggle/input/visdrone-yolo/visdrone.yaml \
  --project /kaggle/working/experiments/visdrone
```

Or run each model separately:

```bash
python scripts/train_visdrone.py --config configs/experiments/visdrone_yolov8n.yaml
python scripts/train_visdrone.py --config configs/experiments/visdrone_yolov8n_p2.yaml
python scripts/train_visdrone.py --config configs/experiments/visdrone_yolov8s.yaml
```

Traffic application experiments:

```bash
python scripts/train_traffic.py --config configs/experiments/traffic_yolov8s.yaml
python scripts/train_traffic.py --config configs/experiments/traffic_yolov8n_p2.yaml
```

Use `--dry-run` to check paths without starting training.

On Kaggle, override dataset and output paths:

```bash
python scripts/train_visdrone.py \
  --config configs/experiments/visdrone_yolov8n_p2.yaml \
  --data /kaggle/input/visdrone-yolo/visdrone.yaml \
  --project /kaggle/working/experiments/visdrone
```
