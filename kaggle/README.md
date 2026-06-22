# Kaggle Training

This folder is for Kaggle only. Use it to train, tune, and export checkpoints.

For the current Top-View Vehicle + W&B notebook workflow, see:

```text
kaggle/TOP_VIEW_VEHICLE_YOLOV8N_P2_TRAIN_GUIDE.md
```

Legacy VisDrone workflows are still kept for reference:

```text
kaggle/VISDRONE_RUN_GUIDE.md
kaggle/VISDRONE_TEST_BENCHMARK_GUIDE.md
```

The Top-View Vehicle guide is the primary workflow for the highway traffic
monitoring application. It trains the project YOLOv8n-P2 architecture with
transfer learning, logs train/validation to W&B, and exports the final checkpoint.

## Train YOLOv8n-P2 on Top-View Vehicle Detection

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

## Train YOLOv8n-P2 on VisDrone

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

## Train baseline YOLOv8n on VisDrone

```bash
python kaggle/train.py \
  --dataset-name VisDrone \
  --data /kaggle/working/visdrone.yaml \
  --model yolov8n \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0,1
```

## Tune Hyperparameters

```bash
python kaggle/train.py \
  --dataset-name VisDrone \
  --data /kaggle/working/visdrone.yaml \
  --model yolov8n-p2 \
  --tune \
  --iterations 30
```

Export layout:

```text
/kaggle/working/export/<Dataset>/<Model>/best.pt
/kaggle/working/export/<Dataset>/<Model>/last.pt
/kaggle/working/export/<Dataset>/<Model>/run_info.yaml
```

Example:

```text
/kaggle/working/export/VisDrone/yolov8n-p2/best.pt
```
