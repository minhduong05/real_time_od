# Kaggle Training

This folder is for Kaggle only. Use it to train, tune, and export checkpoints.

For the full VisDrone + W&B notebook workflows, see:

```text
kaggle/VISDRONE_RUN_GUIDE.md
kaggle/VISDRONE_TEST_BENCHMARK_GUIDE.md
kaggle/INTERSECTION_FLOW_5K_YOLOV8N_P2_TRAIN_GUIDE.md
```

The Intersection-Flow-5K guide is the primary workflow for the traffic
monitoring application. It trains the project YOLOv8n-P2 architecture with
transfer learning, logs train/validation to a dedicated W&B project, and
exports the final checkpoint.

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
