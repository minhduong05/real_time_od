# Kaggle Workflow

This project is designed to keep code in the local repository and run heavy
fine-tuning on Kaggle.

## 1. Prepare Kaggle Notebook

Create a Kaggle Notebook and enable GPU:

- Accelerator: GPU T4 x2 or P100
- Internet: On, at least for the first run if pretrained weights are not attached

Add this repository as code by either:

- uploading the project folder as a Kaggle Dataset, or
- cloning/pulling the repository inside the notebook.

## 2. Attach Datasets

### Intersection-Flow-5K

Use Kaggle `Add Input` and attach:

```text
starsw/intersection-flow-5k
```

Find the dataset YAML or root path in the notebook:

```python
import os
for root, dirs, files in os.walk("/kaggle/input/intersection-flow-5k"):
    if "intersection.yaml" in files:
        print(os.path.join(root, "intersection.yaml"))
```

Use the printed path as `--data`.

### VisDrone

Preferred options:

1. Attach an existing VisDrone YOLO-format Kaggle Dataset.
2. Or run `scripts/prepare_visdrone.py` once in Kaggle and save the converted
   output as your own Kaggle Dataset for reuse.

The expected YOLO structure is:

```text
dataset_root/
+-- images/
|   +-- train/
|   +-- val/
|   +-- test/
+-- labels/
|   +-- train/
|   +-- val/
|   +-- test/
+-- visdrone.yaml
```

## 3. Install Dependencies

Inside Kaggle:

```bash
pip install -r requirements.txt
```

If Kaggle already has compatible `ultralytics` and `ultralytics-thop`, this can
be skipped or limited to missing packages.

## 4. Prepare Weights

Create YOLOv8n, YOLOv8s, and YOLOv8s-P2 initial weights:

```bash
python scripts/prepare_models.py --check-p2 --save-p2 weights/pretrained/yolov8s-p2-init.pt
```

YOLOv8-P2 has no official pretrained checkpoint. The project builds YOLOv8s-P2
from the architecture config and transfers compatible weights from YOLOv8s.

## 5. Train VisDrone Research Experiments

Replace `/kaggle/input/visdrone-yolo/visdrone.yaml` with the actual path.

One-command version:

```bash
python scripts/train_visdrone_research.py \
  --data /kaggle/input/visdrone-yolo/visdrone.yaml \
  --project /kaggle/working/experiments/visdrone \
  --device 0
```

Equivalent separate commands:

```bash
python scripts/train_visdrone.py \
  --config configs/experiments/visdrone_yolov8n.yaml \
  --data /kaggle/input/visdrone-yolo/visdrone.yaml \
  --project /kaggle/working/experiments/visdrone \
  --device 0

python scripts/train_visdrone.py \
  --config configs/experiments/visdrone_yolov8s.yaml \
  --data /kaggle/input/visdrone-yolo/visdrone.yaml \
  --project /kaggle/working/experiments/visdrone \
  --device 0

python scripts/train_visdrone.py \
  --config configs/experiments/visdrone_yolov8p2.yaml \
  --data /kaggle/input/visdrone-yolo/visdrone.yaml \
  --project /kaggle/working/experiments/visdrone \
  --device 0
```

## 6. Train Traffic Experiments

Replace `/kaggle/input/intersection-flow-5k/Intersection-Flow-5K/intersection.yaml`
with the actual path printed in step 2.

```bash
python scripts/train_traffic.py \
  --config configs/experiments/traffic_yolov8s.yaml \
  --data /kaggle/input/intersection-flow-5k/Intersection-Flow-5K/intersection.yaml \
  --project /kaggle/working/experiments/traffic \
  --device 0

python scripts/train_traffic.py \
  --config configs/experiments/traffic_yolov8p2.yaml \
  --data /kaggle/input/intersection-flow-5k/Intersection-Flow-5K/intersection.yaml \
  --project /kaggle/working/experiments/traffic \
  --device 0
```

## 7. Save Outputs

After training, download or commit these Kaggle outputs:

```text
/kaggle/working/experiments/
/kaggle/working/weights/
```

Important files for the report:

- `results.csv`
- `weights/best.pt`
- `confusion_matrix.png`
- `PR_curve.png`
- `F1_curve.png`
- `val_batch*_pred.jpg`
