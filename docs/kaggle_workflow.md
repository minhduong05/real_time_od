# Kaggle Workflow

This repository should stay code-first: keep source code, configs, scripts, and
documentation in GitHub. Keep datasets, training logs, and checkpoints in
Kaggle outputs and W&B.

## 1. Push the Repository

Before opening Kaggle, push the cleaned repository to GitHub:

```bash
git status
git add .
git commit -m "Add Kaggle VisDrone training workflow"
git push
```

The repository intentionally ignores heavy runtime outputs such as `.pt`,
`wandb/`, `runs/`, `experiments/`, and generated result tables/figures.

## 2. Create the Kaggle Notebook

In Kaggle:

- Create or open a notebook.
- Enable `Accelerator: GPU`.
- Enable `Internet: On` for package install, GitHub clone, pretrained weights,
  and W&B sync.
- Add the VisDrone dataset with `Add Input`.

If you use W&B, add a Kaggle secret:

```text
Name: WANDB_API_KEY
Value: your W&B API key
```

Get the key from your W&B account settings or authorization page.

## 3. Link GitHub Code

Option A: use Kaggle's GitHub integration if available.

Option B: clone manually in the first notebook cell:

```bash
%cd /kaggle/working
!git clone https://github.com/<your-user>/<your-repo>.git real_time_od
%cd /kaggle/working/real_time_od
```

If you attached the repository as a Kaggle Dataset instead, copy only the code
into `/kaggle/working` so scripts can write outputs:

```bash
!mkdir -p /kaggle/working/real_time_od
!cp -r /kaggle/input/<repo-dataset-folder>/real_time_od/. /kaggle/working/real_time_od/
%cd /kaggle/working/real_time_od
```

The trailing `/.` matters: it copies the contents of the repository folder
instead of creating a nested `real_time_od/real_time_od` path.

## 4. Install Dependencies

```bash
!pip install -q -r requirements.txt
```

Avoid `pip install -U -r requirements.txt` on Kaggle. A broad upgrade can pull
core notebook packages such as `numpy`, `pandas`, and `matplotlib` beyond the
versions expected by preinstalled Kaggle libraries.

## 5. Check Dataset Discovery

The Kaggle entrypoint can find `visdrone.yaml` automatically:

```bash
!python scripts/kaggle_train_visdrone.py --dry-run
```

If multiple files are found, pass the exact one:

```bash
!python scripts/kaggle_train_visdrone.py \
  --data /kaggle/input/<dataset-folder>/VisDrone_Dataset/visdrone.yaml \
  --dry-run
```

You do not need to copy VisDrone into this repository when the dataset is
already attached under `/kaggle/input`.

## 6. Train a Small Baseline First

Start with YOLOv8n to verify the full pipeline:

```bash
!python scripts/kaggle_train_visdrone.py \
  --models yolov8n \
  --project /kaggle/working/experiments/visdrone \
  --epochs 20 \
  --batch 8 \
  --workers 2 \
  --device 0 \
  --wandb-project real-time-od-visdrone
```

Outputs are written to:

```text
/kaggle/working/experiments/visdrone/yolov8n/
```

Important files:

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `confusion_matrix.png`
- `PR_curve.png`
- `F1_curve.png`
- `val_batch*_pred.jpg`

## 7. Train Comparison Experiments

After the baseline works, train YOLOv8n-P2 and YOLOv8s for comparison:

```bash
!python scripts/kaggle_train_visdrone.py \
  --models yolov8n-p2 yolov8s \
  --project /kaggle/working/experiments/visdrone \
  --epochs 100 \
  --batch 8 \
  --workers 2 \
  --device 0 \
  --wandb-project real-time-od-visdrone
```

Use YOLOv8n-P2 to test whether adding a P2 head improves the lightweight
YOLOv8n baseline. Use YOLOv8s as the larger-model reference. If GPU memory is
not enough, reduce `--batch` to `4` or `2`.

## 8. W&B Notes

The Kaggle entrypoint enables Ultralytics W&B logging when `--wandb-project` is
provided. It reads the API key from the `WANDB_API_KEY` environment variable or
the Kaggle secret with the same name.

Use offline mode if internet is unstable:

```bash
!python scripts/kaggle_train_visdrone.py \
  --models yolov8n \
  --epochs 20 \
  --batch 8 \
  --device 0 \
  --wandb-project real-time-od-visdrone \
  --wandb-mode offline
```

W&B should show links in the cell output after training starts.

## 9. Save Outputs

Kaggle automatically keeps `/kaggle/working` as notebook output when you save a
version. For the report, download or reference:

```text
/kaggle/working/experiments/
```

For long-term model storage, prefer W&B Artifacts or Kaggle Dataset versions
instead of committing checkpoints to GitHub.
