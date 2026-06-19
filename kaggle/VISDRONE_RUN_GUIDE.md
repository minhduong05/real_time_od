# VisDrone Kaggle Run Guide

Guide nay dung cho notebook Kaggle khi fine-tuning cac model YOLO voi dataset
VisDrone, dac biet la `yolov8n-p2`. Kaggle chi dung de train/tune bang GPU.
Sau khi xong, tai checkpoint ve local va dat vao `models/VisDrone/<model>/`.

## 1. Kaggle Settings

Trong notebook Kaggle:

```text
Settings -> Accelerator -> GPU T4 x2
Settings -> Internet -> On
```

Kiem tra GPU:

```bash
!nvidia-smi
```

## 2. Add VisDrone Dataset

Ben phai notebook:

```text
Add Input -> chon dataset VisDrone YOLO
```

Tim file dataset YAML:

```bash
!find /kaggle/input -name "*.yaml" -o -name "*.yml"
```

Vi du neu thay:

```text
/kaggle/input/visdrone-yolo/visdrone.yaml
```

thi dung path do cho `--data`.

## 3. Clone Repo

Neu repo nam tren GitHub:

```bash
!git clone https://github.com/<your-user>/<your-repo>.git /kaggle/working/real_time_od
%cd /kaggle/working/real_time_od
```

Neu repo da duoc upload bang Kaggle Dataset, `%cd` vao folder repo tuong ung.

## 4. Install Dependencies

```bash
!pip install -r kaggle/requirements.txt
!pip install wandb
```

Kiem tra PyTorch/CUDA/Ultralytics:

```bash
!python - <<'PY'
import torch
import ultralytics

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("ultralytics:", ultralytics.__version__)
PY
```

## 5. W&B Secret

Trong Kaggle:

```text
Add-ons -> Secrets
```

Tao secret:

```text
Label: WANDB_API_KEY
Value: API key lay tu https://wandb.ai/authorize
```

Neu secret da co san, chi can chay cell login:

```python
from kaggle_secrets import UserSecretsClient
import wandb

wandb_api_key = UserSecretsClient().get_secret("WANDB_API_KEY")
wandb.login(key=wandb_api_key)
```

## 6. Set W&B Project

Dung cung project voi run `yolov8n` truoc do:

```python
%env WANDB_ENTITY=minhmit146-hanoi-university-of-science-and-technology
%env WANDB_PROJECT=-kaggle-working-experiments-visdrone
%env WANDB_NAME=yolov8n-p2
```

Bat W&B cho Ultralytics:

```bash
!yolo settings wandb=True
```

## 7. Dry Run

Sua `--data` theo path YAML tim duoc o buoc 2.

```bash
!python kaggle/train.py \
  --dataset-name VisDrone \
  --data /kaggle/input/visdrone-yolo/visdrone.yaml \
  --model yolov8n-p2 \
  --epochs 1 \
  --imgsz 640 \
  --batch 16 \
  --workers 4 \
  --device 0,1 \
  --project /kaggle/working/experiments/visdrone \
  --name yolov8n-p2 \
  --dry-run
```

Log dung se co dang:

```text
model: yolov8n-p2
dataset: VisDrone
output: /kaggle/working/experiments/visdrone/yolov8n-p2
export: /kaggle/working/export/VisDrone/yolov8n-p2
```

## 8. Check YOLOv8n-P2 Transfer

Dry-run chi kiem tra path/args, chua build model. Chay lenh nay truoc khi train
de xem `yolov8n-p2` co load dung `yolov8n.pt` va transfer duoc bao nhieu tensor:

```bash
!python kaggle/check_model.py --model yolov8n-p2
```

Trong log can thay:

```text
model: yolov8n-p2
transfer_from: yolov8n.pt
Transferred X/Y items from pretrained weights
```

`X/Y` la so `state_dict` tensor copy duoc tu `yolov8n.pt` sang kien truc
`yolov8n-p2`. Day khong phai so layer. Vi `yolov8n-p2` them P2 head va dang
train voi `nc: 10`, mot so tensor se khong khop shape va se duoc init moi.

Muon xem summary chi tiet hon:

```bash
!python kaggle/check_model.py --model yolov8n-p2 --detailed
```

## 9. Train YOLOv8n-P2

```bash
!python kaggle/train.py \
  --dataset-name VisDrone \
  --data /kaggle/input/visdrone-yolo/visdrone.yaml \
  --model yolov8n-p2 \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --workers 4 \
  --device 0,1 \
  --project /kaggle/working/experiments/visdrone \
  --name yolov8n-p2
```

Neu bi CUDA out of memory, giam batch:

```bash
--batch 8
```

Trong log, `yolov8n-p2` phai transfer tu `yolov8n.pt`, khong phai
`yolov8s.pt`.

## 10. Train Baseline YOLOv8n

Dung khi can train lai baseline de so sanh:

```bash
!python kaggle/train.py \
  --dataset-name VisDrone \
  --data /kaggle/input/visdrone-yolo/visdrone.yaml \
  --model yolov8n \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --workers 4 \
  --device 0,1 \
  --project /kaggle/working/experiments/visdrone \
  --name yolov8n
```

Neu train baseline, doi W&B run name truoc khi train:

```python
%env WANDB_NAME=yolov8n
```

## 11. Train YOLOv8s

Tao notebook Kaggle moi neu muon tach run `yolov8s` khoi `yolov8n-p2`.
Lap lai cac buoc GPU, dataset, repo, dependency, W&B Secret, va tao lai
`/kaggle/working/visdrone_fixed.yaml` trong notebook moi.

Dat ten W&B run:

```python
%env WANDB_NAME=yolov8s
```

Train:

```bash
!python kaggle/train.py \
  --dataset-name VisDrone \
  --data /kaggle/working/visdrone_fixed.yaml \
  --model yolov8s \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --workers 4 \
  --device 0,1 \
  --project /kaggle/working/experiments/visdrone \
  --name yolov8s
```

`yolov8s` su dung pretrained `yolov8s.pt` truc tiep. Khong co P2 architecture
nen khong can kiem tra transfer `219/437` nhu `yolov8n-p2`.

Export sau khi train:

```text
/kaggle/working/export/VisDrone/yolov8s/best.pt
/kaggle/working/export/VisDrone/yolov8s/last.pt
```

Dat checkpoint tai local vao:

```text
models/VisDrone/yolov8s/best.pt
models/VisDrone/yolov8s/last.pt
```

## 12. Check Export

Sau khi train xong:

```bash
!find /kaggle/working/export -type f
```

Voi `yolov8n-p2`, can thay:

```text
/kaggle/working/export/VisDrone/yolov8n-p2/best.pt
/kaggle/working/export/VisDrone/yolov8n-p2/last.pt
/kaggle/working/export/VisDrone/yolov8n-p2/run_info.yaml
```

## 13. Zip De Tai Ve

```bash
!zip -r /kaggle/working/VisDrone_yolov8n-p2_export.zip \
  /kaggle/working/export/VisDrone/yolov8n-p2
```

Tai file nay tu Kaggle output panel:

```text
/kaggle/working/VisDrone_yolov8n-p2_export.zip
```

## 14. Dat Checkpoint Vao Local Repo

Sau khi tai ve va giai nen, dat file vao:

```text
models/VisDrone/yolov8n-p2/best.pt
models/VisDrone/yolov8n-p2/last.pt
models/VisDrone/yolov8n-p2/run_info.yaml
```

Chay inference local:

```bash
python app/run.py --weights models/VisDrone/yolov8n-p2/best.pt --source 0
```

## 15. W&B Bao Cao

Trong W&B project:

```text
-kaggle-working-experiments-visdrone
```

So sanh cac run:

```text
yolov8n
yolov8n-p2
```

Dung cac metric de viet bao cao:

```text
metrics/precision
metrics/recall
metrics/mAP50
metrics/mAP50-95
train/box_loss
train/cls_loss
train/dfl_loss
val/box_loss
val/cls_loss
val/dfl_loss
```
