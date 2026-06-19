# Model Storage

Store fine-tuned checkpoints by dataset and model name:

```text
models/
  VisDrone/
    yolov8n/
      best.pt
      last.pt
    yolov8n-p2/
      best.pt
      last.pt
  Intersection-Flow-5K/
    yolov8n/
      best.pt
      last.pt
    yolov8n-p2/
      best.pt
      last.pt
```

The app does not choose a default checkpoint. Always pass the exact file:

```bash
python app/run.py --weights models/VisDrone/yolov8n-p2/best.pt --source 0
```

Checkpoint files are ignored by git.
