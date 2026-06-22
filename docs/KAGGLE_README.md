# Kaggle Workflows

Thư mục này chỉ giữ workflow liên quan đến VisDrone.

```text
docs/VISDRONE_RUN_GUIDE.md             train/fine-tune YOLOv8n, YOLOv8n-P2, YOLOv8s
docs/VISDRONE_TEST_BENCHMARK_GUIDE.md  đánh giá 3 model trên VisDrone test-dev
kaggle/train.py                          script train chung cho Kaggle
kaggle/check_model.py                    kiểm tra model/config trước khi train
```

Checkpoint export từ Kaggle đặt về local theo cấu trúc:

```text
models/VisDrone/yolov8n/best.pt
models/VisDrone/yolov8n-p2/best.pt
models/VisDrone/yolov8s/best.pt
```

Frontend local sẽ cho chọn trực tiếp 3 model này.
