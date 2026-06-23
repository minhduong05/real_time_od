# Kaggle Workflows

Thư mục `kaggle/` chỉ giữ workflow liên quan đến VisDrone.

```text
kaggle/train.py          Train/tune YOLOv8n, YOLOv8n-P2, YOLOv8s
kaggle/check_model.py    Kiểm tra model config và pretrained transfer
kaggle/requirements.txt  Dependencies tối thiểu cho Kaggle notebook
```

Các hướng dẫn chính:

```text
docs/VISDRONE_RUN_GUIDE.md             Fine-tune 3 model trên Kaggle
docs/VISDRONE_TEST_BENCHMARK_GUIDE.md  Benchmark 3 model, log W&B và export số liệu báo cáo
```

## Output Từ Kaggle

Sau khi train, `kaggle/train.py` export checkpoint vào:

```text
/kaggle/working/export/VisDrone/<model>/
  best.pt
  last.pt
  run_info.yaml
```

Tải các file cần thiết về local:

```text
models/VisDrone/yolov8n/best.pt
models/VisDrone/yolov8n-p2/best.pt
models/VisDrone/yolov8s/best.pt
```

`last.pt` có thể giữ để tham khảo, nhưng app local dùng `best.pt`.

## W&B

Train guide dùng W&B để theo dõi train/val. Benchmark guide dùng W&B để log test-dev metrics, per-class AP và media phục vụ báo cáo.

Nếu chỉ muốn tải số liệu W&B về sau khi notebook cũ đã tắt, đọc phần cuối của:

```text
docs/VISDRONE_TEST_BENCHMARK_GUIDE.md
```

## Lưu Về Local

Các file số liệu tải từ Kaggle Output hoặc W&B export nên đặt vào:

```text
experiments/
```

Ví dụ:

```text
experiments/visdrone_test_dev_report_export.zip
experiments/wandb_visdrone_export.zip
experiments/visdrone_test_dev_comparison.csv
```

Folder này dành cho phân tích thực nghiệm và viết báo cáo. Nội dung bên trong không commit git, chỉ giữ `.gitkeep`.
