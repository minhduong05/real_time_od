# Experiments Folder

`experiments/` dùng để lưu dữ liệu thực nghiệm tải về sau khi train/test trên Kaggle hoặc W&B. Folder này phục vụ viết báo cáo và phân tích so sánh 3 model VisDrone.

## Nên Đặt Gì Ở Đây?

```text
experiments/
  wandb_visdrone_export.zip
  visdrone_test_dev_report_export.zip
  visdrone_test_dev_comparison.csv
  visdrone_test_dev_per_class_comparison.csv
  plots/
  tables/
```

Các file thường đến từ:

```text
docs/VISDRONE_TEST_BENCHMARK_GUIDE.md
```

Đặc biệt là phần export W&B/report:

```text
/kaggle/working/visdrone_test_dev_report_export.zip
/kaggle/working/wandb_visdrone_export.zip
```

## Có Commit Không?

Không nên commit dữ liệu trong `experiments/` vì có thể lớn và thay đổi nhiều. `.gitignore` hiện chỉ giữ:

```text
experiments/.gitkeep
```

Nội dung CSV/ZIP/ảnh tải về sẽ ở local để phân tích báo cáo.

## Khác Gì `outputs/`?

```text
outputs/      Kết quả sinh trực tiếp khi chạy app local: annotated video, logs realtime
experiments/  Số liệu thực nghiệm tải về từ Kaggle/W&B để phân tích và viết báo cáo
```

