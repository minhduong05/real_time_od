# Realtime Frontend Guide

App frontend local nằm ở:

```text
app/realtime_front.py
src/realtime_od/realtime_app.py
```

Giải thích quan hệ giữa các file frontend/backend nằm ở:

```text
FRONTEND_ARCHITECTURE.md
```

## Mục tiêu

App xử lý video realtime trong browser:

```text
video frame -> YOLOv8n-P2 detection -> optional ByteTrack/analytics -> MJPEG stream
```

Không cần chờ xử lý xong mới xuất file video.

## Chạy app

```powershell
conda activate real_time_od
pip install -r requirements.txt
python app/realtime_front.py
```

Mở:

```text
http://127.0.0.1:7860
```

## Tính năng hiện có

1. Chọn video từ thư mục `video/`.
2. Object detection luôn bật.
3. `Traffic Density Estimation`: chọn số vùng, mỗi vùng bấm 4 điểm tọa độ trên frame đầu rồi bấm `OK vùng`.
4. `Vehicle Counting`: bấm 2 điểm tọa độ trên frame đầu rồi bấm `OK thanh`.
5. `Track`: bật ByteTrack, hiển thị ID và trail. Nếu không bật, app chỉ vẽ detection box, trừ khi counting cần dùng ID nội bộ để đếm.

## Khi nào cần vẽ trước khi Start?

`Traffic Density Estimation`: phải vẽ đủ số vùng đã chọn.

`Vehicle Counting`: phải vẽ counting line.

`Track` không cần vẽ thêm gì, nhưng chỉ chạy khi bạn bật checkbox tương ứng.

## Video nên dùng

Khuyến nghị cho demo local:

```text
Resolution: 720p hoặc 1080p
FPS: 24-30 FPS
Độ dài: 20-90 giây khi demo
Góc quay: camera tĩnh, nhìn xuống đường/giao lộ
Ánh sáng: ban ngày hoặc đủ sáng
```

720p là lựa chọn cân bằng nhất cho RTX 3050 Ti vì realtime mượt hơn. 1080p nhìn rõ hơn nhưng nặng hơn, đặc biệt khi bật Track + Counting + Density cùng lúc.

Tránh video:

```text
camera rung mạnh
dashcam chuyển động nhanh
đêm tối hoặc mưa nặng
độ phân giải quá cao 4K khi chạy realtime local
góc quay quá xa khiến object chỉ vài pixel
```

## Ghi chú

Frontend ưu tiên demo realtime. Các script batch `app/export_detection.py` và `app/export_detections.py` chỉ xuất Object Detection để phục vụ báo cáo. Tracking chỉ xử lý trong HTTP frontend.
