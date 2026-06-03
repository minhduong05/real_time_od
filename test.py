from ultralytics import YOLO

model = YOLO("yolov26n.pt")

results = model("bus.jpg", device=0, show=True)
