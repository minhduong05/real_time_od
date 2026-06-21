"""Realtime browser UI for traffic video analysis."""

from __future__ import annotations

import json
import threading
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request
from ultralytics import YOLO

from realtime_od.config import PROJECT_ROOT, resolve_project_path
from realtime_od.types import Detection


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MODEL_WEIGHTS = "models/Intersection-Flow-5K-Yolov8n-P2/best.pt"
VEHICLE_CLASSES = {"vehicle", "bus", "bicycle", "engine", "truck", "tricycle"}


INDEX_HTML = r"""
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Traffic Realtime Monitor</title>
  <style>
    :root {
      --bg: #101418;
      --panel: #171d23;
      --panel-2: #1f2831;
      --text: #eef3f7;
      --muted: #9cacb8;
      --accent: #35c28f;
      --line: #2c3944;
      --danger: #ff6961;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Segoe UI, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      letter-spacing: 0;
    }
    main {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 340px 1fr;
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 18px;
      overflow-y: auto;
    }
    h1 {
      font-size: 20px;
      margin: 0 0 16px;
      font-weight: 700;
    }
    h2 {
      font-size: 13px;
      color: var(--muted);
      margin: 22px 0 10px;
      text-transform: uppercase;
    }
    label {
      display: block;
      font-size: 14px;
      margin: 10px 0;
    }
    select, input[type="number"] {
      width: 100%;
      margin-top: 6px;
      padding: 9px 10px;
      background: var(--panel-2);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 9px;
      margin: 11px 0;
    }
    .check input { width: 18px; height: 18px; }
    button {
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 6px;
      padding: 9px 11px;
      cursor: pointer;
      font-weight: 600;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #062014;
    }
    button.danger {
      background: transparent;
      border-color: var(--danger);
      color: var(--danger);
    }
    button:disabled {
      opacity: .45;
      cursor: not-allowed;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 10px;
    }
    .hint {
      font-size: 13px;
      color: var(--muted);
      line-height: 1.45;
      margin: 8px 0;
    }
    .status {
      margin-top: 12px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--muted);
      font-size: 13px;
      white-space: pre-wrap;
    }
    section.viewer {
      padding: 18px;
      overflow: auto;
    }
    .stage {
      position: relative;
      width: min(100%, 1280px);
      background: #050607;
      border: 1px solid var(--line);
    }
    #snapshot, #stream {
      display: block;
      width: 100%;
      height: auto;
    }
    #drawCanvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      cursor: crosshair;
    }
    .hidden { display: none !important; }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 12px 0 0;
    }
    .badge {
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 5px 8px;
      color: var(--muted);
      margin: 4px 4px 0 0;
      font-size: 12px;
    }
  </style>
</head>
<body>
<main>
  <aside>
    <h1>Traffic Realtime Monitor</h1>

    <h2>Video</h2>
    <label>
      Chọn video
      <select id="videoSelect"></select>
    </label>
    <button id="loadFrame">Load frame đầu</button>

    <h2>Xử lý</h2>
    <div class="hint">Object detection luôn bật.</div>
    <label class="check"><input type="checkbox" id="optDensity"> Traffic Density Estimation</label>
    <label class="check"><input type="checkbox" id="optCounting"> Vehicle Counting</label>
    <label class="check"><input type="checkbox" id="optTrack"> Track</label>

    <div id="densityPanel" class="hidden">
      <h2>Density zones</h2>
      <label>
        Số vùng
        <input type="number" id="zoneCount" min="1" max="8" value="2">
      </label>
      <button id="drawZones">Vẽ vùng</button>
      <button id="okZone">OK vùng</button>
      <button id="clearZones">Xóa vùng</button>
      <div class="hint">Bấm 4 điểm trên frame để tạo một vùng polygon, rồi bấm OK vùng. Lặp lại đến khi đủ số vùng.</div>
      <div id="zoneBadges"></div>
    </div>

    <div id="countPanel" class="hidden">
      <h2>Counting line</h2>
      <button id="drawLine">Vẽ thanh đếm</button>
      <button id="okLine">OK thanh</button>
      <button id="clearLine">Xóa thanh</button>
      <div class="hint">Bấm 2 điểm trên frame để tạo thanh đếm, rồi bấm OK thanh. Object đi qua line sẽ được đếm một lần.</div>
      <div id="lineBadge"></div>
    </div>

    <h2>Runtime</h2>
    <div class="row">
      <label>Confidence<input type="number" id="conf" min="0.05" max="0.95" step="0.05" value="0.35"></label>
      <label>Max box area<input type="number" id="maxArea" min="0.02" max="1" step="0.01" value="0.12"></label>
    </div>
    <div class="toolbar">
      <button class="primary" id="start">Start</button>
      <button class="danger" id="stop">Stop</button>
    </div>
    <div id="status" class="status">Sẵn sàng.</div>
  </aside>

  <section class="viewer">
    <div class="stage" id="stage">
      <img id="snapshot" alt="first frame">
      <img id="stream" class="hidden" alt="processed stream">
      <canvas id="drawCanvas"></canvas>
    </div>
  </section>
</main>

<script>
const videoSelect = document.getElementById("videoSelect");
const snapshot = document.getElementById("snapshot");
const stream = document.getElementById("stream");
const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");
const statusBox = document.getElementById("status");
const optDensity = document.getElementById("optDensity");
const optCounting = document.getElementById("optCounting");
const optTrack = document.getElementById("optTrack");
const densityPanel = document.getElementById("densityPanel");
const countPanel = document.getElementById("countPanel");
const zoneBadges = document.getElementById("zoneBadges");
const lineBadge = document.getElementById("lineBadge");

let videoInfo = null;
let drawMode = null;
let densityZones = [];
let countLine = null;
let currentZonePoints = [];
let currentLinePoints = [];

function setStatus(text) { statusBox.textContent = text; }

function resizeCanvas() {
  const rect = snapshot.classList.contains("hidden") ? stream.getBoundingClientRect() : snapshot.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width));
  canvas.height = Math.max(1, Math.round(rect.height));
  redrawCanvas();
}

function scaleToVideo(point) {
  const sx = videoInfo.width / canvas.width;
  const sy = videoInfo.height / canvas.height;
  return { x: Math.round(point.x * sx), y: Math.round(point.y * sy) };
}

function scaleFromVideo(point) {
  const sx = canvas.width / videoInfo.width;
  const sy = canvas.height / videoInfo.height;
  return { x: Math.round(point.x * sx), y: Math.round(point.y * sy) };
}

function lineFromVideoPoints(points) {
  return { p1: points[0], p2: points[1] };
}

function drawPolygon(zone, color, label) {
  const points = zone.points.map(scaleFromVideo);
  if (!points.length) return;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (const point of points.slice(1)) ctx.lineTo(point.x, point.y);
  if (points.length >= 3) ctx.closePath();
  ctx.stroke();
  for (const point of points) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.font = "14px Segoe UI";
  ctx.fillText(label, points[0].x + 6, points[0].y + 18);
}

function drawLine(line, color, label) {
  const a = scaleFromVideo(line.p1);
  const b = scaleFromVideo(line.p2);
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.font = "14px Segoe UI";
  ctx.fillText(label, a.x + 6, a.y - 8);
}

function redrawCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  densityZones.forEach((zone, index) => drawPolygon(zone, "#35c28f", `Zone ${index + 1}`));
  if (countLine) drawLine(countLine, "#ffcc33", "Counting line");
  if (currentZonePoints.length) drawPolygon({ points: currentZonePoints }, "#8bd9ff", `Draft ${currentZonePoints.length}/4`);
  if (currentLinePoints.length === 1) {
    const p = scaleFromVideo(currentLinePoints[0]);
    ctx.fillStyle = "#8bd9ff";
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
    ctx.fill();
  }
  if (currentLinePoints.length === 2) drawLine(lineFromVideoPoints(currentLinePoints), "#8bd9ff", "Draft line");
  renderBadges();
}

function renderBadges() {
  zoneBadges.innerHTML = densityZones.map((z, i) =>
    `<span class="badge">Z${i + 1}: ${z.points.map(p => `(${p.x},${p.y})`).join(" ")}</span>`
  ).join("");
  lineBadge.innerHTML = countLine
    ? `<span class="badge">(${countLine.p1.x},${countLine.p1.y}) -> (${countLine.p2.x},${countLine.p2.y})</span>`
    : "";
}

async function loadVideos() {
  const response = await fetch("/api/videos");
  const videos = await response.json();
  videoSelect.innerHTML = videos.map(v => `<option value="${v.path}">${v.name}</option>`).join("");
}

async function loadFrame() {
  const video = videoSelect.value;
  const infoResponse = await fetch(`/api/video-info?video=${encodeURIComponent(video)}`);
  videoInfo = await infoResponse.json();
  snapshot.src = `/api/frame?video=${encodeURIComponent(video)}&t=${Date.now()}`;
  snapshot.classList.remove("hidden");
  stream.classList.add("hidden");
  canvas.classList.remove("hidden");
  densityZones = [];
  countLine = null;
  currentZonePoints = [];
  currentLinePoints = [];
  drawMode = null;
  snapshot.onload = resizeCanvas;
  setStatus(`Loaded ${videoInfo.name}\n${videoInfo.width}x${videoInfo.height}, ${videoInfo.fps.toFixed(2)} FPS, ${videoInfo.frames} frames`);
}

function updatePanels() {
  densityPanel.classList.toggle("hidden", !optDensity.checked);
  countPanel.classList.toggle("hidden", !optCounting.checked);
}

canvas.addEventListener("click", event => {
  if (!videoInfo || !drawMode) return;
  const rect = canvas.getBoundingClientRect();
  const point = scaleToVideo({ x: event.clientX - rect.left, y: event.clientY - rect.top });
  if (drawMode === "zones") {
    if (currentZonePoints.length < 4) currentZonePoints.push(point);
    setStatus(`Vùng hiện tại: ${currentZonePoints.length}/4 điểm. Bấm OK vùng khi đủ 4 điểm.`);
  } else {
    if (currentLinePoints.length < 2) currentLinePoints.push(point);
    setStatus(`Thanh đếm: ${currentLinePoints.length}/2 điểm. Bấm OK thanh khi đủ 2 điểm.`);
  }
  redrawCanvas();
});

document.getElementById("loadFrame").onclick = loadFrame;
document.getElementById("drawZones").onclick = () => {
  densityZones = [];
  currentZonePoints = [];
  drawMode = "zones";
  setStatus(`Bấm 4 điểm cho từng density zone. Cần ${document.getElementById("zoneCount").value} vùng.`);
  redrawCanvas();
};
document.getElementById("okZone").onclick = () => {
  const maxZones = Number(document.getElementById("zoneCount").value || 1);
  if (currentZonePoints.length !== 4) {
    setStatus("Một vùng density cần đúng 4 điểm trước khi OK.");
    return;
  }
  if (densityZones.length >= maxZones) {
    setStatus("Đã đủ số vùng đã chọn.");
    return;
  }
  densityZones.push({ points: currentZonePoints });
  currentZonePoints = [];
  drawMode = densityZones.length < maxZones ? "zones" : null;
  setStatus(drawMode ? `Đã lưu vùng ${densityZones.length}. Tiếp tục bấm 4 điểm cho vùng kế tiếp.` : "Đã lưu đủ vùng density.");
  redrawCanvas();
};
document.getElementById("clearZones").onclick = () => { densityZones = []; currentZonePoints = []; drawMode = null; redrawCanvas(); };
document.getElementById("drawLine").onclick = () => { countLine = null; currentLinePoints = []; drawMode = "line"; setStatus("Bấm 2 điểm cho counting line, rồi bấm OK thanh."); redrawCanvas(); };
document.getElementById("okLine").onclick = () => {
  if (currentLinePoints.length !== 2) {
    setStatus("Counting line cần đúng 2 điểm trước khi OK.");
    return;
  }
  countLine = lineFromVideoPoints(currentLinePoints);
  currentLinePoints = [];
  drawMode = null;
  setStatus("Đã lưu counting line.");
  redrawCanvas();
};
document.getElementById("clearLine").onclick = () => { countLine = null; currentLinePoints = []; drawMode = null; redrawCanvas(); };
document.getElementById("optDensity").onchange = updatePanels;
document.getElementById("optCounting").onchange = updatePanels;

document.getElementById("start").onclick = async () => {
  if (!videoInfo) await loadFrame();
  if (optDensity.checked && densityZones.length !== Number(document.getElementById("zoneCount").value)) {
    setStatus("Cần vẽ đủ density zones rồi mới Start.");
    return;
  }
  if (optCounting.checked && !countLine) {
    setStatus("Cần vẽ counting line rồi mới Start.");
    return;
  }
  const config = {
    video: videoSelect.value,
    options: {
      density: optDensity.checked,
      counting: optCounting.checked,
      track: optTrack.checked,
    },
    density_zones: densityZones,
    count_line: countLine,
    conf: Number(document.getElementById("conf").value),
    max_box_area_ratio: Number(document.getElementById("maxArea").value),
  };
  await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  canvas.classList.add("hidden");
  snapshot.classList.add("hidden");
  stream.classList.remove("hidden");
  stream.src = `/video_feed?t=${Date.now()}`;
  setStatus("Đang xử lý realtime stream...");
};

document.getElementById("stop").onclick = () => {
  stream.src = "";
  stream.classList.add("hidden");
  snapshot.classList.remove("hidden");
  canvas.classList.remove("hidden");
  setStatus("Đã dừng stream.");
};

window.addEventListener("resize", resizeCanvas);
loadVideos().then(loadFrame).then(updatePanels);
</script>
</body>
</html>
"""


@dataclass
class RuntimeConfig:
    video: str = ""
    options: dict[str, bool] = field(default_factory=dict)
    density_zones: list[dict[str, Any]] = field(default_factory=list)
    count_line: dict[str, dict[str, int]] | None = None
    conf: float = 0.35
    max_box_area_ratio: float = 0.12


class RealtimeState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.config = RuntimeConfig()
        self.model: YOLO | None = None

    def get_model(self) -> YOLO:
        if self.model is None:
            weights = resolve_project_path(MODEL_WEIGHTS)
            if not weights.exists():
                raise FileNotFoundError(f"Model weights not found: {weights}")
            self.model = YOLO(str(weights))
        return self.model

    def set_config(self, payload: dict[str, Any]) -> None:
        with self.lock:
            self.config = RuntimeConfig(
                video=str(payload.get("video", "")),
                options=dict(payload.get("options", {})),
                density_zones=list(payload.get("density_zones", [])),
                count_line=payload.get("count_line"),
                conf=float(payload.get("conf", 0.35)),
                max_box_area_ratio=float(payload.get("max_box_area_ratio", 0.12)),
            )

    def get_config(self) -> RuntimeConfig:
        with self.lock:
            return RuntimeConfig(
                video=self.config.video,
                options=dict(self.config.options),
                density_zones=list(self.config.density_zones),
                count_line=self.config.count_line,
                conf=self.config.conf,
                max_box_area_ratio=self.config.max_box_area_ratio,
            )


state = RealtimeState()


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return render_template_string(INDEX_HTML)

    @app.get("/api/videos")
    def videos() -> Response:
        video_dir = PROJECT_ROOT / "video"
        items = []
        if video_dir.exists():
            for path in sorted(video_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                    items.append({"name": path.name, "path": f"video/{path.name}"})
        return jsonify(items)

    @app.get("/api/video-info")
    def video_info() -> Response:
        video_path = resolve_project_path(request.args["video"])
        info = read_video_info(video_path)
        return jsonify(info)

    @app.get("/api/frame")
    def first_frame() -> Response:
        video_path = resolve_project_path(request.args["video"])
        capture = cv2.VideoCapture(str(video_path))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            return Response("Could not read first frame", status=400)

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return Response("Could not encode frame", status=500)
        return Response(buffer.tobytes(), mimetype="image/jpeg")

    @app.post("/api/config")
    def set_config() -> Response:
        payload = request.get_json(force=True)
        state.set_config(payload)
        return jsonify({"ok": True})

    @app.get("/video_feed")
    def video_feed() -> Response:
        return Response(process_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def read_video_info(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return {
            "name": path.name,
            "width": width,
            "height": height,
            "fps": fps,
            "frames": frames,
            "duration_sec": frames / fps if fps else 0,
        }
    finally:
        capture.release()


def process_stream() -> Iterable[bytes]:
    config = state.get_config()
    video_path = resolve_project_path(config.video)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    model = state.get_model()
    use_tracking = config.options.get("track", False) or config.options.get("counting", False)
    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=30))
    previous_centers: dict[int, tuple[int, int]] = {}
    counted_ids: set[int] = set()
    count_by_class: Counter[str] = Counter()
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            results = (
                model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=config.conf,
                    iou=0.7,
                    imgsz=640,
                    device=0,
                    verbose=False,
                )
                if use_tracking
                else model.predict(
                    source=frame,
                    conf=config.conf,
                    iou=0.7,
                    imgsz=640,
                    device=0,
                    verbose=False,
                )
            )
            detections = extract_detections(results[0], frame_index, frame_index / fps, frame.shape, config)
            annotated = frame.copy()
            active_counts = Counter(d.class_name for d in detections)

            if config.options.get("density"):
                draw_density_zones(annotated, detections, config.density_zones)

            if config.options.get("counting") and config.count_line:
                update_and_draw_counting(
                    annotated,
                    detections,
                    config.count_line,
                    previous_centers,
                    counted_ids,
                    count_by_class,
                )

            for detection in detections:
                draw_detection(annotated, detection, trails, show_track=config.options.get("track", False))

            draw_active_overlay(annotated, active_counts, frame_index, config.options)
            ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                break
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            frame_index += 1
    finally:
        capture.release()


def extract_detections(
    result: object,
    frame_index: int,
    time_sec: float,
    frame_shape: tuple[int, ...],
    config: RuntimeConfig,
) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    names = result.names if hasattr(result, "names") else {}
    xyxy_values = boxes.xyxy.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    if boxes.id is None:
        track_ids = np.arange(len(xyxy_values), dtype=int) * -1 - 1
    else:
        track_ids = boxes.id.cpu().numpy().astype(int)

    frame_height, frame_width = frame_shape[:2]
    frame_area = max(frame_width * frame_height, 1)
    detections = []
    for xyxy, track_id, class_id, confidence in zip(xyxy_values, track_ids, class_ids, confidences):
        x1, y1, x2, y2 = (float(value) for value in xyxy)
        box_area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / frame_area
        if box_area_ratio > config.max_box_area_ratio:
            continue
        class_name = str(names.get(int(class_id), f"class_{int(class_id)}"))
        detections.append(
            Detection(
                frame_index=frame_index,
                time_sec=time_sec,
                track_id=int(track_id),
                class_id=int(class_id),
                class_name=class_name,
                confidence=float(confidence),
                xyxy=(x1, y1, x2, y2),
            )
        )
    return detections


def draw_detection(
    frame: np.ndarray,
    detection: Detection,
    trails: dict[int, deque[tuple[int, int]]],
    show_track: bool,
) -> None:
    color = color_for_id(detection.track_id if detection.track_id >= 0 else detection.class_id)
    x1, y1, x2, y2 = (int(value) for value in detection.xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label_id = f" #{detection.track_id}" if show_track and detection.track_id >= 0 else ""
    label = f"{detection.class_name}{label_id} {detection.confidence:.2f}"
    draw_label(frame, label, (x1, y1 - 5), color)

    if show_track and detection.track_id >= 0:
        trails[detection.track_id].append(detection.center)
        points = np.array(trails[detection.track_id], dtype=np.int32)
        if len(points) >= 2:
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)
        cv2.circle(frame, detection.center, 3, color, -1)


def draw_density_zones(
    frame: np.ndarray,
    detections: list[Detection],
    zones: list[dict[str, int]],
) -> None:
    for index, zone in enumerate(zones, start=1):
        points = [(int(point["x"]), int(point["y"])) for point in zone.get("points", [])]
        if len(points) != 4:
            continue
        contour = np.array(points, dtype=np.int32)
        count = sum(
            1 for detection in detections
            if detection.class_name in VEHICLE_CLASSES
            and point_in_polygon(detection.center, points)
        )
        level = "LOW"
        color = (60, 190, 95)
        if count >= 4:
            level = "MED"
            color = (40, 190, 230)
        if count >= 9:
            level = "HIGH"
            color = (45, 80, 230)
        cv2.polylines(frame, [contour], isClosed=True, color=color, thickness=2)
        label_origin = points[0]
        draw_label(frame, f"Zone {index}: {level} ({count})", (label_origin[0] + 6, label_origin[1] + 24), color)


def point_in_polygon(point: tuple[int, int], polygon: list[tuple[int, int]]) -> bool:
    contour = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def update_and_draw_counting(
    frame: np.ndarray,
    detections: list[Detection],
    line: dict[str, dict[str, int]],
    previous_centers: dict[int, tuple[int, int]],
    counted_ids: set[int],
    count_by_class: Counter[str],
) -> None:
    p1 = (int(line["p1"]["x"]), int(line["p1"]["y"]))
    p2 = (int(line["p2"]["x"]), int(line["p2"]["y"]))
    cv2.line(frame, p1, p2, (0, 220, 255), 3)

    for detection in detections:
        if detection.track_id < 0:
            continue
        current = detection.center
        previous = previous_centers.get(detection.track_id)
        if previous and detection.track_id not in counted_ids:
            if crossed_line(previous, current, p1, p2):
                counted_ids.add(detection.track_id)
                count_by_class[detection.class_name] += 1
        previous_centers[detection.track_id] = current

    total = sum(count_by_class.values())
    draw_label(frame, f"Count: {total}", (p1[0] + 8, p1[1] - 12), (0, 220, 255))
    y = 92
    for class_name, count in sorted(count_by_class.items()):
        draw_label(frame, f"{class_name}: {count}", (12, y), (0, 220, 255))
        y += 24


def crossed_line(
    previous: tuple[int, int],
    current: tuple[int, int],
    line_a: tuple[int, int],
    line_b: tuple[int, int],
) -> bool:
    prev_side = side_of_line(previous, line_a, line_b)
    curr_side = side_of_line(current, line_a, line_b)
    return prev_side != 0 and curr_side != 0 and prev_side != curr_side


def side_of_line(point: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> int:
    value = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def draw_active_overlay(
    frame: np.ndarray,
    counts: Counter[str],
    frame_index: int,
    options: dict[str, bool],
) -> None:
    enabled = ["Detection"]
    enabled.extend(name for name, is_on in options.items() if is_on)
    lines = [f"Frame {frame_index}", "On: " + ", ".join(enabled)]
    lines.extend(f"{name}: {count}" for name, count in sorted(counts.items()))

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    line_height = 23
    width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines) + 22
    height = line_height * len(lines) + 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (12 + width, 12 + height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (23, 35 + index * line_height), font, scale, (245, 245, 245), thickness, cv2.LINE_AA)


def draw_label(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    x, y = origin
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(y, height + baseline + 4)
    cv2.rectangle(frame, (x, y - height - baseline - 5), (x + width + 6, y + 3), color, -1)
    cv2.putText(frame, text, (x + 3, y - 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def color_for_id(identifier: int) -> tuple[int, int, int]:
    palette = (
        (52, 152, 219),
        (46, 204, 113),
        (241, 196, 15),
        (231, 76, 60),
        (155, 89, 182),
        (26, 188, 156),
        (230, 126, 34),
        (149, 165, 166),
    )
    return palette[identifier % len(palette)]
