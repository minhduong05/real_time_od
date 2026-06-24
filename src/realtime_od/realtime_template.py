"""HTML template for the realtime traffic frontend."""

INDEX_HTML = r"""
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VisDrone Realtime Traffic Monitor</title>
  <style>
    :root {
      --bg: #101418;
      --panel: #171d23;
      --field: #202a33;
      --line: #2f3c47;
      --text: #eef3f7;
      --muted: #9cacb8;
      --accent: #35c28f;
      --danger: #ff6961;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Segoe UI, Arial, sans-serif;
      letter-spacing: 0;
    }
    main {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 340px 1fr;
    }
    aside {
      background: var(--panel);
      border-right: 1px solid var(--line);
      padding: 18px;
      overflow-y: auto;
    }
    h1 { margin: 0 0 16px; font-size: 20px; }
    h2 { margin: 22px 0 10px; color: var(--muted); font-size: 13px; text-transform: uppercase; }
    label { display: block; margin: 10px 0; font-size: 14px; }
    select, input[type="number"] {
      width: 100%;
      margin-top: 6px;
      padding: 9px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--field);
      color: var(--text);
    }
    button {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 11px;
      background: var(--field);
      color: var(--text);
      cursor: pointer;
      font-weight: 600;
    }
    button.primary { background: var(--accent); border-color: var(--accent); color: #062014; }
    button.danger { background: transparent; border-color: var(--danger); color: var(--danger); }
    .check { display: flex; align-items: center; gap: 9px; }
    .check input { width: 18px; height: 18px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .hint, .status, .badge { color: var(--muted); font-size: 13px; line-height: 1.45; }
    .status {
      margin-top: 12px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      white-space: pre-wrap;
    }
    .badge {
      display: inline-block;
      margin: 4px 4px 0 0;
      padding: 5px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    .hidden { display: none !important; }
    .viewer { padding: 18px; overflow: auto; }
    .stage {
      position: relative;
      width: min(100%, 1280px);
      background: #050607;
      border: 1px solid var(--line);
    }
    #snapshot, #stream { display: block; width: 100%; height: auto; }
    #drawCanvas { position: absolute; inset: 0; width: 100%; height: 100%; cursor: crosshair; }
  </style>
</head>
<body>
<main>
  <aside>
    <h1>VisDrone Traffic Monitor</h1>

    <h2>Video</h2>
    <label>Chọn video<select id="videoSelect"></select></label>
    <button id="loadFrame">Load frame đầu</button>

    <h2>Model</h2>
    <label>Chọn model VisDrone<select id="modelSelect"></select></label>
    <div id="modelHint" class="hint"></div>

    <h2>Xử lý</h2>
    <div class="hint">Object detection luôn bật. Mỗi lần Start sẽ dùng model đang chọn.</div>
    <label class="check"><input type="checkbox" id="optDensity"> Traffic Density Estimation</label>
    <label class="check"><input type="checkbox" id="optCounting"> Vehicle Counting</label>
    <label class="check"><input type="checkbox" id="optTrack"> Track</label>

    <div id="densityPanel" class="hidden">
      <h2>Density zones</h2>
      <label>Số vùng<input type="number" id="zoneCount" min="1" max="99" value="2"></label>
      <div class="toolbar">
        <button id="drawZones">Vẽ vùng</button>
        <button id="okZone">OK vùng</button>
        <button id="clearZones">Xóa vùng</button>
      </div>
      <div class="hint">Mỗi vùng cần đúng 4 điểm trên frame, rồi bấm OK vùng.</div>
      <div id="zoneBadges"></div>
    </div>

    <div id="countPanel" class="hidden">
      <h2>Counting line</h2>
      <label>Số thanh<input type="number" id="lineCount" min="1" max="99" value="1"></label>
      <div class="toolbar">
        <button id="drawLine">Vẽ thanh đếm</button>
        <button id="okLine">OK thanh</button>
        <button id="clearLine">Xóa thanh</button>
      </div>
      <div class="hint">Mỗi thanh đếm cần đúng 2 điểm trên frame, rồi bấm OK thanh.</div>
      <div id="lineBadge"></div>
    </div>

    <h2>Runtime</h2>
    <div class="row">
      <label>Confidence<input type="number" id="conf" min="0.05" max="0.95" step="0.05" value="0.35"></label>
      <label>Max box area<input type="number" id="maxArea" min="0.02" max="1" step="0.01" value="0.12"></label>
    </div>
    <div class="row">
      <label>Stream width
        <select id="streamWidth">
          <option value="1280" selected>1280</option>
          <option value="960">960</option>
          <option value="0">Original</option>
        </select>
      </label>
      <label>JPEG quality<input type="number" id="jpegQuality" min="65" max="90" step="1" value="75"></label>
    </div>
    <label class="check"><input type="checkbox" id="optLogging"> Log CSV</label>
    <div class="hint">Tắt log để test FPS sạch hơn; bật lại khi cần lưu kết quả từng frame.</div>
    <div class="toolbar">
      <button class="primary" id="start">Start</button>
      <button id="pauseToggle">Stop</button>
      <button class="danger" id="finish">Finish</button>
    </div>
    <div id="status" class="status">Sẵn sàng.</div>
  </aside>

  <section class="viewer">
    <div class="stage">
      <img id="snapshot" alt="first frame">
      <img id="stream" class="hidden" alt="processed stream">
      <canvas id="drawCanvas"></canvas>
    </div>
  </section>
</main>

<script>
const $ = id => document.getElementById(id);
const videoSelect = $("videoSelect");
const modelSelect = $("modelSelect");
const modelHint = $("modelHint");
const snapshot = $("snapshot");
const stream = $("stream");
const canvas = $("drawCanvas");
const ctx = canvas.getContext("2d");
const optDensity = $("optDensity");
const optCounting = $("optCounting");
const optTrack = $("optTrack");
const optLogging = $("optLogging");
const densityPanel = $("densityPanel");
const countPanel = $("countPanel");
const statusBox = $("status");

let videoInfo = null;
let modelSpecs = [];
let drawMode = null;
let densityZones = [];
let countLines = [];
let currentZonePoints = [];
let currentLinePoints = [];
let isRunning = false;
let isPaused = false;

function setStatus(text) { statusBox.textContent = text; }

function getPositiveInt(id) {
  const input = $(id);
  const value = Math.floor(Number(input.value));
  const normalized = Number.isFinite(value) && value >= 1 ? value : 1;
  input.value = normalized;
  return normalized;
}

async function sendPlaybackAction(action) {
  const response = await fetch("/api/playback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action }),
  });
  return response.json();
}

function updatePanels() {
  densityPanel.classList.toggle("hidden", !optDensity.checked);
  countPanel.classList.toggle("hidden", !optCounting.checked);
}

async function loadVideos() {
  const videos = await (await fetch("/api/videos")).json();
  videoSelect.innerHTML = videos.map(v => `<option value="${v.path}">${v.name}</option>`).join("");
  if (!videos.length) {
    videoSelect.innerHTML = "";
    setStatus("Chưa có video trong thư mục video/. Hãy thêm file .mp4/.avi/.mov/.mkv/.webm rồi refresh trang.");
  }
}

async function loadModels() {
  modelSpecs = await (await fetch("/api/models")).json();
  modelSelect.innerHTML = modelSpecs.map(m =>
    `<option value="${m.key}" ${m.default ? "selected" : ""} ${m.exists ? "" : "disabled"}>${m.label}${m.exists ? "" : " (missing)"}</option>`
  ).join("");
  updateModelHint();
}

function updateModelHint() {
  const selected = modelSpecs.find(m => m.key === modelSelect.value);
  modelHint.textContent = selected ? `${selected.description}\n${selected.weights}` : "";
}

async function loadFrame() {
  if (!videoSelect.value) {
    setStatus("Chưa có video để load frame.");
    return;
  }
  const video = videoSelect.value;
  videoInfo = await (await fetch(`/api/video-info?video=${encodeURIComponent(video)}`)).json();
  snapshot.src = `/api/frame?video=${encodeURIComponent(video)}&t=${Date.now()}`;
  snapshot.classList.remove("hidden");
  stream.classList.add("hidden");
  canvas.classList.remove("hidden");
  densityZones = [];
  countLines = [];
  currentZonePoints = [];
  currentLinePoints = [];
  drawMode = null;
  snapshot.onload = resizeCanvas;
  setStatus(`Loaded ${videoInfo.name}\n${videoInfo.width}x${videoInfo.height}, ${videoInfo.fps.toFixed(2)} FPS, ${videoInfo.frames} frames`);
}

function resizeCanvas() {
  const image = snapshot.classList.contains("hidden") ? stream : snapshot;
  const rect = image.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width));
  canvas.height = Math.max(1, Math.round(rect.height));
  redrawCanvas();
}

function scaleToVideo(point) {
  return {
    x: Math.round(point.x * videoInfo.width / canvas.width),
    y: Math.round(point.y * videoInfo.height / canvas.height),
  };
}

function scaleFromVideo(point) {
  return {
    x: Math.round(point.x * canvas.width / videoInfo.width),
    y: Math.round(point.y * canvas.height / videoInfo.height),
  };
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
  ctx.lineCap = "round";
  ctx.strokeStyle = color;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  ctx.fillStyle = color;
  for (const point of [a, b]) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.font = "14px Segoe UI";
  ctx.fillText(label, a.x + 6, a.y - 8);
}

function redrawCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  densityZones.forEach((zone, index) => drawPolygon(zone, "#35c28f", `Zone ${index + 1}`));
  countLines.forEach((line, index) => drawLine(line, "#ffcc33", `Line ${index + 1}`));
  if (currentZonePoints.length) drawPolygon({ points: currentZonePoints }, "#8bd9ff", `Draft ${currentZonePoints.length}/4`);
  if (currentLinePoints.length === 1) {
    const p = scaleFromVideo(currentLinePoints[0]);
    ctx.fillStyle = "#8bd9ff";
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
    ctx.fill();
  }
  if (currentLinePoints.length === 2) drawLine({ p1: currentLinePoints[0], p2: currentLinePoints[1] }, "#8bd9ff", "Draft line");
  $("zoneBadges").innerHTML = densityZones.map((z, i) => `<span class="badge">Z${i + 1}: ${z.points.map(p => `(${p.x},${p.y})`).join(" ")}</span>`).join("");
  $("lineBadge").innerHTML = countLines.map((line, i) => `<span class="badge">L${i + 1}: (${line.p1.x},${line.p1.y}) -> (${line.p2.x},${line.p2.y})</span>`).join("");
}

canvas.addEventListener("click", event => {
  if (!videoInfo || !drawMode) return;
  const rect = canvas.getBoundingClientRect();
  const point = scaleToVideo({ x: event.clientX - rect.left, y: event.clientY - rect.top });
  if (drawMode === "zones") {
    if (currentZonePoints.length < 4) currentZonePoints.push(point);
    setStatus(`Vùng hiện tại: ${currentZonePoints.length}/4 điểm.`);
  } else {
    if (currentLinePoints.length < 2) currentLinePoints.push(point);
    setStatus(`Thanh đếm: ${currentLinePoints.length}/2 điểm.`);
  }
  redrawCanvas();
});

$("loadFrame").onclick = loadFrame;
modelSelect.onchange = updateModelHint;
optDensity.onchange = updatePanels;
optCounting.onchange = updatePanels;
$("drawZones").onclick = () => { densityZones = []; currentZonePoints = []; drawMode = "zones"; setStatus("Bấm 4 điểm cho từng density zone."); redrawCanvas(); };
$("okZone").onclick = () => {
  const maxZones = getPositiveInt("zoneCount");
  if (currentZonePoints.length !== 4) return setStatus("Một vùng density cần đúng 4 điểm trước khi OK.");
  if (densityZones.length >= maxZones) return setStatus("Đã đủ số vùng đã chọn.");
  densityZones.push({ points: currentZonePoints });
  currentZonePoints = [];
  drawMode = densityZones.length < maxZones ? "zones" : null;
  setStatus(drawMode ? `Đã lưu vùng ${densityZones.length}. Tiếp tục vẽ vùng kế tiếp.` : "Đã lưu đủ vùng density.");
  redrawCanvas();
};
$("clearZones").onclick = () => { densityZones = []; currentZonePoints = []; drawMode = null; redrawCanvas(); };
$("drawLine").onclick = () => { countLines = []; currentLinePoints = []; drawMode = "line"; setStatus("Bấm 2 điểm cho từng counting line."); redrawCanvas(); };
$("okLine").onclick = () => {
  const maxLines = getPositiveInt("lineCount");
  if (currentLinePoints.length !== 2) return setStatus("Counting line cần đúng 2 điểm trước khi OK.");
  if (countLines.length >= maxLines) return setStatus("Đã đủ số thanh đã chọn.");
  countLines.push({ p1: currentLinePoints[0], p2: currentLinePoints[1] });
  currentLinePoints = [];
  drawMode = countLines.length < maxLines ? "line" : null;
  setStatus(drawMode ? `Đã lưu thanh ${countLines.length}. Tiếp tục vẽ thanh kế tiếp.` : "Đã lưu đủ counting line.");
  redrawCanvas();
};
$("clearLine").onclick = () => { countLines = []; currentLinePoints = []; drawMode = null; redrawCanvas(); };

$("start").onclick = async () => {
  if (!videoSelect.value) return setStatus("Chưa có video để xử lý.");
  if (!videoInfo) await loadFrame();
  if (!videoInfo) return;
  if (optDensity.checked && currentZonePoints.length) return setStatus("Vùng density đang vẽ dở. Hãy bấm OK vùng hoặc Xóa vùng trước khi Start.");
  if (optDensity.checked && densityZones.length !== getPositiveInt("zoneCount")) return setStatus("Cần vẽ đủ density zones rồi mới Start.");
  if (optCounting.checked && currentLinePoints.length) return setStatus("Counting line đang vẽ dở. Hãy bấm OK thanh hoặc Xóa thanh trước khi Start.");
  if (optCounting.checked && countLines.length !== getPositiveInt("lineCount")) return setStatus("Cần vẽ đủ counting lines rồi mới Start.");
  await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      video: videoSelect.value,
      model_key: modelSelect.value,
      options: { density: optDensity.checked, counting: optCounting.checked, track: optTrack.checked },
      density_zones: densityZones,
      count_lines: countLines,
      conf: Number($("conf").value),
      max_box_area_ratio: Number($("maxArea").value),
      enable_logging: optLogging.checked,
      stream_width: Number($("streamWidth").value),
      jpeg_quality: Number($("jpegQuality").value),
    }),
  });
  canvas.classList.add("hidden");
  snapshot.classList.add("hidden");
  stream.classList.remove("hidden");
  stream.src = `/video_feed?t=${Date.now()}`;
  isRunning = true;
  isPaused = false;
  $("pauseToggle").textContent = "Stop";
  setStatus("Đang xử lý realtime stream...");
};

$("pauseToggle").onclick = async () => {
  if (!isRunning) return;
  if (isPaused) {
    await sendPlaybackAction("resume");
    isPaused = false;
    $("pauseToggle").textContent = "Stop";
    setStatus("Tiếp tục xử lý realtime stream...");
  } else {
    await sendPlaybackAction("pause");
    isPaused = true;
    $("pauseToggle").textContent = "Continue";
    setStatus("Đã dừng tạm thời. Frame hiện tại được giữ nguyên để quan sát.");
  }
};

$("finish").onclick = async () => {
  if (isRunning) await sendPlaybackAction("finish");
  stream.src = "";
  stream.classList.add("hidden");
  snapshot.classList.remove("hidden");
  canvas.classList.remove("hidden");
  isRunning = false;
  isPaused = false;
  $("pauseToggle").textContent = "Stop";
  setStatus("Đã kết thúc stream và quay về frame đầu.");
};

window.addEventListener("resize", resizeCanvas);
Promise.all([loadVideos(), loadModels()]).then(loadFrame).then(updatePanels);
</script>
</body>
</html>
"""
