"""Flask app factory for the realtime traffic frontend."""

from __future__ import annotations

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

from realtime_od.config import PROJECT_ROOT, resolve_project_path
from realtime_od.model_registry import DEFAULT_MODEL_KEY, MODEL_REGISTRY
from realtime_od.realtime_state import RealtimeState
from realtime_od.realtime_stream import process_stream, read_video_info
from realtime_od.realtime_template import INDEX_HTML


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def create_app(state: RealtimeState | None = None) -> Flask:
    app_state = state or RealtimeState()
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

    @app.get("/api/models")
    def models() -> Response:
        items = []
        for model in MODEL_REGISTRY.values():
            weights_path = resolve_project_path(model.weights)
            items.append(
                {
                    "key": model.key,
                    "label": model.label,
                    "weights": model.weights,
                    "description": model.description,
                    "exists": weights_path.exists(),
                    "default": model.key == DEFAULT_MODEL_KEY,
                }
            )
        return jsonify(items)

    @app.get("/api/video-info")
    def video_info() -> Response:
        video_path = resolve_project_path(request.args["video"])
        return jsonify(read_video_info(video_path))

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
        app_state.set_config(payload)
        return jsonify({"ok": True})

    @app.get("/video_feed")
    def video_feed() -> Response:
        return Response(process_stream(app_state), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app
