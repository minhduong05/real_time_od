"""Video IO helpers."""

from __future__ import annotations

from pathlib import Path

import cv2


def open_video(source: str | int):
    """Open a video file, camera index, or stream URL."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video source: {source}")
    return capture


def make_video_writer(output_path: str | Path, fps: float, width: int, height: int):
    """Create an OpenCV writer using a codec compatible with local demos."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    codec = "mp4v" if suffix == ".mp4" else "XVID"
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {output_path}")
    return writer
