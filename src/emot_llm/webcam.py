"""Webcam capture helpers."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class CapturedFrame:
    jpeg_b64: str
    path: str | None = None
    width: int | None = None
    height: int | None = None


def capture_frame(camera_index: int = 0, save_dir: str | Path | None = None, prefix: str = "tick") -> CapturedFrame:
    """Capture one webcam frame and return base64 JPEG.

    Raises RuntimeError with a clear message if OpenCV/camera access fails.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("opencv-python is not installed. Run: pip install opencv-python") from exc

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open webcam index {camera_index}. Disable --webcam or check camera permissions.")
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not capture a frame from webcam index {camera_index}.")

    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("OpenCV failed to encode webcam frame as JPEG.")
    raw = bytes(buf)
    path: str | None = None
    if save_dir is not None:
        directory = Path(save_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        out = directory / f"{prefix}_{stamp}.jpg"
        out.write_bytes(raw)
        path = str(out)
    height, width = frame.shape[:2]
    return CapturedFrame(jpeg_b64=base64.b64encode(raw).decode("ascii"), path=path, width=width, height=height)
