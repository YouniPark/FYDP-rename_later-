"""
webcam_feeder.py
================
Captures frames from the local webcam and feeds them into ``AppState.frame_queue``
so the existing ``face_recognition_loop`` can process them without modification.

This replaces the WebSocket ``/ws/video`` endpoint (video frames from the AR headset).

Configuration (environment variables or .env):
  WEBCAM_DEVICE_INDEX=0    OpenCV capture device index (default: 0).
  WEBCAM_FPS=15            Target capture framerate (default: 15).
  WEBCAM_JPEG_QUALITY=85   JPEG encoding quality 1-100 (default: 85).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import cv2

logger = logging.getLogger(__name__)

_DEVICE_INDEX = int(os.environ.get("WEBCAM_DEVICE_INDEX", "0"))
_FPS = float(os.environ.get("WEBCAM_FPS", "15"))
_JPEG_QUALITY = int(os.environ.get("WEBCAM_JPEG_QUALITY", "85"))
_FRAME_INTERVAL = 1.0 / max(_FPS, 1.0)


def _open_capture(device_index: int) -> "cv2.VideoCapture | None":
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        return None
    return cap


def _encode_jpeg(frame, quality: int) -> "bytes | None":
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return buf.tobytes()


async def webcam_capture_loop(state) -> None:
    """Background task: continuously captures webcam frames and enqueues them.

    Runs until cancelled.  Logs an error and returns gracefully if the webcam
    cannot be opened.
    """
    cap: "cv2.VideoCapture | None" = await asyncio.to_thread(_open_capture, _DEVICE_INDEX)
    if cap is None:
        logger.error(
            "Webcam: could not open device %d — face recognition will not receive frames. "
            "Check that your webcam is connected and not in use by another application.",
            _DEVICE_INDEX,
        )
        print(
            f"\n[TEST MODE] WARNING: webcam device {_DEVICE_INDEX} could not be opened.\n"
            f"  Face recognition will be idle.\n"
            f"  Set WEBCAM_DEVICE_INDEX= in .env to select a different device.\n"
        )
        return

    logger.info("Webcam: opened device %d at target %.0f fps", _DEVICE_INDEX, _FPS)
    print(
        f"\n[TEST MODE] Webcam active  (device={_DEVICE_INDEX}, fps={_FPS:.0f}, "
        f"jpeg_quality={_JPEG_QUALITY})\n"
    )

    try:
        while True:
            frame_start = time.monotonic()

            ret, frame = await asyncio.to_thread(cap.read)
            if not ret or frame is None:
                logger.warning("Webcam: frame capture failed — retrying in 100 ms")
                await asyncio.sleep(0.1)
                continue

            jpeg_bytes: "bytes | None" = await asyncio.to_thread(_encode_jpeg, frame, _JPEG_QUALITY)
            if jpeg_bytes is None:
                logger.warning("Webcam: JPEG encode failed — skipping frame")
                continue

            ts = time.time()

            # Drop the oldest frame when the queue is full to avoid head-of-line blocking.
            if state.frame_queue.full():
                try:
                    state.frame_queue.get_nowait()
                except Exception:
                    pass

            await state.frame_queue.put((ts, jpeg_bytes))

            # Pace to the target framerate.
            elapsed = time.monotonic() - frame_start
            sleep_time = _FRAME_INTERVAL - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except asyncio.CancelledError:
        logger.info("Webcam: capture loop cancelled")
        raise
    finally:
        cap.release()
        logger.info("Webcam: capture device released")
