from __future__ import annotations

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.face_contracts import LatestFaceDecisionResponse, VideoFrameWsMessage
from app.face_service.settings import face_service_settings
from app.face_service.state import FaceFrameEnvelope, FaceServiceState

settings = face_service_settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("app.face_main")

state = FaceServiceState(settings)


async def _enqueue_frame(frame: VideoFrameWsMessage) -> None:
    try:
        jpeg_bytes = base64.b64decode(frame.data_b64, validate=True)
    except Exception as exc:
        raise ValueError("Malformed base64 frame data") from exc

    if state.frame_queue.full():
        _ = state.frame_queue.get_nowait()

    await state.frame_queue.put(FaceFrameEnvelope(timestamp=frame.timestamp, jpeg_bytes=jpeg_bytes))


def _decode_jpeg(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid jpeg frame")
    return frame


def _map_name_to_people_id(name: str | None) -> int | None:
    if name is None:
        return None
    if name == "Unknown":
        return 0
    return state.name_to_people_id.get(name)


async def _face_worker_loop() -> None:
    min_interval = 1.0 / settings.inference_sample_fps
    last_infer_time = 0.0

    logger.info(
        "Starting face worker",
        extra={
            "sample_fps": settings.inference_sample_fps,
            "memory_window": settings.memory_window_seconds,
            "mapping_csv": str(settings.mapping_csv_path),
            "recognizer_model": str(settings.recognizer_model_path),
            "detector_prototxt": str(state.recognizer.detector_prototxt_path),
            "detector_caffemodel": str(state.recognizer.detector_caffemodel_path),
        },
    )

    while True:
        envelope = await state.frame_queue.get()
        try:
            # Always process the freshest frame available.
            while not state.frame_queue.empty():
                envelope = state.frame_queue.get_nowait()

            elapsed = time.monotonic() - last_infer_time
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)

            frame = await asyncio.to_thread(_decode_jpeg, envelope.jpeg_bytes)
            prediction = await asyncio.to_thread(state.recognizer.predict_frame, frame)

            vote = state.temporal_memory.add_detection(
                label=prediction.name,
                confidence=prediction.confidence,
                observed_ts=envelope.timestamp,
            )

            people_id = _map_name_to_people_id(vote.label)
            is_unknown = vote.label == "Unknown"

            decision = LatestFaceDecisionResponse(
                name=vote.label,
                people_id=people_id,
                confidence=vote.confidence,
                decided_at=vote.decided_at,
                source="memory_vote",
                window_seconds=settings.memory_window_seconds,
                sample_count=vote.sample_count,
                is_unknown=is_unknown,
            )
            await state.set_latest_decision(decision)

            state.last_frame_timestamp = envelope.timestamp
            state.processed_frames += 1
            last_infer_time = time.monotonic()
        except Exception:
            logger.exception("Face worker processing failed")
        finally:
            state.frame_queue.task_done()


@asynccontextmanager
async def lifespan(_: FastAPI):
    worker = asyncio.create_task(_face_worker_loop())
    try:
        yield
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


app = FastAPI(title="ADAD Face Stream Service", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/face/latest", response_model=LatestFaceDecisionResponse)
async def face_latest() -> LatestFaceDecisionResponse:
    return await state.get_latest_decision()


@app.get("/face/diagnostics")
async def face_diagnostics() -> dict[str, object]:
    latest = await state.get_latest_decision()
    now = datetime.now(tz=timezone.utc)

    age_seconds = None
    if latest.decided_at is not None:
        age_seconds = (now - latest.decided_at).total_seconds()

    return {
        "queue_depth": state.frame_queue.qsize(),
        "processed_frames": state.processed_frames,
        "last_frame_timestamp": state.last_frame_timestamp,
        "latest_decision_age_seconds": age_seconds,
        "sample_fps": settings.inference_sample_fps,
        "memory_window_seconds": settings.memory_window_seconds,
    }


@app.websocket("/ws/video")
async def ws_video(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            payload = await ws.receive_json()
            frame = VideoFrameWsMessage.model_validate(payload)
            await _enqueue_frame(frame)
    except WebSocketDisconnect:
        return
    except ValueError as exc:
        await ws.close(code=1003, reason=str(exc))
    except Exception as exc:
        await ws.close(code=1011, reason=str(exc))
