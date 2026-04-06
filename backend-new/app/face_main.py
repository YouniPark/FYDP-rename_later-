from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from app.face_contracts import FaceDebugIdentity, FaceDebugResponse, LatestFaceDecisionResponse, VideoFrameWsMessage
from app.face_service.settings import face_service_settings
from app.face_service.state import FaceFrameEnvelope, FaceServiceState

settings = face_service_settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("app.face_main")

state = FaceServiceState(settings)

class DebugHub:
    def __init__(self) -> None:
        self.connections: set[WebSocket] = set()
        self.lock = asyncio.Lock()
        self.latest_payload: dict[str, Any] | None = None

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self.lock:
            self.connections.add(ws)
            latest_payload = self.latest_payload
        if latest_payload is not None:
            await ws.send_json(latest_payload)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self.lock:
            self.connections.discard(ws)

    async def publish(self, payload: dict[str, Any]) -> None:
        async with self.lock:
            self.latest_payload = payload
            targets = list(self.connections)

        stale: list[WebSocket] = []
        for conn in targets:
            try:
                await conn.send_json(payload)
            except Exception:
                stale.append(conn)

        for conn in stale:
            await self.disconnect(conn)


debug_hub = DebugHub()

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

def _encode_jpeg_b64(frame_bgr: np.ndarray) -> str | None:
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _map_name_to_people_id(name: str | None) -> int | None:
    if name is None:
        return None
    if name == "Unknown":
        return 0
    # KNN backend returns person names; resolve via the CSV mapping
    mapped = state.name_to_people_id.get(name)
    if mapped is not None:
        return mapped
    # ArcFace backend returns face_ids directly (e.g. "1", "2")
    try:
        return int(name)
    except (ValueError, TypeError):
        return None


async def _face_worker_loop() -> None:
    min_interval = 1.0 / settings.inference_sample_fps
    last_infer_time = 0.0

    logger.info(
        "Starting face worker",
        extra={
            "sample_fps": settings.inference_sample_fps,
            "memory_window": settings.memory_window_seconds,
            "recognizer_backend": settings.recognizer_backend,
            "mapping_csv": str(settings.mapping_csv_path),
            "recognizer_model": str(getattr(state.recognizer, "recognizer_model_path", "N/A")),
            "detector_prototxt": str(getattr(state.recognizer, "detector_prototxt_path", "N/A")),
            "detector_caffemodel": str(getattr(state.recognizer, "detector_caffemodel_path", "N/A")),
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

            debug_response = _build_debug_response(decision)
            await debug_hub.publish(
                {
                    "type": "face_debug",
                    "payload": debug_response.model_dump(mode="json"),
                    "frame_jpeg_b64": _encode_jpeg_b64(frame),
                    "frame_timestamp": envelope.timestamp,
                }
            )

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


def _load_people_by_id() -> dict[int, dict[str, object]]:
    people_path = settings.people_json_path
    if not people_path.exists():
        return {}
    payload = json.loads(people_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return {}

    people_by_id: dict[int, dict[str, object]] = {}
    for person in payload:
        if not isinstance(person, dict):
            continue
        person_id = person.get("id")
        if not isinstance(person_id, int):
            continue
        people_by_id[person_id] = person
    return people_by_id


def _build_debug_response(latest: LatestFaceDecisionResponse) -> FaceDebugResponse:
    dnn_face_id = latest.people_id if latest.people_id and latest.people_id > 0 else None

    recognized_identity = None
    if dnn_face_id is not None:
        person = _load_people_by_id().get(dnn_face_id)
        if person is not None:
            recognized_identity = FaceDebugIdentity(
                id=dnn_face_id,
                name=person.get("name"),
                relationship=person.get("relationship"),
            )

    return FaceDebugResponse(
        dnn_name=latest.name,
        dnn_face_id=dnn_face_id,
        recognized_identity=recognized_identity,
        confidence=latest.confidence,
        decided_at=latest.decided_at,
        sample_count=latest.sample_count,
        window_seconds=latest.window_seconds,
        is_unknown=latest.is_unknown,
    )


@app.get("/face/debug/latest", response_model=FaceDebugResponse)
async def face_debug_latest() -> FaceDebugResponse:
    latest = await state.get_latest_decision()
    return _build_debug_response(latest)


@app.websocket("/ws/face-debug")
async def ws_face_debug(ws: WebSocket) -> None:
    await debug_hub.connect(ws)
    try:
        while True:
            _ = await ws.receive_text()
    except WebSocketDisconnect:
        await debug_hub.disconnect(ws)


@app.get("/face/debug/ui")
async def face_debug_ui() -> HTMLResponse:
    return HTMLResponse(
        """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Face Debug UI</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 16px; }
    .row { display: flex; gap: 20px; align-items: flex-start; }
    img { width: 480px; height: auto; border: 1px solid #ccc; background: #f5f5f5; }
    .panel { min-width: 320px; }
    .label { font-weight: bold; }
    .kv { margin: 8px 0; }
  </style>
</head>
<body>
  <h2>Live Face Recognition Debug</h2>
  <div class="row">
    <img id="frame" alt="live frame" />
    <div class="panel">
      <div class="kv"><span class="label">DNN Name:</span> <span id="dnn_name">-</span></div>
      <div class="kv"><span class="label">DNN Face ID:</span> <span id="dnn_face_id">-</span></div>
      <div class="kv"><span class="label">Identity Name (people.json):</span> <span id="identity_name">-</span></div>
      <div class="kv"><span class="label">Relationship:</span> <span id="relationship">-</span></div>
      <div class="kv"><span class="label">Confidence:</span> <span id="confidence">-</span></div>
      <div class="kv"><span class="label">Unknown:</span> <span id="is_unknown">-</span></div>
      <div class="kv"><span class="label">Decided At:</span> <span id="decided_at">-</span></div>
    </div>
  </div>
  <script>
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/face-debug`);
    ws.onmessage = (evt) => {
      const message = JSON.parse(evt.data);
      if (message.type !== "face_debug") return;
      const payload = message.payload || {};
      const identity = payload.recognized_identity || {};
      document.getElementById("dnn_name").textContent = payload.dnn_name ?? "-";
      document.getElementById("dnn_face_id").textContent = payload.dnn_face_id ?? "-";
      document.getElementById("identity_name").textContent = identity.name ?? "-";
      document.getElementById("relationship").textContent = identity.relationship ?? "-";
      document.getElementById("confidence").textContent = payload.confidence ?? "-";
      document.getElementById("is_unknown").textContent = payload.is_unknown ?? "-";
      document.getElementById("decided_at").textContent = payload.decided_at ?? "-";
      if (message.frame_jpeg_b64) {
        document.getElementById("frame").src = `data:image/jpeg;base64,${message.frame_jpeg_b64}`;
      }
    };
  </script>
</body>
</html>
        """
    )

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
