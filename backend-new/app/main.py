import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.cue_service import build_cue_decision
from app.eeg_pipeline import eeg_connect_loop, run_eeg_event_pipeline
from app.face_pipeline import enqueue_frame, face_recognition_loop
from app.event_inlet_pipeline import event_inlet_loop
from app.state import AppState
from app.storage.models import CueDBManifest, EventIn, FaceDBManifest, VideoFrameMessage
from app.face_memory import face_memory_voter, FaceVoteResult
from app.face_contracts import FaceDebugResponse
from app.face_debug import FaceDebugHub, face_debug_hub, build_face_debug_response

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("app.main")

state = AppState(settings)


class WebSocketHub:
    def __init__(self) -> None:
        self.connections: set[WebSocket] = set()
        self.lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self.lock:
            self.connections.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self.lock:
            if ws in self.connections:
                self.connections.remove(ws)

    async def broadcast_json(self, payload: dict[str, Any]) -> None:
        stale: list[WebSocket] = []
        async with self.lock:
            targets = list(self.connections)
        for conn in targets:
            try:
                await conn.send_json(payload)
            except Exception:
                stale.append(conn)
        for conn in stale:
            await self.disconnect(conn)


hub = WebSocketHub()


@asynccontextmanager
async def lifespan(_: FastAPI):
    async def dispatch_fixation(event_id: str, lsl_timestamp: float, _proxy_name: str) -> None:
        """Dispatch a fixation event received from LSL through the EEG pipeline."""
        result = await run_eeg_event_pipeline(state, event_id, lsl_timestamp)
        if "is_unfamiliar" not in result:
            logger.info(
                "Event %s not forwarded to cue service (status=%s)",
                event_id, result.get("status"),
            )
            return
        if result.get("status") != "ok":
            logger.info(
                "Event %s proceeding to cue service despite non-ok status=%s (is_unfamiliar=%s)",
                event_id, result.get("status"), result["is_unfamiliar"],
            )
        # decision = await build_cue_decision(
        #     state,
        #     event_id=event_id,
        #     event_lsl_timestamp=lsl_timestamp,
        #     is_unfamiliar=result["is_unfamiliar"],
        # )
        voted_face = await face_memory_voter.get_voted_face(lsl_timestamp)
        logger.info(
            "Event %s: EEG result is_unfamiliar=%s  voted_face=%s — building cue decision",
            event_id, result["is_unfamiliar"], voted_face,
        )

        decision = await build_cue_decision(
            state,
            event_id=event_id,
            event_lsl_timestamp=lsl_timestamp,
            is_unfamiliar=result["is_unfamiliar"],
            face_id=voted_face,
        )

        decision_json = decision.model_dump(mode="json")
        logger.info(
            "Event %s: cue decision — send_cue=%s  face_id=%s  is_unfamiliar=%s",
            event_id, decision.send_cue, decision.face_id, decision.is_unfamiliar,
        )
        state.latest_cue_decision_json = decision_json
        await hub.broadcast_json({"type": "cue_decision", "payload": decision_json})

    async def _face_debug_publish(vote: FaceVoteResult, frame_b64: str, timestamp: float) -> None:
        debug_response = build_face_debug_response(vote)
        await face_debug_hub.publish({
            "type": "face_debug",
            "payload": debug_response.model_dump(mode="json"),
            "frame_jpeg_b64": frame_b64,
            "frame_timestamp": timestamp,
        })

    eeg_task = asyncio.create_task(eeg_connect_loop(state))
    face_task = asyncio.create_task(face_recognition_loop(state, debug_publish=_face_debug_publish))
    event_lsl_task = asyncio.create_task(event_inlet_loop(state, dispatch_fixation))
    try:
        yield
    finally:
        eeg_task.cancel()
        face_task.cancel()
        event_lsl_task.cancel()
        await asyncio.gather(eeg_task, face_task, event_lsl_task, return_exceptions=True)


app = FastAPI(title="ADAD Python Backend Server", lifespan=lifespan)


# Create public URL for image and audio cues 
app.mount(
    "/media/images",
    StaticFiles(directory=settings.images_dir),
    name="images",
)

app.mount(
    "/media/audio",
    StaticFiles(directory=settings.auditory_cue_dir),
    name="audio",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/events")
async def post_event(event: EventIn) -> dict[str, Any]:
    result = await run_eeg_event_pipeline(state, event.event_id, event.event_lsl_timestamp)
    if "is_unfamiliar" not in result:
        return result
    # decision = await build_cue_decision(
    #     state,
    #     event_id=event.event_id,
    #     event_lsl_timestamp=event.event_lsl_timestamp,
    #     is_unfamiliar=result["is_unfamiliar"],
    # )
    voted_face = await face_memory_voter.get_voted_face(event.event_lsl_timestamp)
    logger.info(
        "Event %s: EEG result is_unfamiliar=%s  voted_face=%s — building cue decision",
        event.event_id, result["is_unfamiliar"], voted_face,
    )

    decision = await build_cue_decision(
        state,
        event_id=event.event_id,
        event_lsl_timestamp=event.event_lsl_timestamp,
        is_unfamiliar=result["is_unfamiliar"],
        face_id=voted_face,
    )

    payload = decision.model_dump(mode="json")
    state.latest_cue_decision_json = payload
    await hub.broadcast_json({"type": "cue_decision", "payload": payload})
    return payload


@app.get("/cue/latest")
async def get_cue_latest() -> dict[str, Any]:
    """Return the most recent cue decision produced by the EEG/face pipeline.

    Returns 404 when no decision has been produced yet (e.g. backend just started).
    Polling clients (AR app) should call this after a fixation event and keep
    retrying until they receive a decision whose server_time is newer than the
    fixation timestamp.
    """
    if state.latest_cue_decision_json is None:
        raise HTTPException(status_code=404, detail="No cue decision available yet")
    return state.latest_cue_decision_json


@app.websocket("/ws/ar")
async def ws_ar(ws: WebSocket) -> None:
    await hub.connect(ws)
    try:
        async with state.face_db_lock:
            face_manifest = FaceDBManifest(faces=list(state.face_db.values())).model_dump(mode="json")
        async with state.cue_db_lock:
            cue_manifest = CueDBManifest(cues=list(state.cue_db.values())).model_dump(mode="json")
        await ws.send_json({"type": "db_sync", "payload": {"face_db": face_manifest, "cue_db": cue_manifest}})
        while True:
            message = await ws.receive_text()
            if message == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        await hub.disconnect(ws)


@app.websocket("/ws/video")
async def ws_video(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            payload = await ws.receive_json()
            frame = VideoFrameMessage.model_validate(payload)
            await enqueue_frame(state, frame.timestamp, frame.data_b64, frame.encoding)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await ws.close(code=1011, reason=str(exc))


@app.post("/db/face")
async def upload_face(face_id: str = Form(...), metadata_json: str = Form("{}"), image: UploadFile = File(...)) -> dict[str, Any]:
    raw = await image.read()
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Face image too large")
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="metadata_json must be valid JSON") from exc

    ext = image.filename.split(".")[-1] if image.filename and "." in image.filename else "jpg"
    image_path = state.db.store_face_image(face_id, raw, ext)
    async with state.face_db_lock:
        record = state.db.upsert_face_record(state.face_db, face_id, metadata, image_path)

    # Auto-enroll for ArcFace backend using the uploaded image bytes
    enrolled = False
    try:
        from user_modules.face import enroll_arcface_from_image_bytes
        enrolled = await asyncio.to_thread(enroll_arcface_from_image_bytes, face_id, raw)
    except Exception:
        logger.exception(
            "ArcFace auto-enrollment failed for face_id=%s (face record was saved; "
            "retry enrollment via POST /face/enroll/%s)",
            face_id,
            face_id,
        )

    return {"status": "ok", "record": record.model_dump(mode="json"), "arcface_enrolled": enrolled}


@app.post("/db/cue")
async def upload_cue(face_id: str = Form(...), cue_json: str = Form(...)) -> dict[str, Any]:
    try:
        cue = json.loads(cue_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="cue_json must be valid JSON") from exc

    async with state.cue_db_lock:
        record = state.db.upsert_cue_record(state.cue_db, face_id, cue)
    return {"status": "ok", "record": record.model_dump(mode="json")}


@app.get("/db/face")
async def get_face_manifest() -> dict[str, Any]:
    async with state.face_db_lock:
        manifest = FaceDBManifest(faces=list(state.face_db.values())).model_dump(mode="json")
    return manifest


@app.get("/db/cue")
async def get_cue_manifest() -> dict[str, Any]:
    async with state.cue_db_lock:
        manifest = CueDBManifest(cues=list(state.cue_db.values())).model_dump(mode="json")
    return manifest


@app.get("/face/debug/latest", response_model=FaceDebugResponse)
async def face_debug_latest() -> FaceDebugResponse:
    vote = await face_memory_voter.get_latest_vote()
    if vote is None:
        raise HTTPException(status_code=404, detail="No face decision available yet")
    return build_face_debug_response(vote)


@app.websocket("/ws/face-debug")
async def ws_face_debug(ws: WebSocket) -> None:
    await face_debug_hub.connect(ws)
    try:
        while True:
            _ = await ws.receive_text()
    except WebSocketDisconnect:
        await face_debug_hub.disconnect(ws)


@app.get("/face/debug/ui")
async def face_debug_ui() -> HTMLResponse:
    return HTMLResponse("""
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
      <div class="kv"><span class="label">Face ID:</span> <span id="dnn_name">-</span></div>
      <div class="kv"><span class="label">Numeric ID:</span> <span id="dnn_face_id">-</span></div>
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
    """)


@app.get("/db/file")
async def get_data_file(path: str) -> FileResponse:
    try:
        target = state.db.resolve_data_file(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)


# ------------------------------------------------------------------
# ArcFace enrollment endpoints
# ------------------------------------------------------------------

@app.post("/face/enroll/{face_id}")
async def enroll_face(face_id: str) -> dict[str, Any]:
    """(Re-)enroll a face for ArcFace recognition using its stored image.

    Looks up *face_id* in the face database, loads the saved enrollment image
    from disk, and extracts a fresh ArcFace embedding.

    Returns 404 when *face_id* is not in the database or has no stored image.
    Returns 422 when the image cannot be decoded or no face is detected.
    Returns 400 when the KNN backend is active (enrollment not applicable).
    """
    from user_modules.face import enroll_arcface_from_image_path
    from app.face_service.settings import face_service_settings

    if face_service_settings.recognizer_backend.lower() != "arcface":
        raise HTTPException(
            status_code=400,
            detail="Enrollment endpoint is only available when FACE_RECOGNIZER_BACKEND=arcface",
        )

    async with state.face_db_lock:
        record = state.face_db.get(face_id)

    if record is None:
        raise HTTPException(status_code=404, detail=f"face_id '{face_id}' not found in database")

    if not record.image_path:
        raise HTTPException(
            status_code=422,
            detail=f"face_id '{face_id}' has no stored image — upload an image first via POST /db/face",
        )

    abs_image_path = str(state.db.resolve_data_file(record.image_path))
    ok = await asyncio.to_thread(enroll_arcface_from_image_path, face_id, abs_image_path)
    if not ok:
        raise HTTPException(
            status_code=422,
            detail=f"No face detected in the stored image for face_id '{face_id}'",
        )

    return {"status": "ok", "face_id": face_id, "image_path": record.image_path}


# ------------------------------------------------------------------
# Performance benchmark endpoint
# ------------------------------------------------------------------

@app.get("/face/benchmark")
async def face_benchmark() -> dict[str, Any]:
    """Compare KNN and ArcFace recognition on every enrolled face image.

    For each face record that has a stored image the endpoint runs both
    recognisers and records whether each one correctly identified the person,
    its confidence score, and its per-frame latency.

    The active backend (``FACE_RECOGNIZER_BACKEND``) is shown in the response
    but both models are evaluated regardless so you can compare them.

    Note: this endpoint is CPU-intensive and may take several seconds.
    """
    import cv2 as _cv2
    from app.face_service.settings import face_service_settings
    from app.face_service.recognizer_runtime import FaceRuntimeRecognizer

    async with state.face_db_lock:
        face_db_snapshot = dict(state.face_db)

    results: list[dict[str, Any]] = []
    knn_correct = 0
    arcface_correct = 0
    knn_total = 0
    arcface_total = 0

    # Lazily initialise both recognisers inside the thread so we don't block
    # the event loop during model loading
    def _run_benchmark() -> list[dict[str, Any]]:
        nonlocal knn_correct, arcface_correct, knn_total, arcface_total

        # -- KNN recogniser --
        knn_rec = None
        knn_error: str | None = None
        try:
            knn_rec = FaceRuntimeRecognizer(face_service_settings)
        except Exception as exc:
            knn_error = str(exc)

        # -- ArcFace recogniser --
        arcface_rec = None
        arcface_error: str | None = None
        try:
            from app.face_service.arcface_recognizer import ArcFaceRuntimeRecognizer
            from app.face_service.embedding_store import ArcFaceEmbeddingStore

            store = ArcFaceEmbeddingStore(face_service_settings.arcface_embedding_store_path)
            arcface_rec = ArcFaceRuntimeRecognizer(face_service_settings, store)
        except Exception as exc:
            arcface_error = str(exc)

        per_face: list[dict[str, Any]] = []

        # Pre-build the face_db_view once (used by KNN name→face_id resolution)
        from user_modules.face import _resolve_face_id
        face_db_view = {k: v.model_dump(mode="json") for k, v in face_db_snapshot.items()}

        for face_id, record in face_db_snapshot.items():
            if not record.image_path:
                per_face.append({
                    "face_id": face_id,
                    "name": record.metadata.get("name"),
                    "skipped": True,
                    "reason": "no stored image",
                })
                continue

            try:
                abs_path = str(state.db.resolve_data_file(record.image_path))
            except ValueError:
                per_face.append({
                    "face_id": face_id,
                    "name": record.metadata.get("name"),
                    "skipped": True,
                    "reason": "invalid image path",
                })
                continue

            frame = _cv2.imread(abs_path)
            if frame is None:
                per_face.append({
                    "face_id": face_id,
                    "name": record.metadata.get("name"),
                    "skipped": True,
                    "reason": "image could not be read",
                })
                continue

            entry: dict[str, Any] = {
                "face_id": face_id,
                "name": record.metadata.get("name"),
                "image_path": record.image_path,
            }

            # KNN
            if knn_rec is not None:
                t0 = time.perf_counter()
                pred = knn_rec.predict_frame(frame)
                latency_ms = (time.perf_counter() - t0) * 1000

                resolved = None
                if pred.name and pred.name != "Unknown":
                    resolved = _resolve_face_id(pred.name, face_db_view)

                correct = resolved == face_id
                knn_total += 1
                if correct:
                    knn_correct += 1

                entry["knn"] = {
                    "predicted_name": pred.name,
                    "predicted_face_id": resolved,
                    "confidence": round(pred.confidence, 4) if pred.confidence is not None else None,
                    "latency_ms": round(latency_ms, 1),
                    "correct": correct,
                }
            else:
                entry["knn"] = {"error": knn_error}

            # ArcFace
            if arcface_rec is not None:
                t0 = time.perf_counter()
                pred = arcface_rec.predict_frame(frame)
                latency_ms = (time.perf_counter() - t0) * 1000

                predicted_face_id = pred.name if pred.name not in (None, "Unknown") else None
                correct = predicted_face_id == face_id
                arcface_total += 1
                if correct:
                    arcface_correct += 1

                entry["arcface"] = {
                    "predicted_face_id": predicted_face_id,
                    "confidence": round(pred.confidence, 4) if pred.confidence is not None else None,
                    "latency_ms": round(latency_ms, 1),
                    "correct": correct,
                }
            else:
                entry["arcface"] = {"error": arcface_error}

            per_face.append(entry)

        return per_face

    results = await asyncio.to_thread(_run_benchmark)

    return {
        "active_backend": face_service_settings.recognizer_backend,
        "enrolled_faces": len(face_db_snapshot),
        "knn_accuracy": round(knn_correct / knn_total, 4) if knn_total > 0 else None,
        "arcface_accuracy": round(arcface_correct / arcface_total, 4) if arcface_total > 0 else None,
        "results": results,
    }
