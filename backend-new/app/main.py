import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from app.config import settings
from app.cue_service import build_cue_decision
from app.eeg_pipeline import eeg_connect_loop, run_eeg_event_pipeline
from app.face_pipeline import enqueue_frame, face_recognition_loop
from app.event_inlet_pipeline import event_inlet_loop
from app.state import AppState
from app.storage.models import CueDBManifest, EventIn, FaceDBManifest, VideoFrameMessage

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
        if result.get("status") != "ok":
            logger.info(
                "Event %s not processed (status=%s)",
                event_id, result.get("status"),
            )
            return
        decision = await build_cue_decision(
            state,
            event_id=event_id,
            event_lsl_timestamp=lsl_timestamp,
            is_unfamiliar=result["is_unfamiliar"],
        )
        await hub.broadcast_json({"type": "cue_decision", "payload": decision.model_dump(mode="json")})

    eeg_task = asyncio.create_task(eeg_connect_loop(state))
    face_task = asyncio.create_task(face_recognition_loop(state))
    event_lsl_task = asyncio.create_task(event_inlet_loop(state, dispatch_fixation))
    try:
        yield
    finally:
        eeg_task.cancel()
        face_task.cancel()
        event_lsl_task.cancel()
        await asyncio.gather(eeg_task, face_task, event_lsl_task, return_exceptions=True)


app = FastAPI(title="ADAD Python Backend Server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/events")
async def post_event(event: EventIn) -> dict[str, Any]:
    result = await run_eeg_event_pipeline(state, event.event_id, event.event_lsl_timestamp)
    if result.get("status") != "ok":
        return result
    decision = await build_cue_decision(
        state,
        event_id=event.event_id,
        event_lsl_timestamp=event.event_lsl_timestamp,
        is_unfamiliar=result["is_unfamiliar"],
    )
    payload = decision.model_dump(mode="json")
    await hub.broadcast_json({"type": "cue_decision", "payload": payload})
    return payload


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
    return {"status": "ok", "record": record.model_dump(mode="json")}


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


@app.get("/db/file")
async def get_data_file(path: str) -> FileResponse:
    try:
        target = state.db.resolve_data_file(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)
