from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.config import APP_NAME, DB_PATH, LOG_LEVEL
from app.cue_service import CueService
from app.db import Database
from app.eeg_service import EEGService
from app.face_service import FaceService
from app.logging_config import configure_logging
from app.models import (
    CueInfoRequest,
    DBName,
    DBSnapshotResponse,
    EventTimingRequest,
    FaceInputRequest,
    HealthResponse,
    PullDBRequest,
    WSIncomingMessage,
)
from app.state import AppStateCache
from app.ws_manager import WebSocketManager

configure_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(title=APP_NAME)
db = Database(DB_PATH)
state_cache = AppStateCache()
eeg_service = EEGService(db, state_cache)
face_service = FaceService(db, state_cache)
cue_service = CueService(db, state_cache)
ws_manager = WebSocketManager()


@app.on_event("startup")
async def startup() -> None:
    app.state.eeg_retry_task = asyncio.create_task(eeg_service.retry_connect_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    task = getattr(app.state, "eeg_retry_task", None)
    if task:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def send_db_snapshot(websocket: WebSocket, db_name: DBName) -> None:
    if db_name == DBName.faces:
        records, last_updated, version = db.list_faces()
    else:
        records, last_updated, version = db.list_cues()
    await ws_manager.send_json(
        websocket,
        DBSnapshotResponse(db=db_name, version=version, last_updated=last_updated, records=records).model_dump() | {"type": "db_snapshot"},
    )


async def process_event_timing(event_lsl_timestamp: float) -> dict:
    result = await eeg_service.handle_event(event_lsl_timestamp)
    if result["status"] == "ignored":
        return {"type": "event_ignored", "reason": result["reason"], "event_lsl_timestamp": event_lsl_timestamp}
    if result["status"] == "error":
        return {"type": "error", "where": "event_timing", "detail": result["reason"]}

    cue_payload = await cue_service.build_cue(classifier_flag=result["is_unfamiliar"])
    await ws_manager.broadcast({"type": "cue", "timestamp": event_lsl_timestamp, "payload": cue_payload})
    return {"status": "ok", "is_unfamiliar": result["is_unfamiliar"], "cue": cue_payload}


async def process_face_input(payload: FaceInputRequest) -> dict:
    face_result = await face_service.handle_face_input(payload.timestamp, payload.image_base64, payload.face_json)
    cue_payload = await cue_service.build_cue(face_result=face_result)
    await ws_manager.broadcast({"type": "cue", "timestamp": payload.timestamp, "payload": cue_payload})
    return {"status": "ok", "face_result": face_result, "cue": cue_payload}


async def process_cue_info(payload: CueInfoRequest) -> dict:
    state_cache.set_cue_json(payload.timestamp, payload.cue_json)
    version = db.add_or_update_cue(payload.cue_json, key=payload.cue_json.get("id"))
    await ws_manager.broadcast({"type": "db_updated", "db": "cues", "version": version})
    cue_payload = await cue_service.build_cue(cue_json=payload.cue_json)
    await ws_manager.broadcast({"type": "cue", "timestamp": payload.timestamp, "payload": cue_payload})
    return {"status": "ok", "cue": cue_payload}


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", eeg_connected=eeg_service.connected, versions=db.get_versions())


@app.get("/db/faces", response_model=DBSnapshotResponse)
async def get_faces() -> DBSnapshotResponse:
    records, last_updated, version = db.list_faces()
    return DBSnapshotResponse(db=DBName.faces, version=version, last_updated=last_updated, records=records)


@app.get("/db/cues", response_model=DBSnapshotResponse)
async def get_cues() -> DBSnapshotResponse:
    records, last_updated, version = db.list_cues()
    return DBSnapshotResponse(db=DBName.cues, version=version, last_updated=last_updated, records=records)


@app.post("/event_timing")
async def post_event_timing(payload: EventTimingRequest) -> JSONResponse:
    return JSONResponse(await process_event_timing(payload.event_lsl_timestamp))


@app.post("/face_input")
async def post_face_input(payload: FaceInputRequest) -> JSONResponse:
    try:
        result = await process_face_input(payload)
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("face_input failed")
        return JSONResponse({"type": "error", "where": "face_input", "detail": str(exc)}, status_code=400)


@app.post("/cue_info")
async def post_cue_info(payload: CueInfoRequest) -> JSONResponse:
    return JSONResponse(await process_cue_info(payload))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await ws_manager.connect(websocket)
    try:
        hello = {
            "type": "hello",
            "server_time": time.time(),
            "eeg_connected": eeg_service.connected,
            "last_event_time": state_cache.latest_classifier_result.timestamp if state_cache.latest_classifier_result else None,
            "versions": db.get_versions(),
        }
        await ws_manager.send_json(websocket, hello)
        await send_db_snapshot(websocket, DBName.faces)
        await send_db_snapshot(websocket, DBName.cues)

        while True:
            raw = await websocket.receive_json()
            try:
                msg = WSIncomingMessage.model_validate(raw)
            except Exception as exc:
                await ws_manager.send_json(websocket, {"type": "error", "where": "ws.validate", "detail": str(exc)})
                continue

            if msg.type == "event_timing":
                await ws_manager.send_json(websocket, await process_event_timing(float(msg.event_lsl_timestamp)))
            elif msg.type == "face_input":
                payload = FaceInputRequest(
                    timestamp=float(msg.timestamp),
                    image_base64=str(msg.image_base64),
                    face_json=msg.face_json or {},
                )
                try:
                    await ws_manager.send_json(websocket, await process_face_input(payload))
                except Exception as exc:
                    await ws_manager.send_json(websocket, {"type": "error", "where": "ws.face_input", "detail": str(exc)})
            elif msg.type == "cue_info":
                payload = CueInfoRequest(timestamp=float(msg.timestamp), cue_json=msg.cue_json or {})
                await ws_manager.send_json(websocket, await process_cue_info(payload))
            elif msg.type == "pull_db":
                payload = PullDBRequest(db=msg.db)
                await send_db_snapshot(websocket, payload.db)
            else:
                await ws_manager.send_json(websocket, {"type": "error", "where": "ws", "detail": f"Unknown message type: {msg.type}"})
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(websocket)
