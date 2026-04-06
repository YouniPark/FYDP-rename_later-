"""
backend-test/main.py
====================
Test-mode FastAPI server for the ADAD face recognition pipeline.

Replaces all hardware dependencies with software stubs so you can validate
the full ArcFace pipeline using only a webcam and a keyboard:

  • Webcam (OpenCV) → feeds frames to the face recognition loop
  • Keyboard / HTTP POST /test/trigger → fires fixation events
  • Dummy EEG pipeline → returns a configurable ``is_unfamiliar`` result
  • Console print → displays every cue decision (also broadcast on /ws/ar)

All face-DB management, ArcFace enrollment, and debug UI endpoints from the
production backend are preserved so this server is a drop-in replacement for
``backend-new`` during local testing.

Start:
    cd backend-test
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

# ── Bootstrap: add backend-new to the Python path ────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_NEW = os.path.normpath(os.path.join(_HERE, "..", "backend-new"))
if _BACKEND_NEW not in sys.path:
    sys.path.insert(0, _BACKEND_NEW)
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.cue_service import build_cue_decision
from app.face_pipeline import face_recognition_loop
from app.state import AppState
from app.storage.models import CueDBManifest, FaceDBManifest
from app.face_memory import face_memory_voter, FaceVoteResult
from app.face_contracts import FaceDebugResponse
from app.face_debug import FaceDebugHub, face_debug_hub, build_face_debug_response

# Test-mode overrides (local to backend-test/)
from dummy_eeg import run_dummy_eeg_event_pipeline
from webcam_feeder import webcam_capture_loop
from keyboard_trigger import keyboard_event_loop

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("backend-test.main")

state = AppState(settings)


# ── WebSocket hub (same pattern as backend-new) ───────────────────────────────

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
            self.connections.discard(ws)

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


# ── Console cue printer ───────────────────────────────────────────────────────

def _print_cue_decision(decision_json: dict[str, Any]) -> None:
    """Pretty-print a cue decision to stdout (replaces AR WebSocket output)."""
    sep = "═" * 64
    face_id = decision_json.get("face_id")
    is_unfamiliar = decision_json.get("is_unfamiliar")
    send_cue = decision_json.get("send_cue")
    cue = decision_json.get("cue")
    event_id = decision_json.get("event_id")
    server_time = decision_json.get("server_time", "")

    print(f"\n{sep}")
    print(f"  CUE DECISION  [{server_time}]")
    print(f"  event_id     : {event_id}")
    print(f"  face_id      : {face_id}")
    print(f"  is_unfamiliar: {is_unfamiliar}")
    print(f"  send_cue     : {send_cue}")
    if send_cue and cue:
        cue_str = json.dumps(cue, indent=4, ensure_ascii=False)
        indented = "\n".join("    " + line for line in cue_str.splitlines())
        print(f"  cue payload  :\n{indented}")
    else:
        print(f"  cue payload  : (none — {'familiar' if not is_unfamiliar else 'face not identified'})")
    print(f"{sep}\n")


# ── App lifespan (startup / shutdown) ────────────────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI):
    async def dispatch_fixation(event_id: str, lsl_timestamp: float, _proxy_name: str) -> None:
        """Handle a fixation event: run dummy EEG, build cue, print + broadcast."""
        result = await run_dummy_eeg_event_pipeline(state, event_id, lsl_timestamp)
        if "is_unfamiliar" not in result:
            logger.info(
                "Event %s not forwarded to cue service (status=%s)",
                event_id,
                result.get("status"),
            )
            return

        voted_face = await face_memory_voter.get_voted_face(lsl_timestamp)
        logger.info(
            "Event %s: EEG result is_unfamiliar=%s  voted_face=%s — building cue decision",
            event_id,
            result["is_unfamiliar"],
            voted_face,
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
            event_id,
            decision.send_cue,
            decision.face_id,
            decision.is_unfamiliar,
        )
        state.latest_cue_decision_json = decision_json
        _print_cue_decision(decision_json)
        await hub.broadcast_json({"type": "cue_decision", "payload": decision_json})

    async def _face_debug_publish(vote: FaceVoteResult, frame_b64: str, timestamp: float) -> None:
        debug_response = build_face_debug_response(vote)
        await face_debug_hub.publish({
            "type": "face_debug",
            "payload": debug_response.model_dump(mode="json"),
            "frame_jpeg_b64": frame_b64,
            "frame_timestamp": timestamp,
        })

    webcam_task = asyncio.create_task(webcam_capture_loop(state))
    face_task = asyncio.create_task(face_recognition_loop(state, debug_publish=_face_debug_publish))
    trigger_task = asyncio.create_task(keyboard_event_loop(state, dispatch_fixation))

    try:
        yield
    finally:
        webcam_task.cancel()
        face_task.cancel()
        trigger_task.cancel()
        await asyncio.gather(webcam_task, face_task, trigger_task, return_exceptions=True)


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="ADAD Backend — Test Mode (Webcam + Dummy EEG)",
    description=(
        "Drop-in replacement for backend-new that uses a webcam and a dummy EEG "
        "pipeline so you can test ArcFace face recognition without AR hardware."
    ),
    lifespan=lifespan,
)

app.mount("/media/images", StaticFiles(directory=str(settings.images_dir)), name="images")
app.mount("/media/audio", StaticFiles(directory=str(settings.auditory_cue_dir)), name="audio")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": "test (webcam + dummy EEG)"}


# ── Test-specific endpoints ───────────────────────────────────────────────────

@app.post("/test/trigger")
async def trigger_event(proxy_name: str = "http") -> dict[str, Any]:
    """Manually fire a fixation event (simulates an AR gaze fixation).

    Equivalent to pressing Enter in the terminal.  Call this endpoint with curl
    or from a browser to trigger one recognition + cue cycle.

    Query parameter:
      proxy_name  Label for the synthetic event (default: ``http``).
    """
    ts = time.time()
    event_id = f"event_{proxy_name}_{ts:.6f}"
    logger.info("HTTP trigger: event_id=%s", event_id)

    result = await run_dummy_eeg_event_pipeline(state, event_id, ts)
    if "is_unfamiliar" not in result:
        return result

    voted_face = await face_memory_voter.get_voted_face(ts)
    decision = await build_cue_decision(
        state,
        event_id=event_id,
        event_lsl_timestamp=ts,
        is_unfamiliar=result["is_unfamiliar"],
        face_id=voted_face,
    )
    decision_json = decision.model_dump(mode="json")
    state.latest_cue_decision_json = decision_json
    _print_cue_decision(decision_json)
    await hub.broadcast_json({"type": "cue_decision", "payload": decision_json})
    return decision_json


@app.get("/test/trigger")
async def trigger_event_get(proxy_name: str = "http") -> dict[str, Any]:
    """GET alias for ``POST /test/trigger`` — convenient for browser testing."""
    return await trigger_event(proxy_name=proxy_name)


# ── Cue polling ───────────────────────────────────────────────────────────────

@app.get("/cue/latest")
async def get_cue_latest() -> dict[str, Any]:
    """Return the most recent cue decision produced by the pipeline."""
    if state.latest_cue_decision_json is None:
        raise HTTPException(status_code=404, detail="No cue decision available yet")
    return state.latest_cue_decision_json


# ── AR WebSocket (optional — connects any WebSocket client) ───────────────────

@app.websocket("/ws/ar")
async def ws_ar(ws: WebSocket) -> None:
    """WebSocket endpoint for cue decisions.

    Clients receive ``{"type": "cue_decision", "payload": {...}}`` after each
    fixation event.  Useful for integrating a custom display or logging tool.
    """
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


# ── Face database management ──────────────────────────────────────────────────

@app.post("/db/face")
async def upload_face(
    face_id: str = Form(...),
    metadata_json: str = Form("{}"),
    image: UploadFile = File(...),
) -> dict[str, Any]:
    """Upload an enrollment image for a person and auto-enroll in ArcFace."""
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

    enrolled = False
    try:
        from user_modules.face import enroll_arcface_from_image_bytes
        enrolled = await asyncio.to_thread(enroll_arcface_from_image_bytes, face_id, raw)
    except Exception:
        logger.exception(
            "ArcFace auto-enrollment failed for face_id=%s "
            "(face record was saved; retry via POST /face/enroll/%s)",
            face_id,
            face_id,
        )

    return {"status": "ok", "record": record.model_dump(mode="json"), "arcface_enrolled": enrolled}


@app.post("/db/cue")
async def upload_cue(face_id: str = Form(...), cue_json: str = Form(...)) -> dict[str, Any]:
    """Store a cue payload for a person."""
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


# ── ArcFace enrollment ────────────────────────────────────────────────────────

@app.post("/face/enroll/{face_id}")
async def enroll_face(face_id: str) -> dict[str, Any]:
    """(Re-)enroll a face using its stored image.

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


# ── Face debug ────────────────────────────────────────────────────────────────

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
    """Live face recognition debug UI — open in a browser."""
    return HTMLResponse("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Face Debug UI — Test Mode</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 16px; background: #fafafa; }
    h2 { color: #333; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
             font-size: 0.75em; font-weight: bold; margin-left: 8px; }
    .badge-test { background: #ffe066; color: #7a5c00; }
    .row { display: flex; gap: 20px; align-items: flex-start; margin-top: 12px; }
    img { width: 480px; height: auto; border: 1px solid #ccc; background: #f5f5f5; }
    .panel { min-width: 320px; background: #fff; border: 1px solid #ddd;
             border-radius: 6px; padding: 16px; }
    .label { font-weight: bold; color: #555; }
    .kv { margin: 8px 0; }
    .trigger-btn {
      margin-top: 20px; padding: 10px 24px; font-size: 1em;
      background: #2563eb; color: white; border: none; border-radius: 6px;
      cursor: pointer;
    }
    .trigger-btn:hover { background: #1d4ed8; }
    .trigger-btn:active { background: #1e40af; }
    #trigger-status { margin-top: 8px; font-size: 0.85em; color: #666; }
  </style>
</head>
<body>
  <h2>Live Face Recognition Debug <span class="badge badge-test">TEST MODE</span></h2>
  <div class="row">
    <img id="frame" alt="webcam frame" />
    <div class="panel">
      <div class="kv"><span class="label">Face ID:</span> <span id="dnn_name">-</span></div>
      <div class="kv"><span class="label">Numeric ID:</span> <span id="dnn_face_id">-</span></div>
      <div class="kv"><span class="label">Identity Name:</span> <span id="identity_name">-</span></div>
      <div class="kv"><span class="label">Relationship:</span> <span id="relationship">-</span></div>
      <div class="kv"><span class="label">Confidence:</span> <span id="confidence">-</span></div>
      <div class="kv"><span class="label">Unknown:</span> <span id="is_unknown">-</span></div>
      <div class="kv"><span class="label">Decided At:</span> <span id="decided_at">-</span></div>
      <button class="trigger-btn" onclick="triggerEvent()">&#9654; Trigger fixation event</button>
      <div id="trigger-status"></div>
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
      document.getElementById("confidence").textContent =
        payload.confidence != null ? payload.confidence.toFixed(3) : "-";
      document.getElementById("is_unknown").textContent = payload.is_unknown ?? "-";
      document.getElementById("decided_at").textContent = payload.decided_at ?? "-";
      if (message.frame_jpeg_b64) {
        document.getElementById("frame").src = `data:image/jpeg;base64,${message.frame_jpeg_b64}`;
      }
    };

    async function triggerEvent() {
      const statusEl = document.getElementById("trigger-status");
      statusEl.textContent = "Triggering…";
      try {
        const resp = await fetch("/test/trigger?proxy_name=ui", { method: "POST" });
        const data = await resp.json();
        statusEl.textContent =
          `Done — send_cue=${data.send_cue}, face_id=${data.face_id ?? "none"}`;
      } catch (e) {
        statusEl.textContent = "Error: " + e.message;
      }
    }
  </script>
</body>
</html>
    """)
