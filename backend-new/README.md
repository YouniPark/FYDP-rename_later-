# ADAD Python Backend Server

This repository implements only the Python backend server block from the architecture diagram. It orchestrates EEG-event processing, live face recognition, cue decisioning, and local face/cue database storage for AR integration.

## What this backend does

- Connects to EEG LSL on startup with automatic retry.
- Accepts AR event timestamps via REST and runs EEG pipeline only when events arrive.
- Accepts video frames over WebSocket and runs face recognition continuously in a background worker.
- Combines EEG result + latest face identity to run cue preparation and emit real-time cue decisions over WebSocket.
- Supports face database uploads (image + metadata) and cue database uploads (JSON).
- Serves face/cue manifests for AR startup sync and provides file download endpoint for stored images.

## Project layout

- `app/main.py`
  - FastAPI app creation and startup lifecycle.
  - REST routes for events, database upload/download, health checks.
  - WebSocket endpoints:
    - `/ws/ar` for cue decision output and optional DB sync push on connect.
    - `/ws/video` for incoming JPEG frame messages from AR.
- `app/config.py`
  - Environment-based application settings using Pydantic settings.
- `app/state.py`
  - Shared mutable state with async locks and frame queue.
  - In-memory face DB, cue DB, EEG stream handle, and latest recognized face ID.
- `app/eeg_pipeline.py`
  - EEG connection retry loop.
  - Event-triggered EEG orchestration:
    - `event_filter -> create_epoch -> eeg_processing -> ml_classifier`.
  - Handles create_epoch signature compatibility when user function accepts either `(event_ts)` or `(stream, event_ts)`.
- `app/face_pipeline.py`
  - Validates/decodes incoming base64 JPEG frames.
  - Background face recognition loop using the provided DNN function.
- `app/cue_service.py`
  - Calls the provided cue preparation function and shapes final cue decision payload.
- `app/storage/models.py`
  - Pydantic message/data schemas for events, frames, cue decisions, and manifests.
- `app/storage/db.py`
  - Local disk persistence for face and cue manifests.
  - File writing for face images and secure file-path resolution for downloads.
- `requirements.txt`
  - Python dependencies.

## Hook points to your existing implementations

This backend expects these existing functions and imports:

- `from user_modules.eeg import connect_eeg, event_filter, create_epoch, eeg_processing`
- `from user_modules.model import ml_classifier`
- `from user_modules.face import dnn_face_recognition`
- `from user_modules.cue import cue_preparation`

If those imports are missing, the backend still starts but raises `NotImplementedError` when those functions are called.

## Message contracts

### Incoming event JSON `POST /events`

```json
{
  "event_id": "uuid-or-string",
  "event_lsl_timestamp": 12345.6789,
  "optional_context": {}
}
```

### Incoming video frame JSON over `WS /ws/video`

```json
{
  "timestamp": 12345.67,
  "encoding": "jpeg",
  "data_b64": "..."
}
```

### Outgoing cue decision JSON over `WS /ws/ar`

```json
{
  "type": "cue_decision",
  "payload": {
    "event_id": "...",
    "event_lsl_timestamp": 12345.6789,
    "face_id": "john_doe",
    "is_unfamiliar": true,
    "send_cue": true,
    "cue": {},
    "server_time": "2026-01-01T12:00:00+00:00"
  }
}
```

### DB sync timing

- Preferred pull model: AR fetches current data at startup using:
  - `GET /db/face`
  - `GET /db/cue`
- Push model (also implemented): backend sends `{"type":"db_sync", ...}` immediately when AR connects to `WS /ws/ar`.

## API summary

- `GET /health`
- `POST /events`
- `WS /ws/ar`
- `WS /ws/video`
- `POST /db/face` as multipart fields:
  - `face_id` (text)
  - `metadata_json` (text JSON)
  - `image` (file)
- `POST /db/cue` as multipart fields:
  - `face_id` (text)
  - `cue_json` (text JSON)
- `GET /db/face`
- `GET /db/cue`
- `GET /db/file?path=faces/<file_name>.jpg`

## Environment variables

- `EEG_LSL_RETRY_SECONDS=3`
- `DATA_DIR=./data`
- `FACE_IMAGES_DIR=./data/faces`
- `CUE_IMAGES_DIR=./data/cues`
- `VIDEO_MODE=ws`
- `VIDEO_PULL_URL=`
- `MAX_FRAME_QUEUE=32`
- `LOG_LEVEL=INFO`
- `MAX_UPLOAD_BYTES=5000000`

## Run instructions

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the WebServer
```bash
uvicorn WebServer.main:app --host 0.0.0.0 --port 8000
```

4. Start the backend:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

5. AR startup flow:
   - Connect to `WS /ws/ar` to receive `db_sync` and real-time cue decisions.
   - Fetch `GET /db/face` and `GET /db/cue` for latest manifests.
   - Send events to `POST /events`.
   - Stream JPEG frames to `WS /ws/video`.

## Standalone face stream service (new path)

The repository now includes a dedicated face-only runtime that does not start EEG/cue orchestration from `app.main`.

Run:

```bash
uvicorn app.face_main:app --host 0.0.0.0 --port 8001
```

Endpoints:
- `GET /health`
- `GET /face/latest`
- `GET /face/debug/latest` (includes both DNN label/ID + identity resolved from `people.json`)
- `GET /face/debug/ui` (live browser interface showing incoming frame + DNN/identity fields)
- `GET /face/diagnostics`
- `WS /ws/face-debug` (stream of live debug payload + jpeg frame)
- `WS /ws/video`

Notes:
- This runtime is intended for the Unity fixation lookup path.
- It reads model and mapping files through path-based settings (no direct module import from `facial-recognition-DNN`).

## Notes on robustness

- Event requests return a structured error if EEG stream is not connected.
- Invalid frame encoding/base64 is rejected.
- Database manifests are loaded from disk on startup; missing files initialize as empty databases.
- Face recognition failures do not crash server loops.
