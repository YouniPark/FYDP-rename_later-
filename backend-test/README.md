# backend-test — ArcFace test harness (webcam + dummy EEG)

A standalone test server that replicates the full `backend-new` pipeline but
replaces every hardware dependency with a software stub so you can validate
ArcFace face recognition using **only a laptop webcam and a keyboard**.

| Component | Production (`backend-new`) | Test (`backend-test`) |
|---|---|---|
| Video source | Magic Leap 2 → WebSocket `/ws/video` | Built-in webcam (OpenCV) |
| Fixation events | LSL `FixationEvents` stream from AR app | Enter key / POST `/test/trigger` |
| EEG pipeline | Real LSL EEG stream + ML classifier | Configurable dummy result |
| Cue delivery | WebSocket `/ws/ar` → Magic Leap 2 | **Console print** + `/ws/ar` |

Everything else — ArcFace enrollment, face DB, cue DB, WebServer database,
debug UI — is **identical** to the production backend because `backend-test`
imports directly from `backend-new` at runtime.

---

## Quick-start

### 1 — Install dependencies

```bash
cd backend-test
pip install -r requirements.txt
```

### 2 — Configure

```bash
cp .env.example .env
# Edit .env — at minimum verify FACE_RECOGNIZER_BACKEND=arcface
```

### 3 — Enroll faces (if not already done via `backend-new`)

The test server shares the same face database as `backend-new`.  If you have
already enrolled faces there you can skip this step.

```bash
# Upload a face image and auto-enroll in ArcFace
curl -X POST http://localhost:8000/db/face \
  -F "face_id=1" \
  -F 'metadata_json={"name":"Alice","relationship":"colleague"}' \
  -F "image=@/path/to/alice.jpg"

# Re-enroll from existing stored image
curl -X POST http://localhost:8000/face/enroll/1
```

### 4 — Start the server

```bash
cd backend-test
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5 — Test face recognition

1. Point your webcam at a person whose face is enrolled.
2. Trigger a fixation event using **one** of:
   - Press **Enter** in the server terminal.
   - `curl -X POST http://localhost:8000/test/trigger`
   - Open `http://localhost:8000/face/debug/ui` in a browser and click the
     **Trigger fixation event** button.
3. The cue decision is **printed to the server console** and also broadcast
   over `/ws/ar` for any connected WebSocket client.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Server health check |
| POST | `/test/trigger` | Fire a synthetic fixation event |
| GET | `/test/trigger` | Same as POST (convenient for browsers) |
| GET | `/cue/latest` | Last cue decision (JSON) |
| WS | `/ws/ar` | Real-time cue decisions over WebSocket |
| GET | `/face/debug/ui` | Live webcam + face recognition debug UI |
| GET | `/face/debug/latest` | Latest face vote result (JSON) |
| WS | `/ws/face-debug` | Real-time face debug feed over WebSocket |
| POST | `/db/face` | Upload/enroll a face image |
| GET | `/db/face` | List enrolled faces |
| POST | `/db/cue` | Store a cue payload |
| GET | `/db/cue` | List stored cues |
| POST | `/face/enroll/{face_id}` | Re-enroll from stored image |

---

## Configuration reference

All settings are read from `backend-test/.env` (copy from `.env.example`).

### Dummy EEG

| Variable | Default | Description |
|---|---|---|
| `DUMMY_EEG_ALWAYS_UNFAMILIAR` | `true` | `true` = always unfamiliar; `false` = always familiar; `random` = 50/50 |
| `DUMMY_EEG_DELAY` | `0.2` | Simulated processing delay (seconds) |

### Webcam

| Variable | Default | Description |
|---|---|---|
| `WEBCAM_DEVICE_INDEX` | `0` | OpenCV device index (0 = built-in) |
| `WEBCAM_FPS` | `15` | Target capture framerate |
| `WEBCAM_JPEG_QUALITY` | `85` | JPEG quality (1–100) |

### Trigger

| Variable | Default | Description |
|---|---|---|
| `TEST_AUTO_TRIGGER_INTERVAL` | `0` | Auto-fire every N seconds; 0 = off |

### Face Recognition (shared with `backend-new`)

| Variable | Default | Description |
|---|---|---|
| `FACE_RECOGNIZER_BACKEND` | `arcface` | `arcface` or `knn` |
| `ARCFACE_MODEL_NAME` | `buffalo_l` | InsightFace model pack |
| `ARCFACE_SIMILARITY_THRESHOLD` | `0.35` | Cosine similarity threshold |

---

## How it works

```
Webcam (OpenCV)
    │  JPEG frames → frame_queue
    ▼
face_recognition_loop()  [from backend-new unchanged]
    │  dnn_face_recognition() → ArcFace
    ▼
FaceMemoryVoter          [temporal voting, 5 s window]
    │
    │  ← Enter key / POST /test/trigger / auto-timer
    ▼
run_dummy_eeg_event_pipeline()   [returns is_unfamiliar config]
    │
    ▼
build_cue_decision()     [from backend-new unchanged]
    │
    ├──▶ Console print   ← NEW (replaces AR WebSocket delivery)
    └──▶ /ws/ar broadcast
```

The only components that differ from the production backend are:

1. **`webcam_feeder.py`** — replaces the AR WebSocket video source.
2. **`keyboard_trigger.py`** — replaces the LSL `FixationEvents` inlet.
3. **`dummy_eeg.py`** — replaces the real EEG pipeline.
4. **`main.py`** — wires the above together and adds `_print_cue_decision()`.

All face recognition, cue preparation, database, and debug logic is imported
directly from `backend-new` without modification.
