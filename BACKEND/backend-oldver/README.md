# python backend server (orang box)

i,mplements python backend
- eeg familiarity pipeline (via LSL trigger events)
- face input pipeline (assume image + face JSON objects)
- cue info pipeline
- local face/cue sqlite storage (FIX, ASK HAILEY)
- websocket + http for ar

## setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## endpoints? check later

- `POST /event_timing`
- `POST /face_input`
- `POST /cue_info`
- `GET /db/faces`
- `GET /db/cues`
- `GET /health`
- `WS /ws`

## ADD EXISTING FUNCTIONS

pre-existing functions:
- `app/adapters/existing_functions.py`

see 'to-do' for where to change
