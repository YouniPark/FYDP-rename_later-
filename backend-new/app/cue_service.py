from datetime import datetime, timezone
from typing import Any

from app.state import AppState
from app.storage.models import CueDecisionMessage

try:
    from user_modules.cue import cue_preparation
except ImportError:
    # TO DO: implement import from user cue module
    def cue_preparation(is_unfamiliar: bool, face_id: str | None) -> tuple[bool, dict[str, Any]]:
        raise NotImplementedError("TO DO: implement cue_preparation import from user modules")


async def build_cue_decision(
    state: AppState,
    event_id: str,
    event_lsl_timestamp: float,
    is_unfamiliar: bool,
    face_id: str | None = None,
) -> CueDecisionMessage:
    # face_id = state.current_face_id
    # send_cue, cue_payload = cue_preparation(is_unfamiliar, face_id)
    resolved_face_id = face_id if face_id is not None else state.current_face_id
    send_cue, cue_payload = cue_preparation(is_unfamiliar, resolved_face_id)

    return CueDecisionMessage(
        event_id=event_id,
        event_lsl_timestamp=event_lsl_timestamp,
        face_id=resolved_face_id,
        is_unfamiliar=is_unfamiliar,
        send_cue=bool(send_cue),
        cue=cue_payload if send_cue else None,
        server_time=datetime.now(timezone.utc),
    )
