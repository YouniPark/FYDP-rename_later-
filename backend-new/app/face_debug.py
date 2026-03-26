from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket

from app.config import settings
from app.face_contracts import FaceDebugIdentity, FaceDebugResponse
from app.face_memory import FaceVoteResult, face_memory_voter


class FaceDebugHub:
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


face_debug_hub = FaceDebugHub()


def _load_people_by_id() -> dict[int, dict[str, Any]]:
    people_path = settings.people_json_path
    if not people_path.exists():
        return {}
    try:
        payload = json.loads(people_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, list):
        return {}
    result: dict[int, dict[str, Any]] = {}
    for person in payload:
        if not isinstance(person, dict):
            continue
        person_id = person.get("id")
        if isinstance(person_id, int):
            result[person_id] = person
    return result


def build_face_debug_response(vote: FaceVoteResult) -> FaceDebugResponse:
    dnn_face_id: int | None = None
    if vote.face_id is not None:
        try:
            dnn_face_id = int(vote.face_id)
        except (ValueError, TypeError):
            pass

    recognized_identity: FaceDebugIdentity | None = None
    if dnn_face_id is not None and dnn_face_id > 0:
        person = _load_people_by_id().get(dnn_face_id)
        if person is not None:
            recognized_identity = FaceDebugIdentity(
                id=dnn_face_id,
                name=person.get("name"),
                relationship=person.get("relationship"),
            )

    decided_at = datetime.fromtimestamp(vote.decided_at_ts, tz=timezone.utc)

    return FaceDebugResponse(
        dnn_name=vote.face_id,
        dnn_face_id=dnn_face_id,
        recognized_identity=recognized_identity,
        confidence=vote.confidence,
        decided_at=decided_at,
        sample_count=vote.sample_count,
        window_seconds=face_memory_voter.window_seconds,
        is_unknown=vote.face_id is None,
    )
