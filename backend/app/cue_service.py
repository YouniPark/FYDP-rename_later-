from __future__ import annotations

import asyncio
import time
from typing import Any

from app.adapters.existing_functions import call_prepare_cue
from app.db import Database
from app.state import AppStateCache


class CueService:
    def __init__(self, db: Database, state_cache: AppStateCache) -> None:
        self.db = db
        self.state_cache = state_cache
        self._latest_cue_payload: dict[str, Any] | None = None

    @property
    def latest_cue_payload(self) -> dict[str, Any] | None:
        return self._latest_cue_payload

    async def build_cue(
        self,
        classifier_flag: bool | None = None,
        face_result: dict[str, Any] | None = None,
        cue_json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Fill missing values from newest cache snapshot for coherent cue composition.
        snap = self.state_cache.snapshot()
        resolved_classifier = classifier_flag
        if resolved_classifier is None and snap["classifier"] is not None:
            resolved_classifier = bool(snap["classifier"].value)

        resolved_face_result = face_result
        if resolved_face_result is None and snap["face"] is not None:
            resolved_face_result = snap["face"].value

        resolved_cue_json = cue_json
        if resolved_cue_json is None and snap["cue"] is not None:
            resolved_cue_json = snap["cue"].value

        cue_db, _, _ = self.db.list_cues()
        payload = await asyncio.to_thread(
            call_prepare_cue,
            resolved_classifier,
            resolved_face_result,
            resolved_cue_json,
            cue_db,
        )

        if resolved_classifier:
            payload.setdefault("priority", "high")
        if resolved_face_result and not resolved_face_result.get("known", True):
            payload.setdefault("notice", "New face detected")

        payload.setdefault("generated_at", time.time())
        self._latest_cue_payload = payload
        return payload
