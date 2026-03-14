from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.adapters.existing_functions import (
    call_connect_eeg_lsl,
    call_create_epoch,
    call_event_filter,
    call_ml_classifier_predict,
    call_process_eeg_epoch,
)
from app.config import EEG_RETRY_SECONDS
from app.db import Database
from app.state import AppStateCache

logger = logging.getLogger(__name__)


class EEGService:
    def __init__(self, db: Database, state_cache: AppStateCache) -> None:
        self.db = db
        self.state_cache = state_cache
        self.stream: Any | None = None
        self._lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self.stream is not None

    async def ensure_connected(self) -> Any | None:
        if self.stream is not None:
            return self.stream
        async with self._lock:
            if self.stream is not None:
                return self.stream
            try:
                self.stream = await asyncio.to_thread(call_connect_eeg_lsl)
                logger.info("EEG stream connected")
            except Exception:
                logger.exception("Failed to connect EEG stream")
                self.stream = None
        return self.stream

    async def retry_connect_loop(self) -> None:
        while True:
            await self.ensure_connected()
            await asyncio.sleep(EEG_RETRY_SECONDS)

    async def handle_event(self, event_lsl_timestamp: float) -> dict[str, Any]:
        should_process = await asyncio.to_thread(call_event_filter, event_lsl_timestamp)
        if not should_process:
            return {"status": "ignored", "reason": "event_filter_rejected"}

        stream = await self.ensure_connected()
        if stream is None:
            return {"status": "error", "reason": "eeg_not_connected"}

        epoch = await asyncio.to_thread(call_create_epoch, stream, event_lsl_timestamp)
        features = await asyncio.to_thread(call_process_eeg_epoch, epoch)
        is_unfamiliar = await asyncio.to_thread(call_ml_classifier_predict, features)
        self.state_cache.set_classifier(event_lsl_timestamp, is_unfamiliar)
        self.db.insert_event_record(
            event_lsl_timestamp=event_lsl_timestamp,
            classifier_result=is_unfamiliar,
            metadata={"feature_size": int(getattr(features, "size", 0))},
        )
        return {"status": "ok", "is_unfamiliar": is_unfamiliar}
