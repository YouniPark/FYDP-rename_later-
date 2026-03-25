from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.face_contracts import LatestFaceDecisionResponse
from app.face_service.mapping import load_name_to_people_id
from app.face_service.recognizer_runtime import FaceRuntimeRecognizer
from app.face_service.settings import FaceServiceSettings
from app.face_service.temporal_memory import TemporalFaceMemory


@dataclass
class FaceFrameEnvelope:
    timestamp: float
    jpeg_bytes: bytes


class FaceServiceState:
    def __init__(self, settings: FaceServiceSettings):
        self.settings = settings
        self.frame_queue: asyncio.Queue[FaceFrameEnvelope] = asyncio.Queue(maxsize=settings.max_frame_queue)

        self.recognizer = FaceRuntimeRecognizer(settings)
        self.temporal_memory = TemporalFaceMemory(window_seconds=settings.memory_window_seconds)
        self.name_to_people_id = load_name_to_people_id(settings.mapping_csv_path)

        self.latest_decision_lock = asyncio.Lock()
        self.latest_decision = LatestFaceDecisionResponse(
            name=None,
            people_id=None,
            confidence=None,
            decided_at=self.temporal_memory.get_vote().decided_at,
            source="memory_vote",
            window_seconds=settings.memory_window_seconds,
            sample_count=0,
            is_unknown=False,
        )

        self.last_frame_timestamp: float | None = None
        self.processed_frames: int = 0

    async def set_latest_decision(self, decision: LatestFaceDecisionResponse) -> None:
        async with self.latest_decision_lock:
            self.latest_decision = decision

    async def get_latest_decision(self) -> LatestFaceDecisionResponse:
        async with self.latest_decision_lock:
            return self.latest_decision
