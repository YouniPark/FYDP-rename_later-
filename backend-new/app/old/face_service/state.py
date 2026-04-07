from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.face_contracts import LatestFaceDecisionResponse
from app.face_service.mapping import load_name_to_people_id
from app.face_service.recognizer_runtime import FaceRuntimeRecognizer
from app.face_service.settings import FaceServiceSettings
from app.face_service.temporal_memory import TemporalFaceMemory

logger = logging.getLogger(__name__)


@dataclass
class FaceFrameEnvelope:
    timestamp: float
    jpeg_bytes: bytes


def _build_recognizer(settings: FaceServiceSettings):
    """Instantiate the correct recognizer based on *settings.recognizer_backend*."""
    if settings.recognizer_backend.lower() == "arcface":
        from app.face_service.arcface_recognizer import ArcFaceRuntimeRecognizer
        from app.face_service.embedding_store import ArcFaceEmbeddingStore

        store = ArcFaceEmbeddingStore(settings.arcface_embedding_store_path)
        return ArcFaceRuntimeRecognizer(settings, store)
    return FaceRuntimeRecognizer(settings)


class FaceServiceState:
    def __init__(self, settings: FaceServiceSettings):
        self.settings = settings
        self.frame_queue: asyncio.Queue[FaceFrameEnvelope] = asyncio.Queue(maxsize=settings.max_frame_queue)

        self.recognizer = _build_recognizer(settings)
        self.temporal_memory = TemporalFaceMemory(window_seconds=settings.memory_window_seconds)

        # CSV mapping is only required for the KNN backend (ArcFace returns face_ids directly)
        try:
            self.name_to_people_id = load_name_to_people_id(settings.mapping_csv_path)
        except FileNotFoundError:
            if settings.recognizer_backend.lower() != "arcface":
                raise
            logger.info(
                "KNN mapping CSV not found — OK for arcface backend (face_ids are returned directly)"
            )
            self.name_to_people_id = {}

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
