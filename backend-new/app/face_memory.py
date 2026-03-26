from __future__ import annotations

import asyncio
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque

from app.config import settings


@dataclass
class FaceVoteResult:
    face_id: str | None
    sample_count: int
    confidence: float | None
    decided_at_ts: float
    source: str = "memory_vote"


class FaceMemoryVoter:
    def __init__(
        self,
        window_seconds: float = 5.0,
        min_votes: int = 2,
        majority_ratio: float = 0.6,
    ) -> None:
        self.window_seconds = window_seconds
        self.min_votes = min_votes
        self.majority_ratio = majority_ratio

        self._memory: Deque[tuple[float, str | None]] = deque()
        self._lock = asyncio.Lock()
        self._latest_vote: FaceVoteResult | None = None

    def _prune_locked(self, now_ts: float) -> None:
        cutoff = now_ts - self.window_seconds
        while self._memory and self._memory[0][0] < cutoff:
            self._memory.popleft()

    async def add_detection(self, timestamp: float, face_id: str | None) -> FaceVoteResult:
        async with self._lock:
            self._memory.append((timestamp, face_id))
            self._prune_locked(timestamp)

            valid_faces = [fid for _, fid in self._memory if fid is not None]

            if not valid_faces:
                vote = FaceVoteResult(
                    face_id=None,
                    sample_count=0,
                    confidence=None,
                    decided_at_ts=timestamp,
                )
                self._latest_vote = vote
                return vote

            counts = Counter(valid_faces)
            winner, votes = counts.most_common(1)[0]
            total = len(valid_faces)
            ratio = votes / total

            if votes < self.min_votes or ratio < self.majority_ratio:
                voted_face = None
            else:
                voted_face = winner

            vote = FaceVoteResult(
                face_id=voted_face,
                sample_count=total,
                confidence=ratio,
                decided_at_ts=timestamp,
            )
            self._latest_vote = vote
            return vote

    async def get_voted_face(self, now_ts: float | None = None) -> str | None:
        async with self._lock:
            if not self._memory:
                return None

            if now_ts is None:
                now_ts = self._memory[-1][0]

            self._prune_locked(now_ts)

            valid_faces = [fid for _, fid in self._memory if fid is not None]
            if not valid_faces:
                return None

            counts = Counter(valid_faces)
            winner, votes = counts.most_common(1)[0]
            total = len(valid_faces)
            ratio = votes / total

            if votes < self.min_votes or ratio < self.majority_ratio:
                return None

            return winner

    async def get_latest_vote(self) -> FaceVoteResult | None:
        async with self._lock:
            return self._latest_vote


face_memory_voter = FaceMemoryVoter(
    window_seconds=settings.memory_window_seconds,
    min_votes=settings.memory_min_votes,
    majority_ratio=settings.memory_majority_ratio,
)