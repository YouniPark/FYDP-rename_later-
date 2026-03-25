from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class VideoFrameWsMessage(BaseModel):
    """Unity -> server frame payload over WebSocket."""

    timestamp: float = Field(description="Sender-side Unix timestamp in seconds")
    encoding: Literal["jpeg"] = Field(description="Encoded frame format")
    data_b64: str = Field(min_length=1, description="Base64-encoded JPEG frame")


class LatestFaceDecisionResponse(BaseModel):
    """Server -> Unity latest memory-voted face decision."""

    name: str | None = Field(
        default=None,
        description="Resolved identity name; null when no face decision is available.",
    )
    people_id: int | None = Field(
        default=None,
        description="Known mapped ID, 0 for Unknown, null for no-face.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence of the winning class; null for no-face.",
    )
    decided_at: datetime = Field(description="UTC timestamp of latest decision.")
    source: Literal["memory_vote"] = Field(default="memory_vote")
    window_seconds: float = Field(gt=0.0, description="Voting memory window in seconds.")
    sample_count: int = Field(ge=0, description="Samples included in the vote window.")
    is_unknown: bool = Field(
        default=False,
        description="True when winner is Unknown (people_id=0).",
    )
