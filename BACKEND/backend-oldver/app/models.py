from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class DBName(str, Enum):
    faces = "faces"
    cues = "cues"


class EventTimingRequest(BaseModel):
    event_lsl_timestamp: float


class FaceInputRequest(BaseModel):
    timestamp: float
    image_base64: str = Field(..., min_length=1)
    face_json: dict[str, Any] = Field(default_factory=dict)


class CueInfoRequest(BaseModel):
    timestamp: float
    cue_json: dict[str, Any]


class PullDBRequest(BaseModel):
    db: DBName


class WSIncomingMessage(BaseModel):
    type: str
    event_lsl_timestamp: float | None = None
    timestamp: float | None = None
    image_base64: str | None = None
    face_json: dict[str, Any] | None = None
    cue_json: dict[str, Any] | None = None
    db: DBName | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "WSIncomingMessage":
        if self.type == "event_timing" and self.event_lsl_timestamp is None:
            raise ValueError("event_lsl_timestamp is required for event_timing")
        if self.type == "face_input":
            if self.timestamp is None or self.image_base64 is None:
                raise ValueError("timestamp and image_base64 are required for face_input")
        if self.type == "cue_info" and (self.timestamp is None or self.cue_json is None):
            raise ValueError("timestamp and cue_json are required for cue_info")
        if self.type == "pull_db" and self.db is None:
            raise ValueError("db is required for pull_db")
        return self


class DBSnapshotResponse(BaseModel):
    db: DBName
    version: str
    last_updated: float | None
    records: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    eeg_connected: bool
    versions: dict[str, str]
