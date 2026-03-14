from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FaceRecord(BaseModel):
    face_id: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    image_path: str | None = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CueRecord(BaseModel):
    face_id: str = Field(min_length=1)
    cue: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EventIn(BaseModel):
    event_id: str = Field(min_length=1)
    event_lsl_timestamp: float
    optional_context: dict[str, Any] = Field(default_factory=dict)


class VideoFrameMessage(BaseModel):
    timestamp: float
    encoding: str
    data_b64: str


class CueDecisionMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_lsl_timestamp: float
    face_id: str | None
    is_unfamiliar: bool
    send_cue: bool
    cue: dict[str, Any] | None
    server_time: datetime


class FaceDBManifest(BaseModel):
    faces: list[FaceRecord] = Field(default_factory=list)


class CueDBManifest(BaseModel):
    cues: list[CueRecord] = Field(default_factory=list)
