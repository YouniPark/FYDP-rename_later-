from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FaceContractSettings(BaseSettings):
    """Phase 1 configuration contract for streaming and lookup behavior."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Inference pacing
    inference_sample_fps: float = Field(default=5.0, alias="INFERENCE_SAMPLE_FPS", gt=0.0)
    memory_window_seconds: float = Field(default=2.0, alias="MEMORY_WINDOW_SECONDS", gt=0.0)
    unknown_threshold: float = Field(default=0.5, alias="UNKNOWN_THRESHOLD", ge=0.0, le=1.0)

    # Unity sender defaults
    jpeg_quality: int = Field(default=75, alias="JPEG_QUALITY", ge=1, le=100)

    # Integration endpoints
    video_ws_url: str = Field(default="ws://0.0.0.0:8000/ws/video", alias="VIDEO_WS_URL")
    face_lookup_url: str = Field(default="http://0.0.0.0:8000/face/latest", alias="FACE_LOOKUP_URL")
    request_timeout_seconds: float = Field(default=3.0, alias="REQUEST_TIMEOUT_SECONDS", gt=0.0)

    # Tie-break policy for equal vote counts inside the memory window.
    tie_break_strategy: str = Field(default="most_recent", alias="TIE_BREAK_STRATEGY")


face_contract_settings = FaceContractSettings()
