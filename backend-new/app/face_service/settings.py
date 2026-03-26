from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FaceServiceSettings(BaseSettings):
    """Runtime settings for the standalone face streaming service."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    workspace_root: Path = Path(__file__).resolve().parents[3]
    backend_root: Path = Path(__file__).resolve().parents[2]

    host: str = Field(default="0.0.0.0", alias="FACE_SERVICE_HOST")
    port: int = Field(default=8000, alias="FACE_SERVICE_PORT", ge=1, le=65535)

    inference_sample_fps: float = Field(default=10.0, alias="INFERENCE_SAMPLE_FPS", gt=0.0)
    memory_window_seconds: float = Field(default=2.0, alias="MEMORY_WINDOW_SECONDS", gt=0.0)
    unknown_threshold: float = Field(default=0.5, alias="UNKNOWN_THRESHOLD", ge=0.0, le=1.0)
    detector_conf_threshold: float = Field(default=0.7, alias="DETECTOR_CONF_THRESHOLD", ge=0.0, le=1.0)

    max_frame_queue: int = Field(default=32, alias="MAX_FRAME_QUEUE", ge=1)
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    mapping_csv_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
        / "facial-recognition-DNN"
        / "face_id_mapping.csv",
        alias="FACE_ID_MAPPING_CSV",
    )
    recognizer_model_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
        / "facial-recognition-DNN"
        / "face_recognizer.pkl",
        alias="FACE_RECOGNIZER_MODEL",
    )
    embedder_model_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
        / "facial-recognition-DNN"
        / "face_embedding_model"
        / "openface_nn4.small2.v1.t7",
        alias="FACE_EMBEDDER_MODEL",
    )
    detector_prototxt_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
        / "facial-recognition-DNN"
        / "face_detection_model"
        / "deploy.prototxt",
        alias="FACE_DETECTOR_PROTOTXT",
    )
    detector_caffemodel_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
        / "facial-recognition-DNN"
        / "face_detection_model"
        / "res10_300x300_ssd.caffemodel",
        alias="FACE_DETECTOR_CAFFEMODEL",
    )
    detector_prototxt_fallback_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "poc" / "deploy.prototxt",
        alias="FACE_DETECTOR_PROTOTXT_FALLBACK",
    )
    detector_caffemodel_fallback_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "poc" / "res10_300x300_ssd.caffemodel",
        alias="FACE_DETECTOR_CAFFEMODEL_FALLBACK",
    )


face_service_settings = FaceServiceSettings()
