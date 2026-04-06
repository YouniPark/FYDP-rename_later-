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

    # ------------------------------------------------------------------
    # Backend toggle — set FACE_RECOGNIZER_BACKEND=arcface to use ArcFace
    # ------------------------------------------------------------------

    recognizer_backend: str = Field(default="knn", alias="FACE_RECOGNIZER_BACKEND")
    """Which recognition backend to use.  Accepted values: ``knn`` (default),
    ``arcface``.  Can be overridden via the ``FACE_RECOGNIZER_BACKEND``
    environment variable or ``.env`` file."""

    # ------------------------------------------------------------------
    # KNN backend paths
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # ArcFace backend settings
    # ------------------------------------------------------------------

    arcface_model_name: str = Field(default="buffalo_l", alias="ARCFACE_MODEL_NAME")
    """InsightFace model pack name.  ``buffalo_l`` (highest accuracy) is the
    default; lighter alternatives are ``buffalo_m`` and ``buffalo_sc``.
    Set ``ARCFACE_MODEL_NAME`` in ``.env`` to override."""

    arcface_similarity_threshold: float = Field(
        default=0.35, alias="ARCFACE_SIMILARITY_THRESHOLD", ge=0.0, le=1.0
    )
    """Minimum cosine similarity to accept a face as a known identity.
    Faces scoring below this threshold are labelled *Unknown*.
    Typical range: 0.30–0.45.  Set ``ARCFACE_SIMILARITY_THRESHOLD`` in
    ``.env`` to tune."""

    arcface_model_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "insightface",
        alias="ARCFACE_MODEL_DIR",
    )
    """Directory where InsightFace downloads / caches model weights.
    Defaults to ``backend-new/data/insightface``.  Set
    ``ARCFACE_MODEL_DIR`` in ``.env`` to use a shared model cache."""

    arcface_embedding_store_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "data"
        / "arcface_embeddings.pkl",
        alias="ARCFACE_EMBEDDING_STORE",
    )
    """Path to the pickle file that stores enrolled ArcFace embeddings.
    Created automatically on first enrollment.  Set
    ``ARCFACE_EMBEDDING_STORE`` in ``.env`` to override."""

    # ------------------------------------------------------------------
    # Shared / convenience paths used by both backends
    # ------------------------------------------------------------------

    people_json_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
        / "WebServer"
        / "database"
        / "PeopleDatabase"
        / "people.json",
        alias="PEOPLE_JSON_PATH",
    )
    """Path to people.json (shared with the main backend).  Used by the
    standalone face service for debug identity lookups."""


face_service_settings = FaceServiceSettings()
