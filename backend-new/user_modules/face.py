from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from app.face_service.mapping import load_name_to_people_id
from app.face_service.recognizer_runtime import FaceRuntimeRecognizer
from app.face_service.settings import face_service_settings


# ------------------------------------------------------------------
# Recognizer singletons (created lazily on first use)
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def _knn_recognizer() -> FaceRuntimeRecognizer:
    return FaceRuntimeRecognizer(face_service_settings)


@lru_cache(maxsize=1)
def _arcface_recognizer():
    from app.face_service.arcface_recognizer import ArcFaceRuntimeRecognizer
    from app.face_service.embedding_store import ArcFaceEmbeddingStore

    store = ArcFaceEmbeddingStore(face_service_settings.arcface_embedding_store_path)
    return ArcFaceRuntimeRecognizer(face_service_settings, store)


def _get_active_recognizer():
    """Return the recognizer selected by ``FACE_RECOGNIZER_BACKEND``."""
    if face_service_settings.recognizer_backend.lower() == "arcface":
        return _arcface_recognizer()
    return _knn_recognizer()


# ------------------------------------------------------------------
# KNN name → face_id resolution helpers
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def _name_to_people_id() -> dict[str, int]:
    try:
        return load_name_to_people_id(face_service_settings.mapping_csv_path)
    except Exception:
        return {}


def _resolve_face_id(predicted_name: str, face_db: dict[str, Any]) -> str | None:
    """Resolve a KNN-predicted person name to a ``face_id`` string.

    Resolution order:
    1. Check the ``face_id_mapping.csv`` (name → integer people_id).
    2. Check ``face_db`` keys (case-insensitive).
    3. Check ``metadata.name`` values in ``face_db`` (case-insensitive).
    """
    mapped_id = _name_to_people_id().get(predicted_name)
    if mapped_id is not None:
        return str(mapped_id)

    normalized_name = predicted_name.strip().casefold()
    for face_id, record in face_db.items():
        if str(face_id).strip().casefold() == normalized_name:
            return str(face_id)

        metadata = record.get("metadata") if isinstance(record, dict) else None
        if not isinstance(metadata, dict):
            continue

        candidate_name = metadata.get("name")
        if isinstance(candidate_name, str) and candidate_name.strip().casefold() == normalized_name:
            return str(face_id)

    return None


# ------------------------------------------------------------------
# Primary recognition entry point (called from face_pipeline.py)
# ------------------------------------------------------------------

def dnn_face_recognition(frame: Any, face_db: dict[str, Any]) -> str | None:
    """Detect and identify the primary face in *frame*.

    Parameters
    ----------
    frame:
        BGR numpy array (decoded JPEG frame from Unity).
    face_db:
        Snapshot of ``AppState.face_db`` serialised to plain dicts.

    Returns
    -------
    The matched ``face_id`` string when the face is in ``face_db``.
    ``"stranger"`` when a face is detected but not recognised / not in ``face_db``.
    ``None`` when no face is detected in the frame at all.
    """
    recognizer = _get_active_recognizer()
    prediction = recognizer.predict_frame(frame)

    # No face present in the frame at all — return None so the system ignores
    # this frame rather than treating it as a stranger detection.
    if prediction.name is None:
        return None

    # A face was detected but its similarity/confidence is below the threshold —
    # the person is not in the enrollment store / training set.
    if prediction.name == "Unknown":
        return "stranger"

    if face_service_settings.recognizer_backend.lower() == "arcface":
        # ArcFace returns a face_id directly — accept it only if it exists in
        # the live face_db; otherwise the person is a stranger.
        return prediction.name if prediction.name in face_db else "stranger"

    # KNN path: map the predicted label back to a face_id via CSV / face_db.
    # If the resolution fails the recognised person is not in the face_db, so
    # treat them as a stranger.
    resolved = _resolve_face_id(prediction.name, face_db)
    return resolved if resolved is not None else "stranger"


# ------------------------------------------------------------------
# ArcFace enrollment helpers (called from main.py endpoints)
# ------------------------------------------------------------------

def enroll_arcface_from_image_bytes(face_id: str, image_bytes: bytes) -> bool:
    """Decode *image_bytes* and enrol the face under *face_id* for ArcFace.

    Returns ``True`` on success, ``False`` when no face is detected or the
    image cannot be decoded.  Returns ``False`` immediately when the KNN
    backend is active (no-op, not an error).
    """
    if face_service_settings.recognizer_backend.lower() != "arcface":
        return False
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return False
    return _arcface_recognizer().enroll_face(face_id, img)


def enroll_arcface_from_image_path(face_id: str, image_path: str) -> bool:
    """Load an image from disk and enrol it for ArcFace recognition.

    Returns ``True`` on success, ``False`` when the image cannot be read or
    no face is detected.  Returns ``False`` immediately when the KNN
    backend is active.
    """
    if face_service_settings.recognizer_backend.lower() != "arcface":
        return False
    img = cv2.imread(image_path)
    if img is None:
        return False
    return _arcface_recognizer().enroll_face(face_id, img)

