from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.face_service.mapping import load_name_to_people_id
from app.face_service.recognizer_runtime import FaceRuntimeRecognizer
from app.face_service.settings import face_service_settings


@lru_cache(maxsize=1)
def _recognizer() -> FaceRuntimeRecognizer:
    return FaceRuntimeRecognizer(face_service_settings)


@lru_cache(maxsize=1)
def _name_to_people_id() -> dict[str, int]:
    try:
        return load_name_to_people_id(face_service_settings.mapping_csv_path)
    except Exception:
        return {}


def _resolve_face_id(predicted_name: str, face_db: dict[str, Any]) -> str | None:
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


def dnn_face_recognition(frame: Any, face_db: dict[str, Any]) -> str | None:
    prediction = _recognizer().predict_frame(frame)
    if prediction.name is None or prediction.name == "Unknown":
        return None
    return _resolve_face_id(prediction.name, face_db)
