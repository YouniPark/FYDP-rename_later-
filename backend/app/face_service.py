from __future__ import annotations

import asyncio
import base64
import imghdr
from typing import Any

import cv2
import numpy as np

from app.adapters.existing_functions import call_opencv_dnn_face_recognition
from app.config import ALLOWED_IMAGE_FORMATS, MAX_IMAGE_BYTES
from app.db import Database
from app.state import AppStateCache


class FaceService:
    def __init__(self, db: Database, state_cache: AppStateCache) -> None:
        self.db = db
        self.state_cache = state_cache

    def _decode_image(self, image_base64: str) -> np.ndarray:
        raw = base64.b64decode(image_base64, validate=True)
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError(f"Image exceeds max size of {MAX_IMAGE_BYTES} bytes")
        image_format = imghdr.what(None, h=raw)
        if image_format not in ALLOWED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image format: {image_format}")
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img

    async def handle_face_input(self, timestamp: float, image_base64: str, face_json: dict[str, Any]) -> dict[str, Any]:
        image = await asyncio.to_thread(self._decode_image, image_base64)
        face_db, _, _ = self.db.list_faces()
        result = await asyncio.to_thread(call_opencv_dnn_face_recognition, image, face_db, face_json)
        self.state_cache.set_face_result(timestamp, result)
        self.db.insert_recognition_record(timestamp=timestamp, result=result)

        # Optional: auto-add unknown faces into face db when upstream includes enrollment payload.
        if result.get("enroll"):
            self.db.add_or_update_face(result["enroll"], external_id=result["enroll"].get("id"))

        return result
