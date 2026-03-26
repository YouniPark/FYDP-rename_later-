from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.face_service.settings import FaceServiceSettings


@dataclass
class FramePrediction:
    name: str | None
    confidence: float | None


class FaceRuntimeRecognizer:
    """Loads DNN detector/embedder + sklearn classifier from configured paths."""

    def __init__(self, settings: FaceServiceSettings):
        self.settings = settings

        self.detector_prototxt_path = self._resolve_detector_prototxt()
        self.detector_caffemodel_path = self._resolve_detector_caffemodel()
        self.embedder_model_path = settings.embedder_model_path
        self.recognizer_model_path = settings.recognizer_model_path

        self._validate_required_paths(
            [
                self.detector_prototxt_path,
                self.detector_caffemodel_path,
                self.embedder_model_path,
                self.recognizer_model_path,
            ]
        )

        self.detector = cv2.dnn.readNetFromCaffe(
            str(self.detector_prototxt_path),
            str(self.detector_caffemodel_path),
        )
        if self.detector.empty():
            raise RuntimeError("Failed to load face detector network")

        self.embedder = cv2.dnn.readNetFromTorch(str(self.embedder_model_path))
        if self.embedder.empty():
            raise RuntimeError("Failed to load face embedding network")

        with self.recognizer_model_path.open("rb") as f:
            loaded = pickle.load(f)
            self.model, self.label_encoder = loaded[0], loaded[1]

    def predict_frame(self, frame_bgr: np.ndarray) -> FramePrediction:
        face = self._extract_primary_face(frame_bgr)
        if face is None:
            return FramePrediction(name=None, confidence=None)

        embedding = self._embedding(face)
        probs = self.model.predict_proba([embedding])[0]
        pred_idx = int(np.argmax(probs))
        pred_encoded = self.model.predict([embedding])[0]
        pred_name = str(self.label_encoder.inverse_transform([pred_encoded])[0])
        pred_prob = float(probs[pred_idx])

        if pred_prob < self.settings.unknown_threshold:
            return FramePrediction(name="Unknown", confidence=pred_prob)

        return FramePrediction(name=pred_name, confidence=pred_prob)

    def _extract_primary_face(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300), (104, 177, 123), swapRB=False)
        self.detector.setInput(blob)
        detections = self.detector.forward()

        best_face: np.ndarray | None = None
        best_area = 0

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.settings.detector_conf_threshold:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_face = frame_bgr[y1:y2, x1:x2]

        return best_face

    def _embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        face_blob = cv2.dnn.blobFromImage(
            face_bgr,
            1.0 / 255,
            (96, 96),
            (0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.embedder.setInput(face_blob)
        return self.embedder.forward().flatten()

    def _resolve_detector_prototxt(self) -> Path:
        if self.settings.detector_prototxt_path.exists():
            return self.settings.detector_prototxt_path
        return self.settings.detector_prototxt_fallback_path

    def _resolve_detector_caffemodel(self) -> Path:
        if self.settings.detector_caffemodel_path.exists():
            return self.settings.detector_caffemodel_path
        return self.settings.detector_caffemodel_fallback_path

    @staticmethod
    def _validate_required_paths(paths: list[Path]) -> None:
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required face runtime assets:\n" + "\n".join(missing)
            )
