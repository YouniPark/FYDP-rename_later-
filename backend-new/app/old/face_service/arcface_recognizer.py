"""InsightFace ArcFace runtime recognizer."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.face_service.embedding_store import ArcFaceEmbeddingStore
from app.face_service.recognizer_runtime import FramePrediction
from app.face_service.settings import FaceServiceSettings

logger = logging.getLogger(__name__)


def _bbox_area(bbox: object) -> float:
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return max(0.0, (x2 - x1) * (y2 - y1))


class ArcFaceRuntimeRecognizer:
    """Face recogniser built on InsightFace's ArcFace model.

    Detection is performed by the bundled RetinaFace detector (part of the
    ``buffalo_l`` / ``buffalo_m`` / ``buffalo_sc`` model packs).  Recognition
    uses 512-D ArcFace embeddings matched against the enrolled embeddings in
    *embedding_store* via cosine similarity.

    Key differences from the KNN recogniser
    ----------------------------------------
    * No pre-training step — enrol faces at runtime via :meth:`enroll_face`.
    * Enrolled faces are looked up by ``face_id`` directly; no CSV mapping is
      needed.
    * Prediction returns ``FramePrediction(name=face_id, ...)`` so that the
      existing ``_resolve_face_id`` call in ``user_modules.face`` is bypassed.

    Requirements
    ------------
    ``insightface>=0.7.3`` and ``onnxruntime>=1.18.0`` (CPU) or
    ``onnxruntime-gpu`` must be installed.  These are listed in
    ``requirements.txt`` but are *not* imported at module level so that the
    KNN backend can still be used in environments where InsightFace is
    unavailable.
    """

    def __init__(
        self,
        settings: FaceServiceSettings,
        embedding_store: ArcFaceEmbeddingStore,
    ) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "insightface is required for the ArcFace backend. "
                "Install it with: pip install insightface onnxruntime"
            ) from exc

        self.settings = settings
        self.store = embedding_store

        self._app = FaceAnalysis(
            name=settings.arcface_model_name,
            root=str(settings.arcface_model_dir),
            providers=["CPUExecutionProvider"],
        )
        # ctx_id=-1 → CPU inference; set to 0 for CUDA GPU
        self._app.prepare(ctx_id=-1, det_size=(640, 640))

        logger.info(
            "ArcFaceRuntimeRecognizer ready — model=%s  enrolled=%d  threshold=%.2f",
            settings.arcface_model_name,
            len(embedding_store),
            settings.arcface_similarity_threshold,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_frame(self, frame_bgr: np.ndarray) -> FramePrediction:
        """Detect the primary face in *frame_bgr* and match against the store.

        Returns
        -------
        ``FramePrediction(name=face_id, confidence=cosine_score)``
            Recognised face whose cosine similarity exceeds the threshold.
        ``FramePrediction(name="Unknown", confidence=best_score)``
            A face was detected but the best match is below the threshold, or
            the embedding store is empty.
        ``FramePrediction(name=None, confidence=None)``
            No face detected in the frame.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = self._app.get(rgb)
        if not faces:
            return FramePrediction(name=None, confidence=None)

        # Use the largest detected face (most prominent in frame)
        primary = max(faces, key=lambda f: _bbox_area(f.bbox))
        embedding = np.asarray(primary.embedding, dtype=np.float32)

        face_id, score = self.store.find_closest(
            embedding, self.settings.arcface_similarity_threshold
        )
        if face_id is None:
            return FramePrediction(name="Unknown", confidence=float(max(score, 0.0)))

        return FramePrediction(name=face_id, confidence=float(score))

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def enroll_face(self, face_id: str, image_bgr: np.ndarray) -> bool:
        """Extract an ArcFace embedding from *image_bgr* and store it for *face_id*.

        Parameters
        ----------
        face_id:
            The identifier under which the embedding is stored.  Should match
            the ``face_id`` in ``face_db``.
        image_bgr:
            BGR image containing the person's face (any size).

        Returns
        -------
        ``True`` on success, ``False`` when no face is detected in the image.
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        faces = self._app.get(rgb)
        if not faces:
            logger.warning(
                "ArcFace enrollment: no face detected for face_id=%s", face_id
            )
            return False

        primary = max(faces, key=lambda f: _bbox_area(f.bbox))
        self.store.upsert(str(face_id), np.asarray(primary.embedding, dtype=np.float32))
        logger.info(
            "ArcFace enrolled face_id=%s  det_score=%.3f  store_size=%d",
            face_id,
            float(primary.det_score),
            len(self.store),
        )
        return True
