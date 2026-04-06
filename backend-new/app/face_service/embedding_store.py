"""Persistent store mapping face_id → normalised ArcFace embeddings."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class ArcFaceEmbeddingStore:
    """Persistent mapping of ``face_id`` → normalised ArcFace embedding (512-D float32).

    Embeddings are stored pre-normalised so that cosine similarity reduces to a
    plain dot product at query time, keeping ``find_closest`` fast.

    The store is backed by a pickle file that is rewritten after every
    mutation (``upsert`` / ``remove``).  All mutating calls are synchronous
    and not thread-safe on their own; callers that run in an async context
    should protect them with an ``asyncio.Lock``.
    """

    def __init__(self, store_path: Path) -> None:
        self._path = Path(store_path)
        self._embeddings: dict[str, list[np.ndarray]] = self._load() ## changed to list
        logger.info(
            "ArcFaceEmbeddingStore: loaded %d embedding(s) from %s",
            len(self._embeddings),
            self._path,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, np.ndarray]:
        if not self._path.exists():
            return {}
        try:
            with self._path.open("rb") as fh:
                data = pickle.load(fh)
            if isinstance(data, dict):
                return {str(k): np.asarray(v, dtype=np.float32) for k, v in data.items()}
        except Exception:
            logger.exception(
                "ArcFaceEmbeddingStore: failed to load from %s — starting with empty store",
                self._path,
            )
        return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("wb") as fh:
            pickle.dump(self._embeddings, fh)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def upsert(self, face_id: str, embedding: np.ndarray) -> None:
        """Store *embedding* for *face_id*, overwriting any previous value.

        The embedding is L2-normalised before storage so that cosine similarity
        is equivalent to a dot product at query time.
        """
        vec = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        face_id = str(face_id)

        if face_id not in self._embeddings:
            self._embeddings[str(face_id)] = []

        self._embeddings[str(face_id)].append(vec)
        self._save()

    def remove(self, face_id: str) -> bool:
        """Remove the embedding for *face_id*.  Returns True if it existed."""
        if str(face_id) in self._embeddings:
            del self._embeddings[str(face_id)]
            self._save()
            return True
        return False

    def get_all(self) -> dict[str, np.ndarray]:
        """Return a shallow copy of the current store."""
        return dict(self._embeddings)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def find_closest(
        self,
        query: np.ndarray,
        threshold: float,
    ) -> tuple[str | None, float]:
        """Cosine-similarity nearest-neighbour search.

        Parameters
        ----------
        query:
            Raw (unnormalised) query embedding.
        threshold:
            Minimum cosine similarity to accept a match.  Values below this
            are treated as *Unknown*.

        Returns
        -------
        ``(face_id, score)`` of the closest stored embedding when
        *score* ≥ *threshold*, or ``(None, best_score)`` otherwise.
        Returns ``(None, 0.0)`` when the store is empty.
        """
        if not self._embeddings:
            return None, 0.0

        q = np.asarray(query, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return None, 0.0
        q = q / q_norm

        best_id: str | None = None
        best_score: float = float("-inf")  # tracks best cosine similarity seen so far

        for fid, emb_list in self._embeddings.items():
            for emb in emb_list:
                score = float(np.dot(q, emb))
                if score > best_score:
                    best_score = score
                    best_id = fid

        if best_score < threshold:
            return None, best_score

        return best_id, best_score

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._embeddings)

    def __contains__(self, face_id: object) -> bool:
        return str(face_id) in self._embeddings
