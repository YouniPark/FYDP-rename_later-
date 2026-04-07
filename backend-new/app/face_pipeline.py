import asyncio
import base64
import logging
from typing import Any

import cv2
import numpy as np

from app.state import AppState
from app.face_memory import face_memory_voter

logger = logging.getLogger(__name__)

try:
    from user_modules.face import dnn_face_recognition
except ImportError:
    # TO DO: implement import from user face module
    def dnn_face_recognition(frame: Any, face_db: dict[str, Any]) -> str | None:
        raise NotImplementedError("TO DO: implement dnn_face_recognition import from user modules")


async def enqueue_frame(state: AppState, timestamp: float, data_b64: str, encoding: str) -> None:
    if encoding.lower() != "jpeg":
        raise ValueError("Only jpeg encoding is supported")
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except Exception as exc:
        raise ValueError("Malformed base64 frame data") from exc
    if state.frame_queue.full():
        _ = state.frame_queue.get_nowait()
    await state.frame_queue.put((timestamp, raw))


def _decode_jpeg(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid jpeg frame")
    return frame


async def face_recognition_loop(state: AppState, debug_publish=None) -> None:
    while True:
        timestamp, image_bytes = await state.frame_queue.get()
        try:
            frame = await asyncio.to_thread(_decode_jpeg, image_bytes)
            async with state.face_db_lock:
                face_db_view = {k: v.model_dump(mode="json") for k, v in state.face_db.items()}

            face_id = await asyncio.to_thread(dnn_face_recognition, frame, face_db_view)
            await state.set_current_face(face_id)
            vote = await face_memory_voter.add_detection(timestamp, face_id)
            # logger.info(
            #     "Processed frame",
            #     extra={
            #         "frame_ts": timestamp,
            #         "face_id": face_id,
            #         "voted_face_id": vote.face_id,
            #         "vote_confidence": vote.confidence,
            #         "vote_sample_count": vote.sample_count,
            #     },
            # )

            if debug_publish is not None:
                frame_b64 = base64.b64encode(image_bytes).decode("ascii")
                await debug_publish(vote, frame_b64, timestamp)
        except Exception:
            logger.exception("Face recognition loop error")
        finally:
            state.frame_queue.task_done()
