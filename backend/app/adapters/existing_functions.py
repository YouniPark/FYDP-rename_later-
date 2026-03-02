from __future__ import annotations

from typing import Any

import numpy as np

try:
    # TODO: Replace `external_impl` with your real module path if names differ.
    from external_impl import (  # type: ignore
        connect_eeg_lsl,
        create_epoch,
        event_filter,
        ml_classifier_predict,
        opencv_dnn_face_recognition,
        prepare_cue,
        process_eeg_epoch,
    )
except Exception:  # pragma: no cover
    def _missing(*_: Any, **__: Any):
        raise NotImplementedError("TODO: wire real existing functions in app/adapters/existing_functions.py")

    connect_eeg_lsl = _missing
    event_filter = _missing
    create_epoch = _missing
    process_eeg_epoch = _missing
    ml_classifier_predict = _missing
    opencv_dnn_face_recognition = _missing
    prepare_cue = _missing


def call_connect_eeg_lsl():
    return connect_eeg_lsl()


def call_event_filter(event_lsl_timestamp: float) -> bool:
    return event_filter(event_lsl_timestamp)


def call_create_epoch(stream: Any, event_lsl_timestamp: float):
    return create_epoch(stream, event_lsl_timestamp)


def call_process_eeg_epoch(epoch: Any) -> np.ndarray:
    return process_eeg_epoch(epoch)


def call_ml_classifier_predict(features: np.ndarray) -> bool:
    return ml_classifier_predict(features)


def call_opencv_dnn_face_recognition(image: Any, face_db: list[dict[str, Any]], face_json: dict[str, Any]) -> dict[str, Any]:
    # TODO: If your real function signature differs, adapt only here.
    return opencv_dnn_face_recognition(image, face_db, face_json)


def call_prepare_cue(
    classifier_flag: bool | None,
    face_result: dict[str, Any] | None,
    cue_json: dict[str, Any] | None,
    cue_db: list[dict[str, Any]],
) -> dict[str, Any]:
    return prepare_cue(classifier_flag, face_result, cue_json, cue_db)
