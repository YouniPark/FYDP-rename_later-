import asyncio
import inspect
import logging
from typing import Any

import numpy as np

from app.state import AppState
from eeg_backend_functions.eeg_processing import EpochRejectedError

logger = logging.getLogger(__name__)

try:
    from user_modules.eeg import connect_eeg, create_epoch, eeg_processing, event_filter
    from user_modules.model import ml_classifier
except ImportError:
    def connect_eeg() -> Any:
        raise NotImplementedError("TO DO: implement connect_eeg import from user modules")

    def event_filter(event_lsl_timestamp: float) -> bool:
        raise NotImplementedError("TO DO: implement event_filter import from user modules")

    def create_epoch(stream: Any, event_lsl_timestamp: float) -> Any:
        raise NotImplementedError("TO DO: implement create_epoch import from user modules")

    def eeg_processing(epoch: Any) -> np.ndarray:
        raise NotImplementedError("TO DO: implement eeg_processing import from user modules")

    def ml_classifier(features: np.ndarray) -> bool:
        raise NotImplementedError("TO DO: implement ml_classifier import from user modules")


async def eeg_connect_loop(state: AppState) -> None:
    while True:
        try:
            stream = await asyncio.to_thread(connect_eeg)
            if stream is None:
                await state.set_eeg_stream(None)
                logger.info("EEG LSL stream not found; retrying")
                await asyncio.sleep(state.settings.eeg_lsl_retry_seconds)
                continue
            await state.set_eeg_stream(stream)
            logger.info("EEG LSL connected")
            await asyncio.sleep(state.settings.eeg_lsl_retry_seconds)
        except Exception:
            logger.exception("EEG connection attempt failed")
            await state.set_eeg_stream(None)
            await asyncio.sleep(state.settings.eeg_lsl_retry_seconds)


def _create_epoch_wrapper(stream: Any, event_lsl_timestamp: float) -> Any:
    sig = inspect.signature(create_epoch)
    positional_count = sum(
        parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for parameter in sig.parameters.values()
    )
    if positional_count <= 1:
        return create_epoch(event_lsl_timestamp)
    return create_epoch(stream, event_lsl_timestamp)


async def run_eeg_event_pipeline(state: AppState, event_id: str, event_lsl_timestamp: float) -> dict[str, Any]:
    stream = await state.get_eeg_stream()
    if stream is None:
        if state.settings.unfamiliar_if_no_eeg:
            logger.info(
                "EEG stream not connected for event %s; treating as unfamiliar",
                event_id,
            )
            result = {
                "event_id": event_id,
                "event_lsl_timestamp": event_lsl_timestamp,
                "status": "ok",
                "is_unfamiliar": True,
                "reason": "eeg_stream_not_connected",
            }
        else:
            logger.info(
                "EEG stream not connected for event %s; ignoring event",
                event_id,
            )
            result = {
                "event_id": event_id,
                "event_lsl_timestamp": event_lsl_timestamp,
                "status": "no_eeg",
            }
        state.latest_eeg_result[event_id] = result
        return result

    try:
        should_process = await asyncio.to_thread(event_filter, event_lsl_timestamp)
        if not should_process:
            result = {
                "event_id": event_id,
                "event_lsl_timestamp": event_lsl_timestamp,
                "status": "filtered",
                "is_unfamiliar": False,
            }
            state.latest_eeg_result[event_id] = result
            return result

        epoch = await asyncio.to_thread(_create_epoch_wrapper, stream, event_lsl_timestamp)
        features = await asyncio.to_thread(
            eeg_processing,
            epoch,
            l_freq=state.settings.eeg_l_freq,
            h_freq=state.settings.eeg_h_freq,
            notch_freqs=state.settings.eeg_notch_freqs,
            ica_path=state.settings.eeg_ica_path,
            apply_rest=state.settings.eeg_apply_rest,
            forward_path=state.settings.eeg_forward_path,
            baseline_window=(state.settings.eeg_baseline_tmin, state.settings.eeg_baseline_tmax),
            amp_thresh=state.settings.eeg_amp_thresh_uv * 1e-6,
        )
        is_unfamiliar = await asyncio.to_thread(
            ml_classifier,
            features,
            model_path=state.settings.eeg_model_path,
            scaler_path=state.settings.eeg_scaler_path,
        )
        result = {
            "event_id": event_id,
            "event_lsl_timestamp": event_lsl_timestamp,
            "status": "ok",
            "is_unfamiliar": bool(is_unfamiliar),
        }
        state.latest_eeg_result[event_id] = result
        return result
    except EpochRejectedError as exc:
        # Artifact-driven rejection: treat as unfamiliar but log separately for analysis
        logger.warning(
            "Epoch rejected for event %s (artifact): %s",
            event_id,
            exc,
            extra={"event_id": event_id, "bad_channels": exc.bad_channels},
        )
        result = {
            "event_id": event_id,
            "event_lsl_timestamp": event_lsl_timestamp,
            "status": "rejected",
            "is_unfamiliar": True,
            "bad_channels": exc.bad_channels,
        }
        state.latest_eeg_result[event_id] = result
        return result
    except Exception as exc:
        logger.exception("EEG event pipeline failed", extra={"event_id": event_id})
        return {
            "event_id": event_id,
            "event_lsl_timestamp": event_lsl_timestamp,
            "status": "error",
            "reason": str(exc),
        }
