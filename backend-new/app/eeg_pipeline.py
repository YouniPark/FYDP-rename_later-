import asyncio
import inspect
import logging
from typing import Any, Optional, Sequence

import numpy as np

from app.state import AppState
from eeg_backend_functions.eeg_processing import EpochRejectedError

logger = logging.getLogger(__name__)

try:
    from user_modules.eeg import connect_eeg, create_epoch, eeg_processing
    from user_modules.model import ml_classifier
except ImportError:
    def connect_eeg() -> Any:
        raise NotImplementedError("TO DO: implement connect_eeg import from user modules")

    def create_epoch(stream: Any, event_lsl_timestamp: float) -> Any:
        raise NotImplementedError("TO DO: implement create_epoch import from user modules")

    def eeg_processing(epoch: Any) -> tuple:
        raise NotImplementedError("TO DO: implement eeg_processing import from user modules")

    def ml_classifier(features: np.ndarray) -> bool:
        raise NotImplementedError("TO DO: implement ml_classifier import from user modules")


async def eeg_connect_loop(state: AppState) -> None:
    stream_name = state.settings.eeg_lsl_stream_name
    retry_seconds = state.settings.eeg_lsl_retry_seconds
    while True:
        # Only attempt connection when we don't already have a stream.
        current = await state.get_eeg_stream()
        if current is not None:
            await asyncio.sleep(retry_seconds)
            continue
        try:
            stream = await asyncio.to_thread(connect_eeg, stream_name)
            if stream is None:
                logger.warning("EEG LSL: No stream named '%s' found", stream_name)
                logger.info("EEG LSL: retrying in %ds", retry_seconds)
                await asyncio.sleep(retry_seconds)
                continue
            await state.set_eeg_stream(stream)
            logger.info("EEG LSL: connected to '%s'", stream_name)
        except Exception:
            logger.exception("EEG LSL: connection attempt failed")
            await state.set_eeg_stream(None)
            await asyncio.sleep(retry_seconds)


def _create_epoch_wrapper(
    stream: Any,
    event_lsl_timestamp: float,
    epoch_tmin: float,
    epoch_tmax: float,
    channel_names: Optional[Sequence[str]] = None,
) -> Any:
    sig = inspect.signature(create_epoch)
    positional_count = sum(
        parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for parameter in sig.parameters.values()
    )
    epoch_dur = (epoch_tmin, epoch_tmax)
    kwargs = {"epoch_dur": epoch_dur}
    if channel_names is not None and "channel_names" in sig.parameters:
        kwargs["channel_names"] = channel_names
    if positional_count <= 1:
        return create_epoch(event_lsl_timestamp, **kwargs)
    return create_epoch(stream, event_lsl_timestamp, **kwargs)


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
        epoch_tmin = state.settings.eeg_epoch_tmin
        epoch_tmax = state.settings.eeg_epoch_tmax
        poll_interval = state.settings.eeg_buffer_poll_interval
        poll_timeout = state.settings.eeg_buffer_poll_timeout

        # Target: the buffer must contain data at least up to the end of the epoch window.
        # We poll until the stream's latest buffered timestamp crosses that threshold,
        # which handles variable Bluetooth delay robustly.
        epoch_end_ts = event_lsl_timestamp + epoch_tmax
        waited = 0.0
        while True:
            latest = await asyncio.to_thread(stream.latest_timestamp)
            if latest is not None and latest >= epoch_end_ts:
                break
            if waited >= poll_timeout:
                raise RuntimeError(
                    f"Timed out after {poll_timeout:.1f}s waiting for EEG buffer to reach "
                    f"{epoch_end_ts:.3f} (latest={latest})"
                )
            await asyncio.sleep(poll_interval)
            waited += poll_interval

        logger.info(
            "EEG pipeline: buffer ready (waited %.2fs, latest_ts=%.3f) for event=%s",
            waited,
            latest,
            event_id,
        )

        # Check that the buffer holds enough *pre-event* history for the epoch window.
        epoch_start_ts = event_lsl_timestamp + epoch_tmin  # epoch_tmin is negative
        earliest = await asyncio.to_thread(stream.earliest_timestamp)
        if earliest is None or earliest > epoch_start_ts:
            logger.warning(
                "EEG pipeline: insufficient pre-event buffer history for event=%s — "
                "need data from %.3f but buffer starts at %s (stream may have just connected). "
                "Treating as unfamiliar.",
                event_id,
                epoch_start_ts,
                f"{earliest:.3f}" if earliest is not None else "<empty>",
            )
            result = {
                "event_id": event_id,
                "event_lsl_timestamp": event_lsl_timestamp,
                "status": "ok",
                "is_unfamiliar": True,
                "reason": "insufficient_eeg_buffer_history",
            }
            state.latest_eeg_result[event_id] = result
            return result

        epoch = await asyncio.to_thread(
            _create_epoch_wrapper, stream, event_lsl_timestamp, epoch_tmin, epoch_tmax,
            state.settings.eeg_channel_names,
        )
        features, feature_names = await asyncio.to_thread(
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
            ignore_trial_rejection=state.settings.ignore_trial_rejection,
        )
        logger.debug(
            "EEG pipeline: eeg_processing returned features shape=%s, "
            "n_nans=%d, range=[%.3f, %.3f]",
            features.shape,
            np.isnan(features).sum(),
            np.nanmin(features),
            np.nanmax(features),
        )
        is_unfamiliar = await asyncio.to_thread(
            ml_classifier,
            features,
            model_path=state.settings.eeg_model_path,
            scaler_path=state.settings.eeg_scaler_path,
            raw_csv_path=state.settings.eeg_features_raw_csv_path if state.settings.eeg_save_erp_features else None,
            scaled_csv_path=state.settings.eeg_features_scaled_csv_path if state.settings.eeg_save_erp_features else None,
            feature_names=feature_names,
        )
        logger.info(
            "EEG pipeline: classifier result for event=%s — is_unfamiliar=%s",
            event_id,
            is_unfamiliar,
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
        logger.exception(
            "EEG event pipeline failed for event %s — treating as unfamiliar",
            event_id,
            extra={"event_id": event_id},
        )
        result = {
            "event_id": event_id,
            "event_lsl_timestamp": event_lsl_timestamp,
            "status": "error",
            "is_unfamiliar": True,
            "reason": str(exc),
        }
        state.latest_eeg_result[event_id] = result
        return result
