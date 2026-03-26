"""
Create epoch: build a single MNE Epochs object centered on an event timestamp.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Dict, Any
import numpy as np
import pandas as pd
import mne

from eeg_backend_functions.connect_eeg import _get_stream

logger = logging.getLogger(__name__)


def _nearest_indices(ts_array: np.ndarray, targets: Sequence[float]) -> np.ndarray:
    idxs = []
    for t in targets:
        j = int(np.argmin(np.abs(ts_array - float(t))))
        idxs.append(j)
    return np.asarray(idxs, dtype=int)


def create_epoch(
    event_lsl_timestamp: float,
    *,
    epoch_dur: Sequence[float] = (-1.0, 0.65),
    picks: Optional[Sequence[str]] = None,
    event_id: Dict[str, int] = None,
    event_label: str = "AR",
    channel_names: Optional[Sequence[str]] = None,
) -> mne.Epochs:
    """
    Create an epoch around an event time.

    Inputs
    ------
    event_lsl_timestamp : float
        Event time in LSL timebase.
    channel_names : sequence of str, optional
        Desired channel names in device output order.  When provided the stream
        channels are renamed positionally (CH1→channel_names[0], etc.) and a
        standard 10-20 montage is applied so downstream steps (e.g. REST
        re-referencing) work correctly.  Defaults to None (keep stream names).

    Outputs
    -------
    epochs : mne.Epochs
        A single-epoch MNE Epochs object (preloaded).

    Assumptions (matches your notebook workflow)
    --------------------------------------------
    - We can obtain a continuous MNE Raw object and an array of sample timestamps.
      In offline notebooks this came from eeg_df['TIMESTAMP'] and RawArray.
    - In the backend, your stream implementation must provide those two things.
    """
    if event_id is None:
        # Default: treat as one condition
        event_id = {event_label: 1}

    logger.info(
        "create_epoch: event_lsl_timestamp=%.6f  epoch_dur=[%.3f, %.3f]  event_label=%r  channel_names=%s",
        event_lsl_timestamp,
        float(epoch_dur[0]),
        float(epoch_dur[1]),
        event_label,
        list(channel_names) if channel_names is not None else None,
    )

    stream = _get_stream()
    logger.debug(
        "create_epoch: stream type=%s  attrs=%s",
        type(stream).__name__,
        [a for a in ("raw", "get_raw", "as_raw", "timestamps", "get_timestamps", "pull_window", "sfreq", "ch_names")
         if hasattr(stream, a)],
    )

    # ---- Adapters for stream implementations ----
    # We try a few common patterns:
    # 1) stream.raw returns an mne.io.Raw
    # 2) stream.get_raw() returns an mne.io.Raw
    # 3) stream.get_data() returns (data, times_s, sfreq, ch_names) in some form
    raw = None
    ts = None

    if hasattr(stream, "raw"):
        raw = getattr(stream, "raw")
        logger.debug("create_epoch: obtained raw via stream.raw")
    elif hasattr(stream, "get_raw"):
        raw = stream.get_raw()
        logger.debug("create_epoch: obtained raw via stream.get_raw()")
    elif hasattr(stream, "as_raw"):
        raw = stream.as_raw()
        logger.debug("create_epoch: obtained raw via stream.as_raw()")

    if raw is not None:
        logger.debug(
            "create_epoch: raw object — channels=%s  sfreq=%.1f  n_times=%d  duration=%.3fs",
            raw.ch_names,
            raw.info["sfreq"],
            raw.n_times,
            raw.times[-1] if raw.n_times > 0 else 0.0,
        )
        # Build a timestamp array in *LSL timebase* if stream provides it, else assume raw.times are aligned
        if hasattr(stream, "timestamps"):
            ts = np.asarray(getattr(stream, "timestamps"), dtype=float)
            logger.debug("create_epoch: timestamps via stream.timestamps  len=%d  range=[%.6f, %.6f]",
                         len(ts), ts.min() if len(ts) else float("nan"), ts.max() if len(ts) else float("nan"))
        elif hasattr(stream, "get_timestamps"):
            ts = np.asarray(stream.get_timestamps(), dtype=float)
            logger.debug("create_epoch: timestamps via stream.get_timestamps()  len=%d  range=[%.6f, %.6f]",
                         len(ts), ts.min() if len(ts) else float("nan"), ts.max() if len(ts) else float("nan"))
        else:
            # Fallback: treat raw.times as relative; zero at 0 and shift so that event_lsl_timestamp is relative
            # This fallback needs your backend to align event timestamps to the raw buffer timebase.
            ts = raw.times.astype(float)
            logger.warning(
                "create_epoch: no LSL timestamp array on stream — using raw.times as timebase "
                "(range=[%.6f, %.6f]).  Event timestamp must be in the same timebase.",
                ts[0] if len(ts) else float("nan"),
                ts[-1] if len(ts) else float("nan"),
            )
    else:
        # Last resort: attempt to pull a window from stream directly.
        if not hasattr(stream, "pull_window"):
            raise RuntimeError(
                "create_epoch() could not obtain a Raw buffer from the stream.\n"
                "Expected stream.raw / stream.get_raw() / stream.as_raw() or stream.pull_window()."
            )
        # Pull with extra padding so that discrete sample alignment never puts the event
        # sample too close to the buffer boundary (which causes MNE to drop the epoch).
        _PULL_PADDING = 0.5  # seconds of extra buffer on each side
        pull_start = float(event_lsl_timestamp) + float(epoch_dur[0]) - _PULL_PADDING
        pull_end   = float(event_lsl_timestamp) + float(epoch_dur[1]) + _PULL_PADDING
        logger.info(
            "create_epoch: pulling window [%.6f, %.6f]  (event±dur + %.1fs padding)",
            pull_start, pull_end, _PULL_PADDING,
        )
        data_uv, times = stream.pull_window(pull_start, pull_end)
        logger.info(
            "create_epoch: pulled data shape=%s  timestamps range=[%.6f, %.6f]  n_samples=%d",
            data_uv.shape,
            times[0] if len(times) else float("nan"),
            times[-1] if len(times) else float("nan"),
            len(times),
        )
        logger.debug(
            "create_epoch: data amplitude range (µV): min=%.2f  max=%.2f  mean_abs=%.2f",
            float(data_uv.min()),
            float(data_uv.max()),
            float(np.abs(data_uv).mean()),
        )
        # data_uv assumed shape (n_channels, n_times) in microvolts
        sfreq = float(getattr(stream, "sfreq", 250.0))
        ch_names = list(getattr(stream, "ch_names", [f"EEG{i}" for i in range(data_uv.shape[0])]))
        logger.debug(
            "create_epoch: building RawArray — sfreq=%.1f  ch_names=%s",
            sfreq, ch_names,
        )
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(np.asarray(data_uv, dtype=float) * 1e-6, info, verbose=False)
        ts = np.asarray(times, dtype=float)

    # Apply positional channel renaming and set montage if names were provided
    if channel_names is not None:
        desired = list(channel_names)[: raw.info["nchan"]]
        rename_map = {old: new for old, new in zip(raw.ch_names[: len(desired)], desired)}
        logger.debug("create_epoch: renaming channels %s", rename_map)
        raw.rename_channels(rename_map, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="ignore", verbose=False)
        logger.debug("create_epoch: montage applied — final ch_names=%s", raw.ch_names)
        # Restrict the epoch to only the renamed channels, dropping any extras (e.g. CH7, CH8).
        picks = [ch for ch in desired if ch in raw.ch_names]

    # Build single-event DataFrames to reuse epoching logic from notebook
    eeg_df = pd.DataFrame({"TIMESTAMP": ts})
    events_df = pd.DataFrame({"TIMESTAMP": [float(event_lsl_timestamp)], "ID": [event_label]})

    # Log alignment: how far the event timestamp sits from the buffer boundaries
    ts_min, ts_max = float(ts.min()), float(ts.max())
    pre_margin  = float(event_lsl_timestamp) - ts_min   # seconds between buffer start and event
    post_margin = ts_max - float(event_lsl_timestamp)   # seconds between event and buffer end
    need_pre    = abs(float(epoch_dur[0]))               # seconds needed before event
    need_post   = float(epoch_dur[1])                    # seconds needed after event
    logger.info(
        "create_epoch: buffer range=[%.6f, %.6f]  len=%d samples  sfreq=%.1f",
        ts_min, ts_max, len(ts), raw.info["sfreq"],
    )
    logger.info(
        "create_epoch: event at %.6f — pre_margin=%.3fs (need %.3fs)  post_margin=%.3fs (need %.3fs)  %s",
        float(event_lsl_timestamp),
        pre_margin, need_pre,
        post_margin, need_post,
        "OK" if pre_margin >= need_pre and post_margin >= need_post else "INSUFFICIENT MARGIN",
    )
    if pre_margin < need_pre or post_margin < need_post:
        raise RuntimeError(
            f"create_epoch: buffer margin too small — "
            f"pre={pre_margin:.3f}s (need {need_pre:.3f}s), "
            f"post={post_margin:.3f}s (need {need_post:.3f}s). "
            f"Buffer range=[{ts_min:.6f}, {ts_max:.6f}], event={float(event_lsl_timestamp):.6f}."
        )

    # Convert timestamp to sample index (same logic as in epoch_eeg_data)
    sample_indices = _nearest_indices(eeg_df["TIMESTAMP"].to_numpy(dtype=float), events_df["TIMESTAMP"].to_numpy())
    nearest_ts = float(ts[sample_indices[0]])
    logger.info(
        "create_epoch: event sample_index=%d  nearest_ts=%.6f  offset_from_event=%.4fms  "
        "samples_before=%d  samples_after=%d",
        int(sample_indices[0]),
        nearest_ts,
        (nearest_ts - float(event_lsl_timestamp)) * 1000.0,
        int(sample_indices[0]),
        len(ts) - 1 - int(sample_indices[0]),
    )
    events_mne = np.column_stack([sample_indices, np.zeros(1, dtype=int), np.array([event_id[event_label]])])

    epochs = mne.Epochs(
        raw,
        events=events_mne,
        event_id=event_id,
        picks=picks,
        tmin=float(epoch_dur[0]),
        tmax=float(epoch_dur[1]),
        baseline=None,
        preload=True,
        reject=None,
        flat=None,
        verbose=False,
    )

    logger.info(
        "create_epoch: mne.Epochs created — n_epochs=%d  tmin=%.3f  tmax=%.3f  ch_names=%s  drop_log=%s",
        len(epochs),
        epochs.tmin,
        epochs.tmax,
        epochs.ch_names,
        epochs.drop_log,
    )
    if len(epochs) == 0:
        logger.error(
            "create_epoch: ALL EPOCHS DROPPED.  drop_log=%s  "
            "event_sample=%d  raw_n_times=%d  tmin_samples=%d  tmax_samples=%d",
            epochs.drop_log,
            int(sample_indices[0]),
            raw.n_times,
            int(round(abs(float(epoch_dur[0])) * raw.info["sfreq"])),
            int(round(float(epoch_dur[1]) * raw.info["sfreq"])),
        )

    return epochs
