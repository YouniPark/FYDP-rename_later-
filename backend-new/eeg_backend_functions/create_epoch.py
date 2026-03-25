"""
Create epoch: build a single MNE Epochs object centered on an event timestamp.
"""

from __future__ import annotations

from typing import Optional, Sequence, Dict, Any
import numpy as np
import pandas as pd
import mne

from eeg_backend_functions.connect_eeg import _get_stream


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

    stream = _get_stream()

    # ---- Adapters for stream implementations ----
    # We try a few common patterns:
    # 1) stream.raw returns an mne.io.Raw
    # 2) stream.get_raw() returns an mne.io.Raw
    # 3) stream.get_data() returns (data, times_s, sfreq, ch_names) in some form
    raw = None
    ts = None

    if hasattr(stream, "raw"):
        raw = getattr(stream, "raw")
    elif hasattr(stream, "get_raw"):
        raw = stream.get_raw()
    elif hasattr(stream, "as_raw"):
        raw = stream.as_raw()

    if raw is not None:
        # Build a timestamp array in *LSL timebase* if stream provides it, else assume raw.times are aligned
        if hasattr(stream, "timestamps"):
            ts = np.asarray(getattr(stream, "timestamps"), dtype=float)
        elif hasattr(stream, "get_timestamps"):
            ts = np.asarray(stream.get_timestamps(), dtype=float)
        else:
            # Fallback: treat raw.times as relative; zero at 0 and shift so that event_lsl_timestamp is relative
            # This fallback needs your backend to align event timestamps to the raw buffer timebase.
            ts = raw.times.astype(float)
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
        data_uv, times = stream.pull_window(
            float(event_lsl_timestamp) + float(epoch_dur[0]) - _PULL_PADDING,
            float(event_lsl_timestamp) + float(epoch_dur[1]) + _PULL_PADDING,
        )
        # data_uv assumed shape (n_channels, n_times) in microvolts
        sfreq = float(getattr(stream, "sfreq", 250.0))
        ch_names = list(getattr(stream, "ch_names", [f"EEG{i}" for i in range(data_uv.shape[0])]))
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(np.asarray(data_uv, dtype=float) * 1e-6, info, verbose=False)
        ts = np.asarray(times, dtype=float)

    # Apply positional channel renaming and set montage if names were provided
    if channel_names is not None:
        desired = list(channel_names)[: raw.info["nchan"]]
        rename_map = {old: new for old, new in zip(raw.ch_names[: len(desired)], desired)}
        raw.rename_channels(rename_map, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="ignore", verbose=False)

    # Build single-event DataFrames to reuse epoching logic from notebook
    eeg_df = pd.DataFrame({"TIMESTAMP": ts})
    events_df = pd.DataFrame({"TIMESTAMP": [float(event_lsl_timestamp)], "ID": [event_label]})

    # Convert timestamp to sample index (same logic as in epoch_eeg_data)
    sample_indices = _nearest_indices(eeg_df["TIMESTAMP"].to_numpy(dtype=float), events_df["TIMESTAMP"].to_numpy())
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
        verbose=True,
    )

    return epochs
