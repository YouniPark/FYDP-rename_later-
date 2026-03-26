"""
Connect EEG: establish an LSL EEG stream connection.

updated to use pylsl StreamInlet-based context (from eeg_to_lsl.py)

- create_epoch.py calls connect_eeg._get_stream()
- create_epoch.py can fall back to stream.pull_window(start_ts, end_ts)
  (implemented by EEGStreamContext below)

Environment variables
--------------------------------
EEG_LSL_STREAM_TYPE: default "EEG"
EEG_LSL_STREAM_NAME: if set, resolve by name first
EEG_LSL_RESOLVE_TIMEOUT: default 5.0 seconds
EEG_DEFAULT_SFREQ: default 250 Hz (used if nominal_srate is 0)
EEG_BUFFER_SECONDS: default 8.0 seconds (ring buffer length)
EEG_CONNECT_WARMUP_SECONDS: default 0.4 seconds
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Tuple, List

import numpy as np

logger = logging.getLogger(__name__)

# Map any device-specific channel labels to your preferred names.
# If you don't need this, leave it empty.
DEFAULT_CHANNEL_MAPPING: dict[str, str] = {}

# Module-level reference for other functions to use (create_epoch, etc.)
_STREAM = None


@dataclass
class EEGStreamContext:
    """
    Small wrapper around pylsl.StreamInlet that maintains a rolling buffer of
    samples + LSL timestamps, and can return a time-window slice.

    This is intentionally minimal so it can be used by create_epoch() without
    requiring major changes elsewhere.
    """
    inlet: object
    channel_names: List[str]
    sfreq: float
    buffer_seconds: float = 3

    # Internal rolling buffers (timestamps in LSL timebase)
    _ts: deque = field(default_factory=deque, init=False, repr=False)
    _x: deque = field(default_factory=deque, init=False, repr=False)

    @property
    def ch_names(self) -> List[str]:
        return self.channel_names

    def _maxlen_samples(self) -> int:
        # Keep a little extra to avoid edge trimming when selecting windows.
        return max(1, int(self.buffer_seconds * float(self.sfreq)) + int(0.25 * float(self.sfreq)))

    def _update_buffer(self) -> None:
        """
        Pull all currently available samples from LSL into our rolling buffer.
        Uses zero-timeout pulls so it won't block your backend loop.
        """
        # pylsl.StreamInlet supports pull_chunk(timeout=0.0)
        try:
            chunk, ts = self.inlet.pull_chunk(timeout=0.0)  # type: ignore[attr-defined]
        except Exception:
            # Some pylsl versions want max_samples; try a compatible call.
            chunk, ts = self.inlet.pull_chunk(0.0)  # type: ignore[misc]

        if not ts:
            return

        # chunk is list[list[float]] shaped (n_samples, n_channels)
        chunk_arr = np.asarray(chunk, dtype=float)
        ts_arr = np.asarray(ts, dtype=float)

        # Append sample-by-sample to keep ordering simple
        for i in range(len(ts_arr)):
            self._ts.append(float(ts_arr[i]))
            self._x.append(chunk_arr[i])

        # Trim to max length
        maxlen = self._maxlen_samples()
        while len(self._ts) > maxlen:
            self._ts.popleft()
            self._x.popleft()

    def latest_timestamp(self) -> Optional[float]:
        """Pull any pending samples then return the newest timestamp in the buffer, or None if empty."""
        self._update_buffer()
        return float(self._ts[-1]) if self._ts else None

    def earliest_timestamp(self) -> Optional[float]:
        """Return the oldest timestamp currently in the buffer without pulling new data."""
        return float(self._ts[0]) if self._ts else None

    def pull_window(self, start_ts: float, end_ts: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (data, timestamps) for samples whose LSL timestamps fall within
        [start_ts, end_ts].

        Returns
        -------
        data : np.ndarray, shape (n_channels, n_times)
            Data in *microvolts* if your stream is in microvolts. If your device
            outputs volts already, adjust here (or in create_epoch()).
        timestamps : np.ndarray, shape (n_times,)
            LSL timestamps for each sample.
        """
        self._update_buffer()

        if len(self._ts) == 0:
            raise RuntimeError("EEG buffer is empty (no samples received yet).")

        ts = np.asarray(self._ts, dtype=float)

        # Select window
        mask = (ts >= float(start_ts)) & (ts <= float(end_ts))
        idx = np.where(mask)[0]

        if idx.size == 0:
            # Helpful error for debugging timing alignment
            raise RuntimeError(
                f"No EEG samples in requested window [{start_ts:.6f}, {end_ts:.6f}]. "
                f"Buffer covers [{ts.min():.6f}, {ts.max():.6f}] (len={len(ts)})."
            )

        x = np.asarray(list(self._x), dtype=float)[idx]  # (n_times, n_channels)
        data = x.T  # (n_channels, n_times)
        return data, ts[idx]


def connect_eeg(stream_name: Optional[str] = None) -> Optional[EEGStreamContext]:
    """
    Connect EEG (replacing Explore py function).

    Inputs:
        stream_name: LSL stream name to resolve by. Falls back to the
                     EEG_LSL_STREAM_NAME env var, then resolves by type.
    Output: stream/context object used by the rest of the backend
    """
    global _STREAM

    from pylsl import StreamInlet, resolve_byprop  # type: ignore

    stream_name = stream_name or os.getenv("EEG_LSL_STREAM_NAME")
    stream_type = os.getenv("EEG_LSL_STREAM_TYPE", "EEG")
    timeout = float(os.getenv("EEG_LSL_RESOLVE_TIMEOUT", "5.0"))

    streams = []
    if stream_name:
        streams = resolve_byprop("name", stream_name, timeout=timeout)
    if not streams:
        streams = resolve_byprop("type", stream_type, timeout=timeout)
    if not streams:
        _STREAM = None
        return None

    inlet = StreamInlet(streams[0], max_buflen=60, max_chunklen=32)
    info = inlet.info()

    ch_count = int(info.channel_count())
    sfreq = float(info.nominal_srate()) or float(os.getenv("EEG_DEFAULT_SFREQ", "250"))

    # Try to parse channel labels from stream metadata
    channel_names: list[str] = []
    try:
        ch = info.desc().child("channels").child("channel")
        while ch.name() == "channel":
            label = ch.child_value("label") or f"CH{len(channel_names) + 1}"
            channel_names.append(DEFAULT_CHANNEL_MAPPING.get(label.upper(), label))
            ch = ch.next_sibling()
    except Exception:
        channel_names = []

    if not channel_names:
        channel_names = [
            DEFAULT_CHANNEL_MAPPING.get(f"CH{i+1}", f"CH{i+1}") for i in range(ch_count)
        ]

    context = EEGStreamContext(
        inlet=inlet,
        channel_names=channel_names,
        sfreq=sfreq,
        buffer_seconds=float(os.getenv("EEG_BUFFER_SECONDS", "8.0")),
    )

    warmup = float(os.getenv("EEG_CONNECT_WARMUP_SECONDS", "0.4"))
    if warmup > 0:
        time.sleep(warmup)

    _STREAM = context
    return context


def _get_stream() -> EEGStreamContext:
    """Internal accessor used by other modules."""
    if _STREAM is None:
        raise RuntimeError("EEG stream not set. Call connect_eeg() first.")
    return _STREAM
