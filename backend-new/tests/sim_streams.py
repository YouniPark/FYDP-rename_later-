#!/usr/bin/env python3
"""
tests/sim_streams.py  —  Local LSL test dummy

Run: python tests/sim_streams.py tests/exp7.xdf --name Explore_84A1_ExG --proxy PersonTest
==============================================

Spawns two LSL outlets so you can exercise the full backend pipeline
without physical hardware:

  1. EEG Replay outlet  (name: Explore_84A1_ExG, type: EEG)
       Reads a previously recorded .xdf file, finds the EEG stream by name,
       and re-streams its samples in real-time via a new LSL outlet.  Sample
       timestamps are remapped onto the local clock so they align with any
       fixation events fired from this same script.

       Playback loops back to the beginning when the end of the file is
       reached.

  2. FixationEvents outlet  (name: FixationEvents, type: Markers)
       Pushes a string marker with a configurable proxy name each time you
       press the trigger key.  This is exactly what the Unity AR app sends.

Keyboard controls (non-blocking, Windows msvcrt)
-------------------------------------------------
  [S]           Start (or restart) EEG replay from the beginning
  [P]           Pause / resume EEG replay
  [SPACE] / [F] Push a FixationEvent marker
  [H]           Print this help
  [Q] / [ESC]   Quit

Usage
-----
  cd backend-new
  python tests/sim_streams.py path/to/recording.xdf
  python tests/sim_streams.py path/to/recording.xdf --name Explore_84A1_ExG --proxy PersonA

Requirements (extra, not in project venv by default)
-----------------------------------------------------
  pip install pyxdf
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Optional

try:
    import pyxdf  # type: ignore
except ImportError:
    sys.exit(
        "[sim_streams] ERROR: pyxdf is not installed.\n"
        "  Run:  pip install pyxdf\n"
    )

try:
    from pylsl import StreamInfo, StreamOutlet, local_clock  # type: ignore
except ImportError:
    sys.exit(
        "[sim_streams] ERROR: pylsl is not installed.\n"
        "  Run:  pip install pylsl\n"
    )

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PUSH_CHUNK_SAMPLES = 10   # samples pushed per outlet call
SFREQ_FALLBACK     = 250  # Hz — used only if the XDF header reports 0
FIXATION_STREAM_NAME = "FixationEvents"


# ===========================================================================
#  EEGPlayer — reads XDF and re-broadcasts via LSL in real-time
# ===========================================================================

class EEGPlayer:
    """
    Loads an XDF recording, extracts the named EEG stream, and replays it
    via a new LSL StreamOutlet at the correct sample rate.

    XDF timestamps are remapped to the current local_clock() so that
    EEG sample timestamps and FixationEvent timestamps share the same
    clock domain (a requirement for the backend's epoch extraction).
    """

    def __init__(self, xdf_path: Path, stream_name: str) -> None:
        self._xdf_path    = xdf_path
        self._stream_name = stream_name

        self._data:    Optional[np.ndarray] = None   # (n_channels, n_samples)
        self._xdf_ts:  Optional[np.ndarray] = None   # (n_samples,) — original XDF timestamps
        self._ch_names: list[str]           = []
        self._sfreq:    float               = SFREQ_FALLBACK
        self._outlet:   Optional[StreamOutlet] = None

        self._thread: Optional[threading.Thread] = None
        self._stop_event  = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()   # set = playing

        self._state      = "STOPPED"   # "STOPPED" | "RUNNING" | "PAUSED"
        self._state_lock = threading.Lock()

        self._load_xdf()
        self._create_outlet()

    # -----------------------------------------------------------------------
    # Loading / outlet creation
    # -----------------------------------------------------------------------

    def _load_xdf(self) -> None:
        print(f"[EEG] Loading XDF: {self._xdf_path}  …", flush=True)
        streams, _ = pyxdf.load_xdf(str(self._xdf_path))

        target = None
        for s in streams:
            name_list = s["info"].get("name", [])
            if name_list and name_list[0] == self._stream_name:
                target = s
                break

        if target is None:
            found = [s["info"]["name"][0] for s in streams if s["info"].get("name")]
            sys.exit(
                f"[EEG] Stream '{self._stream_name}' not found in XDF.\n"
                f"       Available streams: {found}\n"
            )

        raw_series = target["time_series"]
        self._data   = np.asarray(raw_series, dtype=float).T   # (n_ch, n_samp)
        self._xdf_ts = np.asarray(target["time_stamps"], dtype=float)

        nominal = float(target["info"]["nominal_srate"][0])
        self._sfreq = nominal if nominal > 0 else SFREQ_FALLBACK

        # Parse channel labels from XDF metadata if available
        try:
            ch_entries = target["info"]["desc"][0]["channels"][0]["channel"]
            self._ch_names = [c["label"][0] for c in ch_entries]
        except (KeyError, IndexError, TypeError):
            self._ch_names = [f"CH{i + 1}" for i in range(self._data.shape[0])]

        n_ch   = self._data.shape[0]
        n_samp = self._data.shape[1]
        dur_s  = float(self._xdf_ts[-1] - self._xdf_ts[0]) if n_samp > 1 else 0.0
        print(
            f"[EEG] Loaded  '{self._stream_name}':  "
            f"{n_ch} ch × {n_samp} samples @ {self._sfreq:.0f} Hz  "
            f"({dur_s:.1f} s)",
            flush=True,
        )
        print(f"[EEG] Channels: {self._ch_names}", flush=True)

    def _create_outlet(self) -> None:
        n_ch = self._data.shape[0]
        info = StreamInfo(
            name=self._stream_name,
            type="EEG",
            channel_count=n_ch,
            nominal_srate=self._sfreq,
            channel_format="float32",
            source_id=f"sim_{self._stream_name}",
        )

        # Embed channel metadata so the backend resolves labels correctly
        channels = info.desc().append_child("channels")
        for label in self._ch_names:
            ch = channels.append_child("channel")
            ch.append_child_value("label", label)
            ch.append_child_value("type", "EEG")
            ch.append_child_value("unit", "microvolts")

        self._outlet = StreamOutlet(info, chunk_size=PUSH_CHUNK_SAMPLES)
        print(f"[EEG] LSL outlet '{self._stream_name}' created.", flush=True)

    # -----------------------------------------------------------------------
    # Playback control (thread-safe)
    # -----------------------------------------------------------------------

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    def start(self) -> None:
        """Start or restart playback from the beginning of the XDF file."""
        # Signal any running thread to stop, then wait for clean exit
        self._stop_event.set()
        self._pause_event.set()   # unblock if paused so the thread can exit
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        self._stop_event.clear()
        self._pause_event.set()

        with self._state_lock:
            self._state = "RUNNING"

        self._thread = threading.Thread(
            target=self._play_loop, daemon=True, name="EEGPlayer"
        )
        self._thread.start()
        print("[EEG] Playback started.", flush=True)

    def toggle_pause(self) -> None:
        st = self.state
        if st == "RUNNING":
            self._pause_event.clear()
            with self._state_lock:
                self._state = "PAUSED"
            print("[EEG] Paused.", flush=True)
        elif st == "PAUSED":
            self._pause_event.set()
            with self._state_lock:
                self._state = "RUNNING"
            print("[EEG] Resumed.", flush=True)
        else:
            print("[EEG] Not running — press [S] to start first.", flush=True)

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        with self._state_lock:
            self._state = "STOPPED"

    # -----------------------------------------------------------------------
    # Playback loop (runs in background thread)
    # -----------------------------------------------------------------------

    def _play_loop(self) -> None:
        assert self._outlet is not None
        assert self._data is not None
        assert self._xdf_ts is not None

        n_samp   = self._data.shape[1]
        xdf_ts   = self._xdf_ts
        chunk_sz = PUSH_CHUNK_SAMPLES
        period_s = float(chunk_sz) / self._sfreq  # target sleep between chunk pushes

        # Map XDF timestamps → current local_clock domain
        #   local_ts[i] = t0_lsl + (xdf_ts[i] - t0_xdf)
        t0_lsl = local_clock()
        t0_xdf = xdf_ts[0]
        idx    = 0

        while not self._stop_event.is_set():
            # ── Pause handling ──
            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            # ── Loop back at end of recording ──
            if idx >= n_samp:
                print("[EEG] End of XDF reached — looping back to start.", flush=True)
                t0_lsl = local_clock()
                t0_xdf = xdf_ts[0]
                idx    = 0

            end_idx = min(idx + chunk_sz, n_samp)

            # Build per-sample LSL timestamps for this chunk
            mapped_ts = (t0_lsl + (xdf_ts[idx:end_idx] - t0_xdf)).tolist()

            # Transpose to (chunk_size, n_channels) as expected by push_chunk
            samples = self._data[:, idx:end_idx].T.tolist()

            # Push with per-sample timestamps so the backend's pull_window
            # sees correctly spaced LSL timestamps
            self._outlet.push_chunk(samples, mapped_ts)

            idx = end_idx

            # ── Pace ourselves to real-time ──
            # The target time for the LAST sample we just pushed:
            target_wall = t0_lsl + (xdf_ts[end_idx - 1] - t0_xdf)
            now         = local_clock()
            sleep_s     = target_wall + period_s - now
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        with self._state_lock:
            self._state = "STOPPED"
        print("[EEG] Playback thread exited.", flush=True)


# ===========================================================================
#  FixationEmitter — pushes string markers to FixationEvents outlet
# ===========================================================================

class FixationEmitter:
    """
    Creates a persistent LSL StreamOutlet for FixationEvents string markers
    and pushes one marker per call to fire().
    """

    def __init__(self, proxy_name: str = "TestProxy") -> None:
        self._proxy_name = proxy_name
        info = StreamInfo(
            name=FIXATION_STREAM_NAME,
            type="Markers",
            channel_count=1,
            nominal_srate=0,          # irregular (event-driven)
            channel_format="string",
            source_id="sim_fixation",
        )
        self._outlet = StreamOutlet(info)
        print(
            f"[FIX] LSL outlet '{FIXATION_STREAM_NAME}' created  "
            f"(proxy='{proxy_name}').",
            flush=True,
        )

    def fire(self) -> None:
        ts = local_clock()
        self._outlet.push_sample([self._proxy_name], timestamp=ts)
        print(
            f"[FIX] FixationEvent pushed  proxy='{self._proxy_name}'  "
            f"lsl_ts={ts:.6f}",
            flush=True,
        )


# ===========================================================================
#  Non-blocking keyboard (Windows msvcrt)
# ===========================================================================

def _read_key() -> Optional[str]:
    """
    Non-blocking single-key read.  Returns the pressed character (lowercase)
    or None if no key is waiting.

    Uses msvcrt on Windows.  On other OS the loop falls back to blocking
    threading input — press Enter after each command on non-Windows.
    """
    try:
        import msvcrt
        if msvcrt.kbhit():
            raw = msvcrt.getch()
            # Discard the trailing byte for extended / function keys
            if raw in (b"\xe0", b"\x00"):
                msvcrt.getch()
                return None
            return raw.decode("utf-8", errors="ignore").lower()
        return None
    except ImportError:
        return None


# ===========================================================================
#  CLI
# ===========================================================================

HELP_TEXT = """\
────────────────────────────────────────────────────────
  EEG Replay + FixationEvents LSL Test Simulator
────────────────────────────────────────────────────────
  [S]           Start (or restart) EEG replay
  [P]           Pause / resume EEG replay
  [SPACE] / [F] Push a FixationEvent marker
  [H]           Show this help
  [Q] / [ESC]   Quit
────────────────────────────────────────────────────────
  Tip: start EEG replay first and wait at least 1 second
  before firing a fixation event so the backend has
  enough pre-event EEG data for epoch extraction.
────────────────────────────────────────────────────────
"""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay an XDF EEG recording and simulate FixationEvents on the LSL network."
    )
    p.add_argument(
        "xdf_path",
        type=Path,
        help="Path to the .xdf recording file.",
    )
    p.add_argument(
        "--name",
        default="Explore_84A1_ExG",
        metavar="STREAM_NAME",
        help=(
            "Name of the EEG stream inside the XDF file and the name that "
            "will be advertised on LSL.  Default: Explore_84A1_ExG"
        ),
    )
    p.add_argument(
        "--proxy",
        default="TestProxy",
        metavar="PROXY_NAME",
        help=(
            "Proxy / face name string pushed as the FixationEvent marker "
            "payload.  This is matched by the backend's face-proxy logic. "
            "Default: TestProxy"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.xdf_path.exists():
        sys.exit(f"[sim_streams] XDF file not found: {args.xdf_path}")

    print(HELP_TEXT, flush=True)

    player  = EEGPlayer(xdf_path=args.xdf_path, stream_name=args.name)
    emitter = FixationEmitter(proxy_name=args.proxy)

    print("\n[sim_streams] Ready — waiting for keypresses…\n", flush=True)

    try:
        while True:
            key = _read_key()

            if key is None:
                time.sleep(0.05)
                continue

            if key == "s":
                player.start()

            elif key == "p":
                player.toggle_pause()

            elif key in (" ", "f"):
                emitter.fire()

            elif key == "h":
                print(HELP_TEXT, flush=True)

            elif key in ("q", "\x1b"):
                print("[sim_streams] Quitting…", flush=True)
                break

    except KeyboardInterrupt:
        print("\n[sim_streams] Interrupted.", flush=True)
    finally:
        player.stop()
        print("[sim_streams] Done.", flush=True)


if __name__ == "__main__":
    main()
