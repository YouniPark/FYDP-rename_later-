"""
dummy_eeg.py
============
Dummy EEG event pipeline for test mode.

Replaces ``eeg_pipeline.run_eeg_event_pipeline()`` with a synthetic result so
the full backend pipeline can be exercised without a real EEG device or LSL
stream.

Configuration (environment variables or .env):
  DUMMY_EEG_ALWAYS_UNFAMILIAR=true    Always report unfamiliar (default).
                               false  Always report familiar.
                               random Coin-flip 50/50 on every event.
  DUMMY_EEG_DELAY=0.2                 Simulated processing delay in seconds.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random

logger = logging.getLogger(__name__)

_MODE = os.environ.get("DUMMY_EEG_ALWAYS_UNFAMILIAR", "true").strip().lower()
_DELAY = float(os.environ.get("DUMMY_EEG_DELAY", "0.2"))


def _sample_is_unfamiliar() -> bool:
    if _MODE == "random":
        return random.random() < 0.5
    return _MODE != "false"


async def run_dummy_eeg_event_pipeline(state, event_id: str, event_lsl_timestamp: float) -> dict:
    """Dummy replacement for ``run_eeg_event_pipeline()``.

    Returns a synthetic classification result after an optional simulated delay.
    """
    logger.info("Dummy EEG: processing event=%s (mode=%s, delay=%.2fs)", event_id, _MODE, _DELAY)

    if _DELAY > 0:
        await asyncio.sleep(_DELAY)

    is_unfamiliar = _sample_is_unfamiliar()

    logger.info(
        "Dummy EEG: event=%s → is_unfamiliar=%s",
        event_id,
        is_unfamiliar,
    )

    result = {
        "event_id": event_id,
        "event_lsl_timestamp": event_lsl_timestamp,
        "status": "ok",
        "is_unfamiliar": is_unfamiliar,
        "reason": f"dummy_eeg (mode={_MODE})",
    }
    state.latest_eeg_result[event_id] = result
    return result
