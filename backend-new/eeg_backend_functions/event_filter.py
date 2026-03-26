"""
Event filter: gate AR-triggered events based on minimum inter-event interval.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_LAST_EVENT_TS: Optional[float] = None


def event_filter(event_lsl_timestamp: float, min_interval_s: float = 3.0) -> bool:
    """
    Check whether an incoming AR event should be accepted.

    Inputs
    ------
    event_lsl_timestamp : float
        LSL timestamp of the event.
    min_interval_s : float
        Minimum seconds that must elapse since the last accepted event.

    Outputs
    -------
    ok : bool
        True if accepted, False if rejected due to being too soon after the last event.
    """
    global _LAST_EVENT_TS

    ts = float(event_lsl_timestamp)

    if _LAST_EVENT_TS is None:
        _LAST_EVENT_TS = ts
        logger.info("Event filter: first event accepted (ts=%.6f)", ts)
        return True

    elapsed = ts - _LAST_EVENT_TS
    if elapsed >= float(min_interval_s):
        _LAST_EVENT_TS = ts
        logger.info(
            "Event filter: accepted (%.2fs since last, threshold=%.1fs)",
            elapsed, min_interval_s,
        )
        return True

    logger.info(
        "Event filter: minimum interval between events has not passed "
        "(%.2fs < %.1fs). Ignoring event.",
        elapsed, min_interval_s,
    )
    return False
