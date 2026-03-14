"""
Event filter: gate AR-triggered events based on minimum inter-event interval.
"""

from __future__ import annotations

from typing import Optional

_LAST_EVENT_TS: Optional[float] = None


def event_filter(event_lsl_timestamp: float, min_interval_s: float = 1.0) -> bool:
    """
    Check whether an incoming AR event should be accepted.

    Inputs
    ------
    event_lsl_timestamp : float
        LSL timestamp of the event.

    Outputs
    -------
    ok : bool
        True if accepted, False if rejected due to being too soon after the last event.

    Notes
    -----
    - Designed to be simple & stateful: it remembers the last accepted event timestamp.
    - `min_interval_s` is configurable; pick something that matches AR trigger rate.
    """
    global _LAST_EVENT_TS

    ts = float(event_lsl_timestamp)

    if _LAST_EVENT_TS is None:
        _LAST_EVENT_TS = ts
        return True

    if (ts - _LAST_EVENT_TS) >= float(min_interval_s):
        _LAST_EVENT_TS = ts
        return True

    return False
