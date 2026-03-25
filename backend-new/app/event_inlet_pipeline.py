"""
Event (Fixation) Inlet (LSL): reads FixationEvents markers pushed by the Unity AR app
and dispatches each one to the provided async callback for EEG processing.

The inlet reconnects automatically if the stream disappears (e.g. app restart).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from app.state import AppState

logger = logging.getLogger(__name__)


async def event_inlet_loop(
    state: AppState,
    on_event: Callable[[str, float, str], Awaitable[None]],
) -> None:
    """
    Long-running background task that:

    1. Resolves the FixationEvents LSL stream by name.
    2. Polls for incoming string markers with a non-blocking pull.
    3. For each received marker fires on_event(event_id, lsl_timestamp, face_proxy_name).

    Reconnects (with a configurable sleep) whenever the stream is lost or
    not yet visible on the network.

    Parameters
    ----------
    state : AppState
    on_event : async callable(event_id: str, lsl_timestamp: float, face_proxy_name: str)
        Called for every fixation marker received from LSL.  Each call is
        launched as a separate asyncio task so slow pipeline work does not
        block polling.
    """
    try:
        from pylsl import StreamInlet, resolve_byprop, LostError
    except ImportError:
        logger.error("pylsl is not installed; LSL fixation inlet is disabled")
        return

    stream_name = state.settings.lsl_fixation_stream_name
    resolve_timeout = state.settings.lsl_fixation_resolve_timeout
    poll_interval = state.settings.lsl_fixation_poll_interval
    retry_seconds = state.settings.lsl_fixation_retry_seconds

    while True:
        try:
            logger.info(
                "Event inlet: resolving stream '%s' (timeout=%.1fs)...",
                stream_name,
                resolve_timeout,
            )
            # resolve_byprop blocks for up to resolve_timeout seconds, so run it on a thread.
            streams = await asyncio.to_thread(
                resolve_byprop, "name", stream_name, 1, resolve_timeout
            )
            if not streams:
                logger.info(
                    "Event inlet: stream '%s' not found; retrying in %.1fs",
                    stream_name,
                    retry_seconds,
                )
                await asyncio.sleep(retry_seconds)
                continue

            inlet: StreamInlet = await asyncio.to_thread(
                StreamInlet, streams[0], max_buflen=30
            )
            logger.info("Event inlet: connected to stream '%s'", stream_name)

            while True:
                # pull_sample(timeout=0.0) returns immediately with ("", 0.0) when the
                # queue is empty, so it is safe to call directly without to_thread.
                sample, ts = inlet.pull_sample(timeout=0.0)
                if ts:
                    proxy_name: str = sample[0] if sample else ""
                    event_id = f"event_{proxy_name}_{ts:.6f}"
                    logger.info(
                        "Event inlet: fixation marker received (proxy=%s, ts=%.6f)",
                        proxy_name,
                        ts,
                    )
                    asyncio.create_task(on_event(event_id, ts, proxy_name))

                await asyncio.sleep(poll_interval)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Event inlet: unexpected error; reconnecting in %.1fs", retry_seconds
            )
            await asyncio.sleep(retry_seconds)
