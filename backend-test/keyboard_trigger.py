"""
keyboard_trigger.py
===================
Keyboard-based fixation event trigger for test mode.

Replaces the LSL FixationEvents inlet (``event_inlet_pipeline``) with two
mechanisms:

1. **Keyboard**: press Enter in the terminal to fire a fixation event.
2. **Auto-trigger**: fire automatically every N seconds when
   ``TEST_AUTO_TRIGGER_INTERVAL`` > 0.

The ``/test/trigger`` HTTP endpoint in ``main.py`` provides a third option
(curl or browser) that does not depend on this module.

Configuration (environment variables or .env):
  TEST_AUTO_TRIGGER_INTERVAL=0    Seconds between auto-triggers (0 = off).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

_AUTO_INTERVAL = float(os.environ.get("TEST_AUTO_TRIGGER_INTERVAL", "0"))


async def keyboard_event_loop(
    state,
    on_event: Callable[[str, float, str], Awaitable[None]],
) -> None:
    """Background task that fires fixation events on keyboard input and/or a timer.

    Parameters
    ----------
    state:
        ``AppState`` — passed through to ``on_event`` (not used directly here).
    on_event:
        Async callable matching the signature used by ``event_inlet_pipeline``:
        ``on_event(event_id: str, lsl_timestamp: float, proxy_name: str)``.
    """
    _print_banner()

    async def _fire(source: str) -> None:
        ts = time.time()
        event_id = f"event_{source}_{ts:.6f}"
        logger.info("Fixation event triggered — source=%s  event_id=%s", source, event_id)
        asyncio.create_task(on_event(event_id, ts, source))

    tasks: list[asyncio.Task] = []

    async def _keyboard_loop() -> None:
        """Read lines from stdin; each line (Enter key) fires one event."""
        loop = asyncio.get_event_loop()
        while True:
            try:
                # run_in_executor keeps the event loop free while waiting for input.
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    # EOF — stdin closed (e.g. redirected from /dev/null).
                    logger.info("Keyboard trigger: stdin EOF — keyboard trigger disabled")
                    return
                await _fire("keyboard")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Keyboard trigger: error reading stdin: %s", exc)
                await asyncio.sleep(1.0)

    async def _auto_loop() -> None:
        """Auto-fire every ``_AUTO_INTERVAL`` seconds."""
        logger.info("Auto-trigger: firing every %.1f s", _AUTO_INTERVAL)
        while True:
            await asyncio.sleep(_AUTO_INTERVAL)
            await _fire("auto")

    tasks.append(asyncio.create_task(_keyboard_loop()))
    if _AUTO_INTERVAL > 0:
        tasks.append(asyncio.create_task(_auto_loop()))

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _print_banner() -> None:
    sep = "=" * 64
    lines = [
        "",
        sep,
        "[TEST MODE]  Fixation event trigger ready.",
        "",
        "  • Press ENTER in this terminal to fire one fixation event.",
        "  • POST http://localhost:8000/test/trigger  (curl / browser).",
    ]
    if _AUTO_INTERVAL > 0:
        lines.append(f"  • Auto-trigger: every {_AUTO_INTERVAL:.1f} s  (TEST_AUTO_TRIGGER_INTERVAL).")
    lines += [sep, ""]
    print("\n".join(lines))
