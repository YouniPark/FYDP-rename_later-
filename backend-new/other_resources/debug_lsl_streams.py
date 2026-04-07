"""
Run this script to list all LSL streams currently visible on the network.
Usage (from backend-new/):
    python debug_lsl_streams.py
"""
from pylsl import resolve_streams

print("Scanning for LSL streams (5 second wait)...\n")
streams = resolve_streams(wait_time=5.0)

if not streams:
    print("No streams found.")
else:
    print(f"Found {len(streams)} stream(s):\n")
    for i, s in enumerate(streams):
        print(f"  [{i}]  name        : {repr(s.name())}")
        print(f"       type        : {repr(s.type())}")
        print(f"       source_id   : {repr(s.source_id())}")
        print(f"       channels    : {s.channel_count()}")
        print(f"       srate       : {s.nominal_srate()}")
        print(f"       format      : {s.channel_format()}")
        print(f"       hostname    : {repr(s.hostname())}")
        print()
