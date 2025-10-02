import json
from pylsl import StreamInfo, StreamOutlet
from .config import AppConfig

class MarkerStream:
    def __init__(self, cfg: AppConfig):
        info = StreamInfo(name=cfg.MARKER_NAME, type="Markers", channel_count=1,
                          nominal_srate=0, channel_format="string",
                          source_id=cfg.MARKER_SOURCE_ID)
        self.outlet = StreamOutlet(info, chunk_size=0, max_buffered=3600)

    def push_json_at(self, payload: dict, timestamp: float):
        self.outlet.push_sample([json.dumps(payload)], timestamp=timestamp)
