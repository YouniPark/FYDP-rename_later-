from typing import Dict, Optional, Tuple
import numpy as np
from .config import AppConfig
from .detector import FaceDetector
from .tracker import FaceTracker
from .markers import MarkerStream
from .utils import inside

class AOIEventEngine:
    def __init__(self, cfg: AppConfig, detector: FaceDetector, tracker: FaceTracker, marker_out: MarkerStream):
        self.cfg = cfg
        self.detector = detector
        self.tracker = tracker
        self.marker_out = marker_out
        self.entered: Dict[int, bool] = {}
        self.entry_t_lsl: Dict[int, Optional[float]] = {}
        self.last_hit_t: Dict[int, float] = {}
        self._last_print = 0.0

    def step(self, gx_norm: float, gy_norm: float, frame, t_lsl: float):
        H, W = frame.shape[:2]
        px = int(np.clip(gx_norm, 0, 1) * (W - 1))
        py = int(np.clip(gy_norm, 0, 1) * (H - 1))

        boxes = self.detector.detect(frame)
        face_map = self.tracker.assign_ids(boxes)

        event_fired = False
        fired_bbox: Optional[Tuple[int,int,int,int]] = None

        for fid, box in face_map.items():
            is_in = inside(px, py, box)
            was_in = self.entered.get(fid, False)

            if is_in and not was_in:
                self.entered[fid] = True
                self.entry_t_lsl[fid] = t_lsl
            elif is_in and was_in:
                t0 = self.entry_t_lsl.get(fid, None)
                if t0 is not None and (t_lsl - t0) >= self.cfg.MIN_DWELL_SEC:
                    last = self.last_hit_t.get(fid, -1e9)
                    if (t_lsl - last) >= self.cfg.COOLDOWN_SEC:
                        payload = {
                            "event": "AOI_HIT",
                            "face_id": int(fid),
                            "bbox": [int(v) for v in box],
                            "gaze_norm": [float(gx_norm), float(gy_norm)],
                            "entry_lsl": float(t0),
                            "emit_lsl": float(t_lsl)
                        }
                        self.marker_out.push_json_at(payload, timestamp=t0)
                        self.last_hit_t[fid] = t_lsl
                        self.entry_t_lsl[fid] = None
                        event_fired = True
                        fired_bbox = box
            elif (not is_in) and was_in:
                self.entered[fid] = False
                self.entry_t_lsl[fid] = None

        return event_fired, fired_bbox
