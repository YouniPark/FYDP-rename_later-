import json, logging, asyncio
from typing import Optional
import numpy as np
import cv2
from pylsl import StreamInfo, StreamOutlet, local_clock
from g3pylib import connect_to_glasses

from .config import AppConfig
from .aoi_engine import AOIEventEngine

logging.basicConfig(level=logging.INFO)

class G3toLSL:
    def __init__(self, hostname: str, cfg: AppConfig, engine: AOIEventEngine):
        self.hostname = hostname
        self.cfg = cfg
        self.engine = engine
        self.g3 = None
        self.outlet_ts: Optional[StreamOutlet] = None
        self.outlet_gaze: Optional[StreamOutlet] = None
        self._last_print = 0.0

    async def connect(self):
        logging.info(f"Connecting to Tobii Glasses 3 at {self.hostname}...")
        self.g3 = await connect_to_glasses.with_hostname(self.hostname, using_zeroconf=True)
        logging.info("Connected.")
        self._setup_lsl_streams()

    def _setup_lsl_streams(self):
        info_ts = StreamInfo(self.cfg.G3_TS_NAME, "VideoTimestamps", 2, 0, "float32", "g3_ts")
        ch_ts = info_ts.desc().append_child("channels")
        ch_ts.append_child("channel").append_child_value("label", "local_ts")
        ch_ts.append_child("channel").append_child_value("label", "frame_ts")
        self.outlet_ts = StreamOutlet(info_ts)

        info_gaze = StreamInfo(self.cfg.G3_GAZE_NAME, "Gaze", 4, 0, "float32", "g3_gaze")
        ch_gz = info_gaze.desc().append_child("channels")
        ch_gz.append_child("channel").append_child_value("label", "local_ts")
        ch_gz.append_child("channel").append_child_value("label", "gaze_ts")
        ch_gz.append_child("channel").append_child_value("label", "x-coordinate")
        ch_gz.append_child("channel").append_child_value("label", "y-coordinate")
        self.outlet_gaze = StreamOutlet(info_gaze)

        logging.info("LSL outlets created (G3 timestamps + gaze).")

    async def stream(self, show_video: bool = True):
        if not self.g3:
            raise RuntimeError("Not connected to Tobii Glasses 3.")

        logging.info("Starting RTSP stream...")
        streams = await self.g3.stream_rtsp(scene_camera=True, gaze=True)

        async with streams.scene_camera.decode() as dec_stream, streams.gaze.decode() as dec_gaze:
            try:
                while True:
                    ts = local_clock()
                    frame, frame_timestamp = await dec_stream.get()
                    image = frame.to_ndarray(format="bgr24")
                    H, W = image.shape[:2]

                    gaze, gaze_timestamp = await dec_gaze.get()
                    if "gaze2d" in gaze:
                        gx_norm, gy_norm = float(gaze["gaze2d"][0]), float(gaze["gaze2d"][1])
                    else:
                        gx_norm, gy_norm = 0.5, 0.5

                    self.outlet_ts.push_sample([ts, float(frame_timestamp)], ts)
                    px = int(np.clip(gx_norm, 0, 1) * (W - 1))
                    py = int(np.clip(gy_norm, 0, 1) * (H - 1))
                    self.outlet_gaze.push_sample([ts, float(gaze_timestamp), float(px), float(py)], ts)

                    event_fired, fired_bbox = self.engine.step(gx_norm, gy_norm, image, ts)

                    if show_video:
                        for box in self.engine.tracker.face_map.values():
                            x1, y1, x2, y2 = box
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(image, (px, py), 4, (255, 0, 0), -1)
                        if event_fired and fired_bbox:
                            x1, y1, x2, y2 = fired_bbox
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.imshow("G3 Scene + AOI", image)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    if (ts - self._last_print) >= self.cfg.PRINT_PERIOD:
                        self._last_print = ts
                        print(json.dumps({
                            "lsl_time": round(ts, 6),
                            "gaze_norm": [round(gx_norm, 4), round(gy_norm, 4)],
                            "bbox": [int(v) for v in fired_bbox] if fired_bbox else None,
                            "event": bool(event_fired)
                        }))
            finally:
                self.close()

    def close(self):
        if self.g3:
            self.g3.close()
            logging.info("Closed connection to Tobii Glasses 3.")
        cv2.destroyAllWindows()
