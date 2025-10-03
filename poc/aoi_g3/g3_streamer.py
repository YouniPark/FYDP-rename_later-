import json, logging, asyncio
from typing import Optional
import numpy as np
import cv2
from pylsl import StreamInfo, StreamOutlet, local_clock
from g3pylib import connect_to_glasses
from threading import Event

try:
    from pynput import keyboard
    _HAS_PYNPUT = True
except Exception:
    _HAS_PYNPUT = False

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
        # Keyboard support
        self._kb_listener = None
        self._quit_flag = False
        self._stop_event = Event()

    async def connect(self):
        # Attempt to connect to the Glasses 3 service - retry on failure.

        # Start keyboard listener
        if self._kb_listener is None:
            self._start_keyboard()
        
        retry_delay = 5.0  # seconds between attempts
        attempt = 0
        
        while not self._quit_flag:
            attempt += 1
            try:
                logging.info(f"Connecting to Tobii Glasses 3 at {self.hostname} (attempt {attempt})...")
                self.g3 = await connect_to_glasses.with_hostname(self.hostname, using_zeroconf=True)
                logging.info("Connected.")
                self._setup_lsl_streams()
                return
            except Exception as e:
                # Quit requested during connection attempt
                if self._quit_flag:
                    break

                # Log a warning and retry instead of raising
                logging.warning(
                    f"Glasses 3 service not available (attempt {attempt}): {e}. "
                    f"Retrying in {retry_delay:.1f}s..."
                )
                try:
                    await asyncio.sleep(retry_delay)
                except asyncio.CancelledError:
                    logging.info("Connection attempt interrupted.")
                    break
        # If loop exits without a successful connection
        if not self.g3:
            logging.info("Stopped attempting to connect to Tobii Glasses 3.")

    def _setup_lsl_streams(self):
        info_ts = StreamInfo(self.cfg.G3_TS_NAME, "VideoTimestamps", 2, 0, "float32", "g3_ts")
        ch_ts = info_ts.desc().append_child("channels")
        ch_ts.append_child("channel").append_child_value("label", "local_ts")
        ch_ts.append_child("channel").append_child_value("label", "frame_ts")
        self.outlet_ts = StreamOutlet(info_ts)

        info_gaze = StreamInfo(self.cfg.G3_GAZE_NAME, "Gaze", 6, 0, "float32", "g3_gaze")
        ch_gz = info_gaze.desc().append_child("channels")
        ch_gz.append_child("channel").append_child_value("label", "local_ts")
        ch_gz.append_child("channel").append_child_value("label", "gaze_ts")
        ch_gz.append_child("channel").append_child_value("label", "x-coordinate")
        ch_gz.append_child("channel").append_child_value("label", "y-coordinate")
        self.outlet_gaze = StreamOutlet(info_gaze)

        logging.info("LSL outlets created (G3 timestamps + gaze).")

    def _start_keyboard(self):
        if not self.cfg.ENABLE_KB_MARKERS: # Keyboard markers disabled
            return
        if not _HAS_PYNPUT:
            logging.warning("Keyboard markers enabled but 'pynput' is not available. Install it to use keybinds.")
            return

        def on_press(key):
            try:
                ch = key.char
            except AttributeError:
                return

            if ch == self.cfg.KEY_MARK:
                ts = local_clock()
                payload = {
                    "event": self.cfg.MANUAL_MARKER_EVENT,
                    "emit_lsl": float(ts)
                }
                try:
                    self.engine.marker_out.push_json_at(payload, timestamp=ts)
                    print(f"Sent {self.cfg.MANUAL_MARKER_EVENT} at LSL time {ts:.5f}")
                except Exception as e:
                    logging.error(f"Failed to push manual marker: {e}")
            elif ch == self.cfg.KEY_QUIT:
                print("Quit requested via keyboard.")
                self._quit_flag = True
                self._stop_event.set()

        self._kb_listener = keyboard.Listener(on_press=on_press)
        self._kb_listener.start()
        print(f"Keyboard ready: '{self.cfg.KEY_MARK}' to send {self.cfg.MANUAL_MARKER_EVENT}, '{self.cfg.KEY_QUIT}' to quit.")

    async def stream(self, show_video: bool = True):
        if not self.g3:
            logging.error("Not connected to Tobii Glasses 3.")
            return

        logging.info("Starting RTSP stream...")
        # stream_rtsp returns an async context manager (not awaitable); use 'async with'
        async with self.g3.stream_rtsp(scene_camera=True, gaze=True) as streams:
            async with streams.scene_camera.decode() as dec_stream, streams.gaze.decode() as dec_gaze:
                try:
                    while True:
                        if self._quit_flag:
                            break
                        ts = local_clock()
                        frame, frame_timestamp = await dec_stream.get()
                        image = frame.to_ndarray(format="bgr24")
                        H, W = image.shape[:2]

                        gaze, gaze_timestamp = await dec_gaze.get()
                        if "gaze2d" in gaze:
                            gx_norm, gy_norm = float(gaze["gaze2d"][0]), float(gaze["gaze2d"][1])
                        else:
                            continue


                        if frame_timestamp is not None:
                            self.outlet_ts.push_sample([ts, float(frame_timestamp)], ts)
                        
                        px = int(np.clip(gx_norm, 0, 1) * (W - 1))
                        py = int(np.clip(gy_norm, 0, 1) * (H - 1))
                        
                        if gaze_timestamp is not None:
                            self.outlet_gaze.push_sample([ts, float(gaze_timestamp), float(px), float(py), float(gx_norm), float(gy_norm)], ts)

                        event_fired, fired_bbox = self.engine.step(gx_norm, gy_norm, image, ts)

                        if show_video:
                            for box in self.engine.tracker.face_map.values():
                                x1, y1, x2, y2 = box
                                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.circle(image, (px, py), 10, (255, 0, 0), 2)
                            if event_fired and fired_bbox:
                                x1, y1, x2, y2 = fired_bbox
                                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.imshow("G3 Scene + AOI", image)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                self._quit_flag = True
                                self._stop_event.set()
                                break

                        if (ts - self._last_print) >= self.cfg.PRINT_PERIOD:
                            # print('gaze:', gaze)
                            # print(f'gx_norm: {gx_norm}, gy_norm: {gy_norm}')
                            print(f'px: {px}, py: {py}' )

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
        # Stop keyboard listener if running
        try:
            if self._kb_listener is not None:
                self._kb_listener.stop()
                self._kb_listener = None
        except Exception:
            pass
        cv2.destroyAllWindows()
