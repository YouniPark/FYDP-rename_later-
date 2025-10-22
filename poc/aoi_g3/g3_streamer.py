import json, logging, asyncio, time
from typing import Optional
import numpy as np
import cv2
from pylsl import StreamInfo, StreamOutlet, local_clock
from g3pylib import connect_to_glasses
from threading import Event, Thread, Lock
from queue import Queue

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
        self.outlet_faces: Optional[StreamOutlet] = None
        self.outlet_pupil: Optional[StreamOutlet] = None
        self._last_print = 0.0
        # Video display control
        self._last_display_time = 0.0
        self._display_fps_limit = cfg.DISPLAY_FPS_LIMIT
        self.window_name = cfg.DISPLAY_WINDOW_NAME
        # Keyboard support
        self._kb_listener = None
        self._quit_flag = False
        self._stop_event = Event()
        # Display thread and queue
        self._display_queue = Queue(maxsize=2)  # Keep only latest frames
        self._display_thread = None
        self._display_lock = Lock()

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
        ch_gz.append_child("channel").append_child_value("label", "pixel_x")
        ch_gz.append_child("channel").append_child_value("label", "pixel_y")
        ch_gz.append_child("channel").append_child_value("label", "norm_x")
        ch_gz.append_child("channel").append_child_value("label", "norm_y")
        self.outlet_gaze = StreamOutlet(info_gaze)

        info_faces = StreamInfo(self.cfg.G3_FACEBOX_NAME, "FaceBoxes", 7, 0, "float32", "g3_faces")
        ch_fb = info_faces.desc().append_child("channels")
        for label in ["local_ts", "frame_ts", "face_id", "x1", "y1", "x2", "y2"]: # frame ts is the timestamp of the tobii frame 
            ch_fb.append_child("channel").append_child_value("label", label)
        self.outlet_faces = StreamOutlet(info_faces)

        # pupil diameter in millimeters
        info_pupil = StreamInfo(self.cfg.G3_PUPIL_NAME, "Pupil", 4, 0, "float32", "g3_pupil")
        ch_pd = info_pupil.desc().append_child("channels")
        for label in ["local_ts", "gaze_ts", "pupil_left_mm", "pupil_right_mm"]: # gaze_ts here is thetobii device timestamp
            ch_pd.append_child("channel").append_child_value("label", label)
        self.outlet_pupil = StreamOutlet(info_pupil)

        logging.info("LSL outlets created (G3 timestamps + gaze + face boxes + pupil).")

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

    def set_display_fps_limit(self, fps: float):
        """Method to externally set video display FPS."""
        self._display_fps_limit = max(1.0, min(fps, 120.0))  # Clamp between 1-120 FPS
        logging.info(f"Display FPS limit set to {self._display_fps_limit}")

    def configure_display_window(self, width: int = 800, height: int = 600, x: int = None, y: int = None):
        """Method to externally set display window size and position."""
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(self.window_name, width, height)
            if x is not None and y is not None:
                cv2.moveWindow(self.window_name, x, y)
            logging.info(f"Display window configured: {width}x{height}")
        except Exception as e:
            logging.warning(f"Could not configure display window: {e}")

    def _display_thread_worker(self):
        """Worker thread that handles OpenCV display operations."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.window_name, 1920, 1080)
        
        while not self._quit_flag:
            try:
                # Get the latest frame from the queue (non-blocking)
                if not self._display_queue.empty():
                    image = self._display_queue.get_nowait()
                    cv2.imshow(self.window_name, image)
                
                # Check for key presses and window events
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self._quit_flag = True
                    self._stop_event.set()
                    break
                
                # Check if window was closed manually
                try:
                    if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        self._quit_flag = True
                        self._stop_event.set()
                        break
                except cv2.error:
                    self._quit_flag = True
                    self._stop_event.set()
                    break
                    
            except Exception as e:
                if not self._quit_flag:
                    logging.error(f"Display thread error: {e}")
                break
        
        cv2.destroyWindow(self.window_name)

    async def stream(self, show_video: bool = True):
        if not self.g3:
            logging.error("Not connected to Tobii Glasses 3.")
            return

        # Start display thread if video is enabled
        if show_video and self._display_thread is None:
            self._display_thread = Thread(target=self._display_thread_worker, daemon=True)
            self._display_thread.start()
            logging.info("Display thread started")

        logging.info("Starting RTSP stream...")
        # stream_rtsp returns an async context manager (not awaitable); use 'async with'
        alpha = 0.3  # smoothing factor
        smoothed_uv = None
        async with self.g3.stream_rtsp(scene_camera=True, gaze=True) as streams:
            async with streams.scene_camera.decode() as dec_stream, streams.gaze.decode() as dec_gaze:
                try:
                    while True:
                        if self._quit_flag:
                            break
                        ts = local_clock()
                        frame, frame_timestamp = await dec_stream.get()
                        
                        # Check if frame is valid before processing
                        if frame is None:
                            logging.warning("Received None frame, skipping...")
                            continue

                        try:
                            image = frame.to_ndarray(format="bgr24")
                        except Exception as e:
                            logging.warning(f"Failed to decode frame: {e}")
                            continue

                        if image is None or image.size == 0:
                            logging.warning("Frame decoded but image is empty, skipping...")
                            continue
                        
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
                        
                        # text smoothing
                        if px is not None and py is not None:
                            uv_array = np.array([px, py], dtype=np.float32)
                            if smoothed_uv is None:
                                smoothed_uv = uv_array
                            else:
                                smoothed_uv = alpha * uv_array + (1 - alpha) * smoothed_uv
                        
                            px, py = int(smoothed_uv[0]), int(smoothed_uv[1])

                        if gaze_timestamp is not None:
                            self.outlet_gaze.push_sample([ts, float(gaze_timestamp), float(px), float(py), float(gx_norm), float(gy_norm)], ts)
                        
                        # push pupil diameters; data structure from here: https://www.yixinkeyan.com/uploadfile/202303/627a1fdb755d6a0.pdf
                        if self.outlet_pupil is not None:
                            left_pd = float(gaze.get("eyeleft", {}).get("pupildiameter", float("nan")))
                            right_pd = float(gaze.get("eyeright", {}).get("pupildiameter", float("nan")))
                            gz_ts = float(gaze_timestamp) if gaze_timestamp is not None else float("nan")
                            self.outlet_pupil.push_sample([float(ts), gz_ts, left_pd, right_pd], ts)

                        event_fired, fired_bbox = self.engine.step(gx_norm, gy_norm, image, ts)
                        
                        # push face bounding boxes to lsl (one per frame)
                        if self.outlet_faces is not None:
                            ft = float(frame_timestamp) if frame_timestamp is not None else float("nan") #ft is tobii's scene camera timestamp, is nan if empty
                            # DATA STRUCTURE FOR FaceTracker
                            # engine.tracker.face_map: Dict[face_id, (x1, y1, x2, y2)] 
                            for fid, box in self.engine.tracker.face_map.items():
                                x1, y1, x2, y2 = box
                                self.outlet_faces.push_sample(
                                    [float(ts), ft, float(fid), float(x1), float(y1), float(x2), float(y2)],
                                    ts
                                )

                        # Update AOI and G3 Video Stream display
                        if show_video:
                            # Limit display updates to frame rate
                            display_interval = 1.0 / self._display_fps_limit
                            if (ts - self._last_display_time) >= display_interval:
                                # Draw visualizations on a copy of the image
                                display_image = image.copy()
                                for box in self.engine.tracker.face_map.values():
                                    x1, y1, x2, y2 = box
                                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.circle(display_image, (px, py), 10, (255, 0, 0), 2)
                                if event_fired and fired_bbox:
                                    x1, y1, x2, y2 = fired_bbox
                                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                
                                # Push to display queue (drop old frames if full)
                                if self._display_queue.full():
                                    try:
                                        self._display_queue.get_nowait()
                                    except:
                                        pass
                                self._display_queue.put_nowait(display_image)
                                self._last_display_time = ts

                        # Print if event is fired or print period elapsed
                        should_print = event_fired or (ts - self._last_print) >= self.cfg.PRINT_PERIOD
                        
                        if should_print:
                            self._last_print = ts
                            
                            print(json.dumps({
                                "lsl_time": round(ts, 6),
                                "gaze_norm": [round(gx_norm, 4), round(gy_norm, 4)],
                                "gaze_pixel": [px, py],
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
        # Wait for display thread to finish
        if self._display_thread is not None:
            self._display_thread.join(timeout=2.0)
            self._display_thread = None
        cv2.destroyAllWindows()
