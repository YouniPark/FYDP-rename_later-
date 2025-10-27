#!/usr/bin/env python3
"""
offline aoi engine for scene video from POC

NEW ADDITIONS (oct 24)
- labelled box with gaze/face bounding box data in HUD and above each box
- saves mp4 video with info displayed
- saves all frame info as csv to help with comparison
- pixel radius (circle) around gaze point (60 px found to be the best considering tobii gaze offset)
- lsl timestamp = tobii timestamp + 42063.7852
- per-face filtering (may need to change in future so it's in the AOI engine code rather than just dropping frames) 
    --> drops true events for one face if there are multiple true events within X seconds (meant to target gaze jittering)

CODE FUNCTION

1. load scenevideo.mp4 and jsonl gaze file from tobii
2. ensure gaze coordinates in same space as video (normalized, uses pixels)
3. runs opencv face detection per video frame and assigns face ids with iou tracking
4. checks aoi entry events (when gaze enters face box and dwells). also has cooldown 
5. displays video with face box and gaze dot
6. saves csv of event timetsamps with video and lsl time

"""

import os, sys, json, csv, time, argparse, urllib.request, gzip, io
from threading import Thread, Event
from queue import Queue
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
from tqdm import tqdm


# config
PROTOTXT = "deploy.prototxt"
WEIGHTS  = "res10_300x300_ssd.caffemodel"
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
PROTOTXT_URL_ALT = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
WEIGHTS_URL  = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


# utility fcts

def ensure_face_model():
    # download face detection if missing
    if not os.path.exists(PROTOTXT):
        try:
            print(f"downloading {PROTOTXT} ...")
            try:
                urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT)
            except Exception:
                urllib.request.urlretrieve(PROTOTXT_URL_ALT, PROTOTXT)
        except Exception as e:
            raise RuntimeError(f"failed to download prototxt: {e}")
    if not os.path.exists(WEIGHTS):
        try:
            print(f"downloading {WEIGHTS} ...")
            urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS)
        except Exception as e:
            raise RuntimeError(f"failed to download weights: {e}")

def iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-6
    return inter / ua

def assign_ids(new_boxes: List[Tuple[int,int,int,int]],
               old_map: Dict[int, Tuple[int,int,int,int]],
               thresh: float = 0.3,
               next_id_start: int = 1) -> Dict[int, Tuple[int,int,int,int]]:
    used = set()
    assigned: Dict[int, Tuple[int,int,int,int]] = {}
    next_id = max([*old_map.keys(), next_id_start-1]) + 1 if old_map else next_id_start
    for nb in new_boxes:
        best, best_id = 0.0, None
        for fid, ob in old_map.items():
            if fid in used:
                continue
            score = iou(nb, ob)
            if score > best:
                best, best_id = score, fid
        if best >= thresh and best_id is not None:
            assigned[best_id] = nb
            used.add(best_id)
        else:
            assigned[next_id] = nb
            next_id += 1
    return assigned

def detect_faces(frame: np.ndarray, net: cv2.dnn_Net, conf: float = 0.6) -> List[Tuple[int,int,int,int]]:
    H, W = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    dets = net.forward()
    boxes: List[Tuple[int,int,int,int]] = []
    for i in range(dets.shape[2]):
        if dets[0, 0, i, 2] >= conf:
            x1, y1, x2, y2 = dets[0, 0, i, 3:7] * np.array([W, H, W, H])
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(W - 1, int(x2)), min(H - 1, int(y2))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes

def circle_intersects_box(cx: int, cy: int, r: int, box: Tuple[int,int,int,int]) -> bool:
    # returns true if a circle w/ centre cx,cy and radius r intersects rectangle/face bounding box x1,y1,x2,y2.
    if r <= 0:
        x1, y1, x2, y2 = box
        return (x1 <= cx <= x2) and (y1 <= cy <= y2)
    x1, y1, x2, y2 = box
    qx = min(max(cx, x1), x2)
    qy = min(max(cy, y1), y2)
    dx = cx - qx
    dy = cy - qy
    return (dx*dx + dy*dy) <= (r*r)

def draw_label(img, text, x, y, fg=(0,0,0), bg=(255,255,255)):
    # draw label onto video display
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + 2, y)
    x = max(0, min(x, img.shape[1] - tw - 6))
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + baseline), bg, -1)
    cv2.putText(img, text, (x + 2, y - 2), font, scale, fg, thickness, cv2.LINE_AA)

# load gaze data
@dataclass
class GazeSample:
    t: float # t in second
    x: float
    y: float

class GazeStream:
    def __init__(self, samples: List[GazeSample]):
        self.t = np.array([s.t for s in samples], dtype=np.float64)
        self.x = np.array([s.x for s in samples], dtype=np.float64)
        self.y = np.array([s.y for s in samples], dtype=np.float64)
        if self.t.size == 0:
            raise ValueError("no valid gaze samples found.")

    def at(self, t_query: float) -> Optional[Tuple[float,float]]:
        # interpolation of x,y at t_query
        if t_query < self.t[0] or t_query > self.t[-1]:
            return None
        idx = np.searchsorted(self.t, t_query)
        if idx == 0:
            return float(self.x[0]), float(self.y[0])
        if idx >= self.t.size:
            return float(self.x[-1])
        t0, t1 = self.t[idx-1], self.t[idx]
        u = 0.0 if t1 == t0 else (t_query - t0) / (t1 - t0)
        x = (1.0 - u) * self.x[idx-1] + u * self.x[idx]
        y = (1.0 - u) * self.y[idx-1] + u * self.y[idx]
        return float(x), float(y)

def _open_text_maybe_gz(path: str):
    # open .gz file
    if str(path).lower().endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8")

def load_gaze_jsonl(path: str,
                    use_key: str = "gaze2d",
                    min_ts: Optional[float] = None,
                    max_ts: Optional[float] = None,
                    clamp01: bool = True,
                    drop_outside: bool = True) -> GazeStream:
    ## parse json file
    samples: List[GazeSample] = []
    with _open_text_maybe_gz(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("type") != "gaze":
                    continue
                ts = float(obj.get("timestamp"))
                if min_ts is not None and ts < min_ts:
                    continue
                if max_ts is not None and ts > max_ts:
                    continue
                data = obj.get("data", {})
                g = data.get(use_key)
                if not g or len(g) != 2:
                    continue
                x, y = float(g[0]), float(g[1])
                if clamp01:
                    x = float(np.clip(x, 0.0, 1.0))
                    y = float(np.clip(y, 0.0, 1.0))
                if drop_outside and not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    continue
                samples.append(GazeSample(t=ts, x=x, y=y))
            except Exception:
                continue
    samples.sort(key=lambda s: s.t)
    return GazeStream(samples)

# main

def main():
    ap = argparse.ArgumentParser(description="offline aoi gaze fixation on faces detector")
    ap.add_argument("--video", required=True, help="path to scenevideo.mp4")
    ap.add_argument("--gaze",  required=True, help="path to gazedata.gz")
    ap.add_argument("--out",   required=True, help="output events csv path (event markers)")

    ap.add_argument("--frames-csv", required=True, help="output frames csv path")
    ap.add_argument("--video-out",  required=True, help="output labeled MP4 path")

    ap.add_argument("--min-dwell", type=float, default=0.10, help="seconds to dwell inside a face bounding box before event marked as true")
    ap.add_argument("--cooldown",  type=float, default=0.00, help="seconds btw repeated events per face")
    ap.add_argument("--conf",      type=float, default=0.60, help="face detector confidence threshold (can lower?)")
    ap.add_argument("--event-prune-window", type=float, default=0.0, help="fix gaze jittering.") # maybe future replace this by testing different cooldowns

    ap.add_argument("--gaze-offset", type=float, default=0.0, help="add to gaze timestamps before matching with video time (seconds)")
    ap.add_argument("--lsl-offset",  type=float, default=0.0, help="unused now; keeping for compatibility")
    ap.add_argument("--flip-y", action="store_true", help="interpret gaze y as bottom-left origin (flip to opencv formatting)")

    ap.add_argument("--display", action="store_true", help="show preview window with boxes + gaze dot")
    ap.add_argument("--display-every", type=int, default=1, help="render only every Nth frame in the preview window (>=1)")
    ap.add_argument("--gaze-radius-px", type=int, default=None, help="gaze circle radius in pixels (0 or none = no radius)")
    ap.add_argument("--gaze-offset-x-px", type=float, default=0.0, help="add this many pixels to gaze X (positive = right)")
    ap.add_argument("--gaze-offset-y-px", type=float, default=0.0, help="add this many pixels to gaze Y (positive = down)")
    ap.add_argument("--hud-face-lines", type=int, default=6, help="max number of face box lines to list in display")

    ap.add_argument("--display-threaded", action="store_true", default=(os.name == 'nt'), help="Use a separate thread for display so dragging/minimizing the window won't block processing (recommended on Windows)")
    ap.add_argument("--no-display-threaded", dest="display_threaded", action="store_false", help="Disable threaded display and render inline")
    
    ap.add_argument("--smooth-window", type=int, default=0, help="number of previous gaze points to use for smoothing (0 = no smoothing)")
    ap.add_argument("--smooth-alpha", type=float, default=0.5, help="decay factor for weighted average (0.0-1.0, higher = more weight on recent points)")


    ap.add_argument("--no-progress", action="store_true", help="disable progress bar")
    ap.add_argument("--progress-update-interval", type=int, default=50, help="update progress bar every N frames (default: 50)")

    # DNN backend/target
    ap.add_argument("--use-gpu", action="store_true", default=True, help="enable GPU acceleration when available (CUDA/OpenCL)")
    ap.add_argument("--dnn-backend", choices=["cpu", "cuda", "opencl"], default="cuda", help="preferred DNN backend when using GPU")
    ap.add_argument("--dnn-fp16", action="store_true", default=False, help="use FP16 compute target (CUDA/OpenCL) if supported")

    args = ap.parse_args()

    ensure_face_model()
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, WEIGHTS)

    # configure DNN backend/target
    def _configure_dnn_backend(net_obj, backend_choice: str, use_gpu: bool, fp16: bool) -> str:
        mode = "cpu"
        try:
            backend_choice = (backend_choice or "cpu").lower()
            if use_gpu and backend_choice == "cuda":
                # Quick capability probes
                cuda_mod = getattr(cv2, "cuda", None)
                cuda_count = 0
                try:
                    if cuda_mod is not None:
                        cuda_count = int(cuda_mod.getCudaEnabledDeviceCount())
                except Exception:
                    cuda_count = 0
                if cuda_count <= 0:
                    raise RuntimeError("OpenCV CUDA module not available or no CUDA device detected")
                net_obj.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                try:
                    target = cv2.dnn.DNN_TARGET_CUDA_FP16 if fp16 else cv2.dnn.DNN_TARGET_CUDA
                except Exception:
                    target = cv2.dnn.DNN_TARGET_CUDA
                net_obj.setPreferableTarget(target)
                mode = f"cuda({'fp16' if fp16 else 'fp32'})"
            elif use_gpu and backend_choice == "opencl":
                try:
                    cv2.ocl.setUseOpenCL(True)
                except Exception:
                    pass
                have_ocl = False
                try:
                    have_ocl = bool(cv2.ocl.haveOpenCL())
                except Exception:
                    have_ocl = False
                if not have_ocl:
                    raise RuntimeError("OpenCL not available")
                net_obj.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                try:
                    target = cv2.dnn.DNN_TARGET_OPENCL_FP16 if fp16 else cv2.dnn.DNN_TARGET_OPENCL
                except Exception:
                    target = cv2.dnn.DNN_TARGET_OPENCL
                net_obj.setPreferableTarget(target)
                mode = f"opencl({'fp16' if fp16 else 'fp32'})"
            else:
                net_obj.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net_obj.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                mode = "cpu"
        except Exception as e:
            # Hard fallback to CPU
            try:
                net_obj.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net_obj.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                mode = "cpu"
            except Exception:
                pass
            print(f"[DNN] Falling back to CPU: {e}")
        return mode

    selected_mode = _configure_dnn_backend(net, args.dnn_backend, args.use_gpu, args.dnn_fp16)
    print(f"OpenCV DNN configured: {selected_mode}")

    # open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cant open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_time = 1.0 / fps

    # set gaze radius
    if args.gaze_radius_px is not None and args.gaze_radius_px > 0:
        gaze_r_px = int(args.gaze_radius_px)
    else:
        gaze_r_px = 0  # default is just the point

    # prepare video writer w mp4v
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.video_out, fourcc, fps, (W, H))
    if not vw.isOpened():
        raise RuntimeError(f"cant open video writer: {args.video_out}")

    gzs = load_gaze_jsonl(args.gaze)

    # EVENTS TRUE ONLY CSV
    evt_fields = [
        "lsl_time","video_time","frame_idx",
        "gaze_x","gaze_y","px","py","gaze_radius_px",
        "face_id","x1","y1","x2","y2","event"
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    f_evt = open(args.out, "w", newline="", encoding="utf-8")
    evt_writer = csv.DictWriter(f_evt, fieldnames=evt_fields)
    evt_writer.writeheader()

    # FRAMES CSV
    frm_fields = [
        "lsl_time","video_time","frame_idx",
        "gaze_x","gaze_y","px","py","gaze_radius_px",
        "face_id","x1","y1","x2","y2","event_any_frame"
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.frames_csv)) or ".", exist_ok=True)
    f_frm = open(args.frames_csv, "w", newline="", encoding="utf-8")
    frm_writer = csv.DictWriter(f_frm, fieldnames=frm_fields)
    frm_writer.writeheader()

    face_map: Dict[int, Tuple[int,int,int,int]] = {}
    entered: Dict[int, bool] = {}
    entry_t: Dict[int, float] = {}  # video_time when gaze entered
    entry_lsl_t: Dict[int, float] = {}  # lsl_time when gaze entered
    entry_frame_idx: Dict[int, int] = {}  # frame_idx when gaze entered
    entry_gaze: Dict[int, Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]] = {}  # (gx, gy, px, py) when entered
    last_hit_t: Dict[int, float] = {}
    last_kept_event_time_by_face: Dict[int, float] = {}
    # Smoothing history buffer
    smooth_window = max(0, int(args.smooth_window))
    smooth_alpha = float(np.clip(args.smooth_alpha, 0.0, 1.0))
    gaze_history: List[Tuple[float, float]] = []  # stores (px, py) tuples


    frame_idx = 0

    # Display setup (window + optional thread)
    window_name = "Offline AOI"
    display_every = max(1, int(args.display_every))
    display_queue: Queue = Queue(maxsize=2)
    display_stop = Event()
    display_thread: Optional[Thread] = None

    def _configure_window():
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            # Use video dimensions for initial window size
            cv2.resizeWindow(window_name, W, H)
        except Exception:
            pass

    def _display_worker():
        _configure_window()
        while not display_stop.is_set():
            try:
                if not display_queue.empty():
                    img = display_queue.get_nowait()
                    cv2.imshow(window_name, img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    display_stop.set()
                    break
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        display_stop.set()
                        break
                except cv2.error:
                    display_stop.set(); break
            except Exception:
                # Don't crash the processing loop on display errors
                continue
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass

    if args.display:
        if args.display_threaded:
            display_thread = Thread(target=_display_worker, daemon=True)
            display_thread.start()
        else:
            _configure_window()

    # Setup progress bar
    pbar = None
    if not args.no_progress:
        pbar = tqdm(total=total_frames if total_frames > 0 else None, 
                   desc="Processing", unit="frame", dynamic_ncols=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            video_time = frame_idx * frame_time

            t_query_gaze = video_time - float(args.gaze_offset)
            gxy = gzs.at(t_query_gaze)

            tobii_ts = t_query_gaze
            lsl_time = tobii_ts + 42063.7852

            if gxy is None:
                gx = gy = None
                px = py = None
                px_raw = py_raw = None
            else:
                gx, gy = gxy
                if args.flip_y:
                    gy = 1.0 - gy
                gx = float(np.clip(gx, 0.0, 1.0))
                gy = float(np.clip(gy, 0.0, 1.0))
                px = int(round(gx * (W - 1)))
                py = int(round(gy * (H - 1)))
                
                # apply pixel offsets
                if args.gaze_offset_x_px:
                    try:
                        px += int(round(args.gaze_offset_x_px))
                    except Exception:
                        px += int(args.gaze_offset_x_px)
                if args.gaze_offset_y_px:
                    try:
                        py += int(round(args.gaze_offset_y_px))
                    except Exception:
                        py += int(args.gaze_offset_y_px)
                
                # clamp to video bounds
                px = max(0, min(W - 1, px))
                py = max(0, min(H - 1, py))
                # keep raw before smoothing
                px_raw, py_raw = px, py
                
                # Add to history and compute smoothed gaze
                gaze_history.append((float(px_raw), float(py_raw)))
                if len(gaze_history) > smooth_window:
                    gaze_history.pop(0)
                
                # Apply decaying weighted average if smoothing is enabled
                if smooth_window > 0 and len(gaze_history) > 0:
                    weights = []
                    for i in range(len(gaze_history)):
                        # More recent points get higher weight
                        # weight = alpha^(n-1-i) where i=0 is oldest, i=n-1 is newest
                        weight = smooth_alpha ** (len(gaze_history) - 1 - i)
                        weights.append(weight)
                    
                    # Normalize weights
                    weight_sum = sum(weights)
                    if weight_sum > 0:
                        weights = [w / weight_sum for w in weights]
                    
                    # Compute weighted average
                    px_sm = sum(w * h[0] for w, h in zip(weights, gaze_history))
                    py_sm = sum(w * h[1] for w, h in zip(weights, gaze_history))
                    px = int(round(px_sm))
                    py = int(round(py_sm))
                    # clamp smoothed coords
                    px = max(0, min(W - 1, px))
                    py = max(0, min(H - 1, py))

            # Face detection with runtime fallback if CUDA/OpenCL misconfigured
            try:
                boxes = detect_faces(frame, net, conf=args.conf)
            except cv2.error as e:
                msg = str(e)
                if ("DNN_BACKEND_CUDA" in msg) or ("CUDA" in msg and "backend" in msg):
                    # Reconfigure to CPU and retry once
                    _configure_dnn_backend(net, "cpu", False, False)
                    print("[DNN] CUDA error encountered; switched to CPU and retrying once…")
                    boxes = detect_faces(frame, net, conf=args.conf)
                else:
                    raise
            face_map = assign_ids(boxes, face_map)

            # aoi engine (uses gaze circle)
            event_fired = False
            fired_info = None
            if px is not None and py is not None:
                for fid, box in face_map.items():
                    hit = circle_intersects_box(px, py, gaze_r_px, box)
                    was_in = entered.get(fid, False)

                    if hit and not was_in: # Gaze enters box
                        entered[fid] = True
                        entry_t[fid] = video_time
                        entry_lsl_t[fid] = lsl_time
                        entry_frame_idx[fid] = frame_idx
                        entry_gaze[fid] = (gx, gy, px, py)

                    elif hit and was_in: # Gaze still inside box
                        t0 = entry_t.get(fid)
                        if t0 is not None and (video_time - t0) >= float(args.min_dwell):
                            if (video_time - last_hit_t.get(fid, -1e9)) >= float(args.cooldown):
                                event_fired = True
                                # Pass entry metadata along with current box
                                fired_info = (fid, box, entry_lsl_t.get(fid), entry_t.get(fid), 
                                            entry_frame_idx.get(fid), entry_gaze.get(fid))
                                last_hit_t[fid] = video_time
                                entry_t[fid] = None
                                entry_lsl_t[fid] = None
                                entry_frame_idx[fid] = None
                                entry_gaze[fid] = None

                    elif (not hit) and was_in: # Gaze exits box before dwell complete
                        entered[fid] = False
                        entry_t[fid] = None
                        entry_lsl_t[fid] = None
                        entry_frame_idx[fid] = None
                        entry_gaze[fid] = None

            # face and gaze display
            hud_lines = []
            hud_lines.append(f"video time: {video_time:0.3f}s  (lsl time: {lsl_time:0.3f}s)")
            if gx is None or gy is None:
                hud_lines.append("gaze: None")
            else:
                offset_str = ""
                if args.gaze_offset_x_px != 0 or args.gaze_offset_y_px != 0:
                    offset_str = f" off=({int(round(args.gaze_offset_x_px))},{int(round(args.gaze_offset_y_px))})"
                hud_lines.append(
                    f"gaze: norm=({gx:0.3f},{gy:0.3f}) px=({px},{py}) r={gaze_r_px}px{offset_str}"
                )
                if smooth_window > 0 and px_raw is not None and py_raw is not None:
                    hud_lines.append(f"unsmoothed: px=({px_raw},{py_raw}) window={smooth_window} alpha={smooth_alpha:0.2f}")
            hud_lines.append(f"event_any_frame: {event_fired}")

            MAX_FACE_LINES = max(0, int(args.hud_face_lines))
            if face_map and MAX_FACE_LINES > 0:
                for i, (fid, (x1, y1, x2, y2)) in enumerate(sorted(face_map.items())):
                    if i >= MAX_FACE_LINES:
                        hud_lines.append(f"+ {len(face_map)-MAX_FACE_LINES} more faces…")
                        break
                    hud_lines.append(f"face {fid}: [{x1},{y1},{x2},{y2}]")
            elif not face_map:
                hud_lines.append("face: none")

            # draw faces/labels
            for fid, (x1, y1, x2, y2) in face_map.items():
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID {fid} [{x1},{y1},{x2},{y2}]"
                draw_label(frame, label, x1, y1 - 6, fg=(0,0,0), bg=(200,255,200))

            # draw gaze and circle
            if px is not None and py is not None:
                # Draw unsmoothed gaze point (red) if smoothing is enabled and different from smoothed
                if smooth_window > 0 and px_raw is not None and py_raw is not None:
                    if px_raw != px or py_raw != py:
                        cv2.circle(frame, (px_raw, py_raw), max(2, min(6, gaze_r_px//3)), (0, 0, 255), -1) # unsmoothed gaze point (red)
                        # Draw line connecting unsmoothed to smoothed
                        cv2.line(frame, (px_raw, py_raw), (px, py), (0, 0, 255), 1)
                
                # Draw gaze point (blue)
                cv2.circle(frame, (px, py), max(3, min(8, gaze_r_px//2)), (255, 0, 0), -1) # center (smoothed)
                if gaze_r_px > 0:
                    cv2.circle(frame, (px, py), gaze_r_px, (255, 0, 0), 2) # radius (smoothed)

            y0 = 20
            for ln in hud_lines:
                cv2.putText(frame, ln, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(frame, ln, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,0,0), 1, cv2.LINE_AA)
                y0 += 24

            if args.display:
                # Render only every Nth frame
                if (frame_idx % display_every) == 0:
                    if args.display_threaded:
                        # Push latest frame for display; drop oldest if queue is full
                        if display_queue.full():
                            try:
                                _ = display_queue.get_nowait()
                            except Exception:
                                pass
                        try:
                            display_queue.put_nowait(frame)
                        except Exception:
                            pass
                    else:
                        cv2.imshow(window_name, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:
                            break
                        # Detect window close
                        try:
                            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                                break
                        except cv2.error:
                            break
                # If threaded, also check if display requested stop
                if args.display_threaded and display_stop.is_set():
                    break

            # Update progress bar
            update_interval = max(1, int(args.progress_update_interval))
            if pbar is not None and (frame_idx % update_interval) == 0:
                pbar.update(update_interval)

            vw.write(frame) # save

            # write events csv file (only events true)
            def write_event_row(fid: Optional[int], box: Optional[Tuple[int,int,int,int]], is_event: bool, 
                              entry_lsl: Optional[float] = None, entry_video: Optional[float] = None,
                              entry_fidx: Optional[int] = None, entry_g: Optional[Tuple] = None):
                x1=y1=x2=y2=None
                if box is not None:
                    x1, y1, x2, y2 = map(int, box)
                # Use entry timestamps if provided, otherwise fall back to current frame
                event_lsl_time = entry_lsl if entry_lsl is not None else lsl_time
                event_video_time = entry_video if entry_video is not None else video_time
                event_frame_idx = entry_fidx if entry_fidx is not None else frame_idx
                # Use entry gaze coordinates if provided
                if entry_g is not None:
                    e_gx, e_gy, e_px, e_py = entry_g
                else:
                    e_gx, e_gy, e_px, e_py = gx, gy, px, py
                row = dict(
                    lsl_time = round(event_lsl_time, 6),
                    video_time = round(event_video_time, 6),
                    frame_idx = event_frame_idx,
                    gaze_x = (None if e_gx is None else round(e_gx, 4)),
                    gaze_y = (None if e_gy is None else round(e_gy, 4)),
                    px = e_px, py = e_py, gaze_radius_px = gaze_r_px,
                    face_id = fid,
                    x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                    event = bool(is_event)
                )
                evt_writer.writerow(row)

            if event_fired and fired_info is not None:
                fid, box, e_lsl, e_video, e_fidx, e_gaze = fired_info
                if fid is not None:
                    # Use entry LSL time for prune window check
                    current_event_time = float(e_lsl if e_lsl is not None else lsl_time)
                    last_kept = last_kept_event_time_by_face.get(fid)
                    if (last_kept is None) or ((current_event_time - last_kept) >= float(args.event_prune_window)):
                        write_event_row(fid, box, True, e_lsl, e_video, e_fidx, e_gaze)
                        last_kept_event_time_by_face[fid] = current_event_time
                    else:
                        pass
                else:
                    write_event_row(fid, box, True, e_lsl, e_video, e_fidx, e_gaze)

            # write all frames
            base_frm = dict(
                lsl_time = round(lsl_time, 6),
                video_time = round(video_time, 6),
                frame_idx = frame_idx,
                gaze_x = (None if gx is None else round(gx, 4)),
                gaze_y = (None if gy is None else round(gy, 4)),
                px = px, py = py, gaze_radius_px = gaze_r_px,
                event_any_frame = bool(event_fired),
            )
            if face_map:
                for fid, (x1, y1, x2, y2) in face_map.items():
                    row = dict(base_frm)
                    row.update(dict(face_id=fid, x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)))
                    frm_writer.writerow(row)
            else:
                row = dict(base_frm)
                row.update(dict(face_id=None, x1=None, y1=None, x2=None, y2=None))
                frm_writer.writerow(row)

            frame_idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        if pbar is not None:
            # Update any remaining frames not caught by the modulo
            update_interval = max(1, int(args.progress_update_interval))
            remainder = frame_idx % update_interval
            if remainder > 0:
                pbar.update(remainder)
            pbar.close()
        try:
            f_evt.flush(); f_evt.close()
        except Exception:
            pass
        try:
            f_frm.flush(); f_frm.close()
        except Exception:
            pass
        cap.release()
        try:
            vw.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

        print(f"\nevents csv : {args.out}")
        print(f"frames csv : {args.frames_csv}")
        print(f"labeled scenevideo.mp4: {args.video_out}")
        print("done.")

if __name__ == "__main__":
    main()
