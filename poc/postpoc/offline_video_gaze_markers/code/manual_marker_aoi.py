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
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2


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

    ap.add_argument("--gaze-offset", type=float, default=0.0, help="add to gaze timestamps before matching with video time")
    ap.add_argument("--lsl-offset",  type=float, default=0.0, help="unused now; keeping for compatibility")
    ap.add_argument("--flip-y", action="store_true", help="interpret gaze y as bottom-left origin (flip to opencv formatting)")

    ap.add_argument("--display", action="store_true", help="show preview window with boxes + gaze dot")
    ap.add_argument("--gaze-radius-px", type=int, default=None, help="gaze circle radius in pixels (0 or none = no radius)")
    ap.add_argument("--hud-face-lines", type=int, default=6, help="max number of face box lines to list in display")

    args = ap.parse_args()

    ensure_face_model()
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, WEIGHTS)

    # open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cant open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
    entry_t: Dict[int, float] = {}
    last_hit_t: Dict[int, float] = {}
    last_kept_event_time_by_face: Dict[int, float] = {}

    frame_idx = 0

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
            else:
                gx, gy = gxy
                if args.flip_y:
                    gy = 1.0 - gy
                gx = float(np.clip(gx, 0.0, 1.0))
                gy = float(np.clip(gy, 0.0, 1.0))
                px = int(round(gx * (W - 1)))
                py = int(round(gy * (H - 1)))

            boxes = detect_faces(frame, net, conf=args.conf)
            face_map = assign_ids(boxes, face_map)

            # aoi engine (uses gaze circle)
            event_fired = False
            fired_info = None
            if px is not None and py is not None:
                for fid, box in face_map.items():
                    hit = circle_intersects_box(px, py, gaze_r_px, box)
                    was_in = entered.get(fid, False)

                    if hit and not was_in:
                        entered[fid] = True
                        entry_t[fid] = video_time

                    elif hit and was_in:
                        t0 = entry_t.get(fid)
                        if t0 is not None and (video_time - t0) >= float(args.min_dwell):
                            if (video_time - last_hit_t.get(fid, -1e9)) >= float(args.cooldown):
                                event_fired = True
                                fired_info = (fid, box)
                                last_hit_t[fid] = video_time
                                entry_t[fid] = None

                    elif (not hit) and was_in:
                        entered[fid] = False
                        entry_t[fid] = None

            # face and gaze display
            hud_lines = []
            hud_lines.append(f"video time: {video_time:0.3f}s  (lsl time: {lsl_time:0.3f}s)")
            if gx is None or gy is None:
                hud_lines.append("gaze: None")
            else:
                hud_lines.append(f"gaze: norm=({gx:0.3f},{gy:0.3f}) px=({px},{py}) r={gaze_r_px}px")
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
                cv2.circle(frame, (px, py), max(3, min(8, gaze_r_px//2)), (255, 0, 0), -1) # center
                if gaze_r_px > 0:
                    cv2.circle(frame, (px, py), gaze_r_px, (255, 0, 0), 2) # radius

            y0 = 20
            for ln in hud_lines:
                cv2.putText(frame, ln, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(frame, ln, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,0,0), 1, cv2.LINE_AA)
                y0 += 24

            if args.display:
                cv2.imshow("Offline AOI (labeled)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            vw.write(frame) # save

            # write events csv file (only events true)
            def write_event_row(fid: Optional[int], box: Optional[Tuple[int,int,int,int]], is_event: bool):
                x1=y1=x2=y2=None
                if box is not None:
                    x1, y1, x2, y2 = map(int, box)
                row = dict(
                    lsl_time = round(lsl_time, 6),
                    video_time = round(video_time, 6),
                    frame_idx = frame_idx,
                    gaze_x = (None if gx is None else round(gx, 4)),
                    gaze_y = (None if gy is None else round(gy, 4)),
                    px = px, py = py, gaze_radius_px = gaze_r_px,
                    face_id = fid,
                    x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                    event = bool(is_event)
                )
                evt_writer.writerow(row)

            if event_fired and fired_info is not None:
                fid, box = fired_info
                if fid is not None:
                    current_event_time = float(lsl_time)
                    last_kept = last_kept_event_time_by_face.get(fid)
                    if (last_kept is None) or ((current_event_time - last_kept) >= float(args.event_prune_window)):
                        write_event_row(fid, box, True)
                        last_kept_event_time_by_face[fid] = current_event_time
                    else:
                        pass
                else:
                    write_event_row(fid, box, True)

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
