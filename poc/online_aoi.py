import time, json
import numpy as np
import cv2
from pylsl import resolve_stream, StreamInlet, StreamInfo, StreamOutlet


# extra config

GAZE_STREAM_NAME = "TobiiG3-Gaze"
SCENE_URL = "rtsp://192.168.71.50/live/scene"
MIN_DWELL_SEC = 0.100
COOLDOWN_SEC = 0.300

PROTOTXT = "deploy.prototxt"
WEIGHTS  = "res10_300x300_ssd.caffemodel"
CONF_THRESH = 0.3

PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
WEIGHTS_URL  = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


# DOWNLOAD THESE IF WE DONT HAVE
def ensure_face_model():
    if not os.path.exists(PROTOTXT):
        print(f"[setup] Downloading {PROTOTXT} ...")
        urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT)
    if not os.path.exists(WEIGHTS):
        print(f"[setup] Downloading {WEIGHTS} ...")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS)

# helper functions
def iou(a, b): 
    # taken from https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-6 #added 1e-6 to avoid division by zero (https://www.kaggle.com/code/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy)
    return inter / ua

def assign_ids(new_boxes, old_map, thresh=0.3, next_id_start=1): 
    # adapted from: https://github.com/GBJim/iou_tracker/blob/master/iou_tracker.py
    used = set()
    assigned = {}
    next_id = max([*old_map.keys(), next_id_start-1]) + 1 if old_map else next_id_start
    for nb in new_boxes:
        best, best_id = 0, None
        for fid, ob in old_map.items():
            if fid in used: 
                continue
            score = iou(nb, ob)
            if score > best:
                best, best_id = score, fid
        if best >= thresh:
            assigned[best_id] = nb
            used.add(best_id)
        else:
            assigned[next_id] = nb
            next_id += 1
    return assigned

def detect_faces(frame, net, conf=0.6): 
    # taken from: https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
    H, W = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)),
                                 1.0, (300,300), (104.0,177.0,123.0))
    net.setInput(blob)
    dets = net.forward()
    boxes = []
    for i in range(dets.shape[2]):
        if dets[0,0,i,2] >= conf:
            x1,y1,x2,y2 = dets[0,0,i,3:7] * np.array([W,H,W,H])
            x1,y1 = max(0,int(x1)), max(0,int(y1))
            x2,y2 = min(W-1,int(x2)), min(H-1,int(y2))
            if x2>x1 and y2>y1:
                boxes.append((x1,y1,x2,y2))
    return boxes

# setup for LSL (https://labstreaminglayer.readthedocs.io/dev/examples.html)
print(f"resolving the lsl stream: '{GAZE_STREAM_NAME}' ...")
streams = resolve_stream('name', GAZE_STREAM_NAME, timeout=10)
if not streams:
    raise RuntimeError(f"can't find stream called '{GAZE_STREAM_NAME}'.")
gaze_inlet = StreamInlet(streams[0], max_buflen=2)

marker_info = StreamInfo(name="Markers", type="Markers", channel_count=1,
                         nominal_srate=0, channel_format="string",
                         source_id="face_aoi_markers")
marker_out = StreamOutlet(marker_info, chunk_size=0, max_buffered=3600)

# setup for face (https://shrishailsgajbhar.github.io/post/Deep-Learning-Image-Classification-Opencv-DNN)
cap = cv2.VideoCapture(SCENE_URL)
if not cap.isOpened():
    raise RuntimeError(f"lol sucks we cant open scene video: {SCENE_URL}")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, WEIGHTS)

# to record the states
face_map = {}
entered = {}
entry_t_lsl = {}
last_hit_t = {}

##### CONFIGURATION VARIABLES #####

last_print = 0.0 # initialization
PRINT_PERIOD = 0.100
SAMPLING_TIME = 0.001

MIN_DWELL_SEC = 0.100 # 100ms
COOLDOWN_SEC = 0.300 # 300ms

def inside(px, py, box): # determines if the given coordinates are inside the box
    x1,y1,x2,y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2

try:
    while True:
        #sample, t_lsl = gaze_inlet.pull_sample(timeout=SAMPLING_TIME) # get the sample from LSL (wait up to SAMPLING_TIME; can be removed)

        sample, t_lsl = gaze_inlet.pull_sample()

        # if sample is None: # no sample; we try and grab one video frame from the stream
        #     ok, frame = cap.read()
        #     if not ok:
        #         time.sleep(SAMPLING_TIME)
        #         continue
            
        #     # get bounding boxes from opencv
        #     boxes = detect_faces(frame, net, CONF_THRESH)
        #     face_map = assign_ids(boxes, face_map) # get a stable ID for each face (in case of multiple faces); optional, can remove
        #     continue

        ## print("checkpoint 1") DEBUGGING

        gx, gy = sample # gaze coordinates
        ok, frame = cap.read()
        if not ok: # if getting the next video frame doesn't work, then we retry after a brief pause (sampling_time)
            time.sleep(SAMPLING_TIME)
            continue

        # grab the frame shape of the frame from cap.read()
        H, W = frame.shape[:2]
        # turns the normalized gaze to integer pixel column
        px = int(np.clip(gx,0,1) * (W-1))
        py = int(np.clip(gy,0,1) * (H-1))

        #detect faces on the video frame
        boxes = detect_faces(frame, net, CONF_THRESH)
        face_map = assign_ids(boxes, face_map) # face map (so two faces aren't read as the same)

        event_fired = False #set the flag as false when we start, this will be set to true when gaze enters face
        fired_bbox = None # fired_bbox is basically storing the bounding box that triggered the flag (for printing/logging)

        for fid, box in face_map.items(): #loop over each face (unique id) and its associated bounding box
            is_in = inside(px, py, box) # check if gaze is inside box (see above for inside function)
            was_in = entered.get(fid, False) #check if we were previously inside this same face

            # CASE 1: just entered face this frame
            if is_in and not was_in:
                entered[fid] = True # mark as inside AOI
                entry_t_lsl[fid] = t_lsl # record LSL time

            elif is_in and was_in: # CASE 2: still inside box from before
                t0 = entry_t_lsl.get(fid, None) # get the initial entry time into the AOI
                if t0 is not None and (t_lsl - t0) >= MIN_DWELL_SEC:

                    if (t_lsl - last_hit_t.get(fid, -1e9)) >= COOLDOWN_SEC: #(only fire if the cooldown time has passed)
                    # basically if the face has never been hit before, then t_lsl - (-1e9) is massive and guarantees being outside of the cooldown period
                    # note that t_lsl is the time of lsl

                        payload = {
                            "event": "AOI_HIT", # set event as true!
                            "face_id": int(fid),
                            "bbox": [int(v) for v in box],
                            "gaze_norm": [float(gx), float(gy)],
                            "entry_lsl": float(t0), # first time when face entered box
                            "emit_lsl": float(t_lsl) # lsl time for when fixation threshold (100ms) is satisfied
                        }

                        # push to LSL
                        marker_out.push_sample([json.dumps(payload)], timestamp=t0)
                        last_hit_t[fid] = t_lsl # last hit time for cooldown calculations

                        # ensures that the code doesn't fire every 1ms you stay in the face
                        # counter resets when eyes leave and come back to face
                        entry_t_lsl[fid] = None
                        event_fired = True
                        fired_bbox = box

            elif (not is_in) and was_in: # CASE 3: just exited face region
                entered[fid] = False # no longer in AOI
                entry_t_lsl[fid] = None # clear entry timestamp
        
        ## print("checkpoint 2 im so tired help")

        # prints if event-fired is true, otherwise print every PRINT_PERIOD (100ms suggested)
        if (t_lsl - last_print) >= PRINT_PERIOD:
            last_print = t_lsl
            bbox_print = list(map(int, fired_bbox)) if fired_bbox else None
            print(json.dumps({
                "lsl_time": round(t_lsl, 6),
                "gaze_norm": [round(float(gx),4), round(float(gy),4)],
                "bbox": bbox_print,
                "event": bool(event_fired)
            }))

except KeyboardInterrupt:
    pass
finally:
    cap.release()
