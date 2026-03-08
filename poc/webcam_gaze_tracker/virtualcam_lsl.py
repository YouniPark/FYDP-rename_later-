## Webcam gaze tracker virtual camera with LSL timestamp overlay
# Modified from virtualcam.py in eyetrax
# Adds LSL timestamp and webcam to OBS virtual camera frames
# 
# To run: 
# python virtualcam_lsl.py
# (from within poc/webcam_gaze_tracker)
#
# Or without calibration: 
# python virtualcam_lsl.py --model-file Data/gaze_model.pkl
# (from within poc/webcam_gaze_tracker)
# 
# IF camera not working add --camera 1

import os

import cv2
import numpy as np
import pyvirtualcam

# LSL imports
try:
    from pylsl import StreamInfo, StreamOutlet, local_clock
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False
    StreamInfo = None
    StreamOutlet = None
    local_clock = None
    print("[LSL] pylsl not installed. LSL functionality will be disabled.")

from eyetrax.calibration import (
    run_5_point_calibration,
    run_9_point_calibration,
    run_lissajous_calibration,
)
from cli import parse_common_args
from eyetrax.filters import KalmanSmoother, KDESmoother, NoSmoother, make_kalman
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.draw import draw_cursor, make_thumbnail
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import camera, iter_frames


def run_virtualcam():
    args = parse_common_args()

    filter_method = args.filter
    camera_index = args.camera
    calibration_method = args.calibration
    confidence_level = args.confidence
    gaze_radius = args.radius

    gaze_estimator = GazeEstimator(model_name=args.model)

    if args.model_file and os.path.isfile(args.model_file):
        gaze_estimator.load_model(args.model_file)
        print(f"[virtualcam] Loaded gaze model from {args.model_file}")
    else:
        if calibration_method == "9p":
            run_9_point_calibration(gaze_estimator, camera_index=camera_index)
        elif calibration_method == "5p":
            run_5_point_calibration(gaze_estimator, camera_index=camera_index)
        else:
            run_lissajous_calibration(gaze_estimator, camera_index=camera_index)
        
        gaze_estimator.save_model("Data/gaze_model.pkl")
        print("[virtualcam] Gaze model saved to Data/gaze_model.pkl")

    screen_width, screen_height = get_screen_size()

    if filter_method == "kalman":
        kalman = make_kalman()
        smoother = KalmanSmoother(kalman)
        smoother.tune(gaze_estimator, camera_index=camera_index)
    elif filter_method == "kde":
        kalman = None
        smoother = KDESmoother(screen_width, screen_height, confidence=confidence_level)
    else:
        kalman = None
        smoother = NoSmoother()

    green_bg = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    green_bg[:] = (0, 255, 0)

    cam_width, cam_height = 320, 240
    BORDER = 2
    MARGIN = 20

    # Set up LSL stream for gaze coordinates
    lsl_outlet = None
    if LSL_AVAILABLE and StreamInfo and StreamOutlet:
        info = StreamInfo('GazeWebcam', 'GazePx', 2, 0, 'float32', 'webcam_gaze')
        lsl_outlet = StreamOutlet(info)
        print("[LSL] Webcam gaze coordinate stream created")
    
    if local_clock is None:
        print("[LSL] local_clock not available")

    with camera(camera_index) as cap:
        cam_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        print(f"[virtualcam] Detected camera FPS: {cam_fps}")
        with pyvirtualcam.Camera(
            width=screen_width,
            height=screen_height,
            fps=cam_fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
        ) as cam:
            print(f"Virtual camera started: {cam.device}")
            for frame in iter_frames(cap):
                features, blink_detected = gaze_estimator.extract_features(frame)

                if features is not None and not blink_detected:
                    gaze_point = gaze_estimator.predict(np.array([features]))[0]
                    x, y = map(int, gaze_point)
                    x_pred, y_pred = smoother.step(x, y)
                    contours = smoother.debug.get("contours", [])
                else:
                    x_pred = y_pred = None
                    contours = []

                output = green_bg.copy()
                if contours:
                    cv2.drawContours(output, contours, -1, (0, 0, 255), 3)
                if x_pred is not None and y_pred is not None:
                    # Push gaze coordinates to LSL if on screen
                    if (0 <= x_pred < screen_width and 0 <= y_pred < screen_height and 
                        lsl_outlet is not None):
                        lsl_outlet.push_sample([float(x_pred), float(y_pred)])
                    
                    draw_cursor(
                        output,
                        x_pred,
                        y_pred,
                        alpha=0.9,
                        radius_outer=gaze_radius,
                        radius_inner=0,
                        color_outer=(0, 0, 255),
                    )

                thumb = make_thumbnail(frame, size=(cam_width, cam_height), border=BORDER)
                h, w = thumb.shape[:2]
                output[-h - MARGIN : -MARGIN, -w - MARGIN : -MARGIN] = thumb

                # LSL time overlay
                lsl_time = local_clock() if local_clock is not None else None
                if lsl_time is not None:
                    lsl_text = f"LSL Time: {lsl_time:.6f}"
                    text_position = (20, 40)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    thickness = 2
                    border_size = 6
                    # Draw black border
                    cv2.putText(
                        output,
                        lsl_text,
                        text_position,
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness + border_size,
                        cv2.LINE_AA,
                    )
                    # Draw red text on top
                    cv2.putText(
                        output,
                        lsl_text,
                        text_position,
                        font,
                        font_scale,
                        (0, 0, 255),
                        thickness,
                        cv2.LINE_AA,
                    )

                cam.send(output)
                cam.sleep_until_next_frame()


if __name__ == "__main__":
    run_virtualcam()
