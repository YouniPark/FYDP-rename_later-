from dataclasses import dataclass

@dataclass
class AppConfig:
    MIN_DWELL_SEC: float = 0.100
    COOLDOWN_SEC: float = 0.500

    PROTOTXT: str = "deploy.prototxt"
    WEIGHTS: str = "res10_300x300_ssd.caffemodel"
    CONF_THRESH: float = 0.3
    PROTOTXT_URL: str = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    WEIGHTS_URL: str = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

    PRINT_PERIOD: float = 0.100

    MARKER_NAME: str = "Markers"
    MARKER_SOURCE_ID: str = "face_aoi_markers"
    # Manual marker config
    ENABLE_KB_MARKERS: bool = True
    MANUAL_MARKER_EVENT: str = "MANUAL_MARKER"
    KEY_MARK: str = "m"   # press to emit a manual marker
    KEY_QUIT: str = "q"   # press to quit streaming loop

    G3_TS_NAME: str = "Glasses3_VideoTS"
    G3_GAZE_NAME: str = "Glasses3_Gaze"
