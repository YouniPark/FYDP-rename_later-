import os, urllib.request, cv2
from .config import AppConfig

class FaceModel:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.net = None

    def ensure_files(self):
        if not os.path.exists(self.cfg.PROTOTXT):
            urllib.request.urlretrieve(self.cfg.PROTOTXT_URL, self.cfg.PROTOTXT)
        if not os.path.exists(self.cfg.WEIGHTS):
            urllib.request.urlretrieve(self.cfg.WEIGHTS_URL, self.cfg.WEIGHTS)

    def load(self):
        self.ensure_files()
        self.net = cv2.dnn.readNetFromCaffe(self.cfg.PROTOTXT, self.cfg.WEIGHTS)
