import numpy as np
import cv2
from typing import List, Tuple

class FaceDetector:
    def __init__(self, net, conf_thresh: float):
        self.net = net
        self.conf_thresh = conf_thresh

    def detect(self, frame) -> List[Tuple[int,int,int,int]]:
        H, W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        dets = self.net.forward()
        boxes = []
        for i in range(dets.shape[2]):
            if dets[0, 0, i, 2] >= self.conf_thresh:
                x1, y1, x2, y2 = dets[0, 0, i, 3:7] * np.array([W, H, W, H])
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(W-1, int(x2)), min(H-1, int(y2))
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
        return boxes
