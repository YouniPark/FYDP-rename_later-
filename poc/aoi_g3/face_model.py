import os, urllib.request, cv2, logging
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
        
        # Configure DNN backend/target
        try:
            backend = (self.cfg.DNN_BACKEND or "cpu").lower()
            use_gpu = bool(self.cfg.USE_GPU)
            fp16 = bool(self.cfg.DNN_FP16)

            if use_gpu and backend == "cuda":
                # CUDA requires OpenCV built with CUDA support
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                target = cv2.dnn.DNN_TARGET_CUDA_FP16 if fp16 else cv2.dnn.DNN_TARGET_CUDA
                self.net.setPreferableTarget(target)
                logging.info(f"OpenCV DNN using CUDA target: {'FP16' if fp16 else 'FP32'}")
            elif use_gpu and backend == "opencl":
                # OpenCL path is via OPENCV backend + OPENCL target
                try:
                    cv2.ocl.setUseOpenCL(True)
                except Exception:
                    pass
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                target = cv2.dnn.DNN_TARGET_OPENCL_FP16 if fp16 else cv2.dnn.DNN_TARGET_OPENCL
                self.net.setPreferableTarget(target)
                logging.info(f"OpenCV DNN using OpenCL target: {'FP16' if fp16 else 'FP32'} (ocl.have={getattr(cv2.ocl,'haveOpenCL',lambda:None)()})")
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                logging.info("OpenCV DNN using CPU target")
        except Exception as e:
            logging.warning(f"Failed to set preferred DNN backend/target, falling back to CPU: {e}")
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except Exception:
                pass
