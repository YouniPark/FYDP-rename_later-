import asyncio
from .config import AppConfig
from .face_model import FaceModel
from .detector import FaceDetector
from .tracker import FaceTracker
from .markers import MarkerStream
from .aoi_engine import AOIEventEngine
from .g3_streamer import G3toLSL

def build_app(hostname: str, cfg: AppConfig):
    face_model = FaceModel(cfg); face_model.load()
    detector = FaceDetector(face_model.net, cfg.CONF_THRESH)
    tracker  = FaceTracker(iou_thresh=cfg.CONF_THRESH)
    marker   = MarkerStream(cfg)
    engine   = AOIEventEngine(cfg, detector, tracker, marker)
    return G3toLSL(hostname, cfg, engine)

async def run(hostname: str, show_video: bool = True):
    cfg = AppConfig()
    app = build_app(hostname, cfg)
    await app.connect()
    await app.stream(show_video=show_video)

if __name__ == "__main__":
    # Example: HOSTNAME = "192.168.71.50" or "TG03B-0123456789"
    HOSTNAME = ""
    asyncio.run(run(HOSTNAME, show_video=True))
