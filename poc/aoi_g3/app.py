# To run: python -m aoi_g3.app
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
    '''
    When the recording unit operates in WLAN (wireless) host mode, the IPv4 address is: 192.168.75.51
    Tobii Pro Glasses 3 can also be accessed by entering [serial_number].local (for example, TG03X-0123456789.local)
    
    If the device is connected directly to the machine (peer-to-peer network), 
    the IPv4 address on the Ethernet interface is 0.0.0.0 and cannot be used.
    The recording unit will assign itself a link-local fe80::.. IPv6 address that is reachable by all locally-connected devices with no need for configuration.
    The IPv6 address is a local address that will always remain the same. 
    The broadcast is only sent on IPv6.
    
    Example: HOSTNAME = "192.168.75.51" or "TG03B-080203017551"

    '''
    HOSTNAME = "TG03B-080203017551"
    asyncio.run(run(HOSTNAME, show_video=True))
