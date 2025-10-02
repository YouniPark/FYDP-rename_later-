import asyncio
import logging
import cv2

from pylsl import StreamInfo, StreamOutlet, local_clock
from g3pylib import connect_to_glasses # only works with python 3.10


logging.basicConfig(level=logging.INFO)


class G3toLSL:
    def __init__(self, hostname: str):
        self.hostname = hostname
        self.g3 = None
        self.outlet_ts = None
        self.outlet_gaze = None

    async def connect(self):
        """Connect to Tobii Glasses 3."""
        logging.info(f"Connecting to Tobii Glasses 3 at {self.hostname}...")
        self.g3 = await connect_to_glasses.with_hostname(self.hostname, using_zeroconf=True)
        logging.info("Connected.")

        self._setup_lsl_streams()

    def _setup_lsl_streams(self):
        """Initialize LSL."""

        # Video timestamps
        info_ts = StreamInfo("Glasses3_VideoTS", "VideoTimestamps", 2, 0, "float32", "g3_ts")
        channels_ts = info_ts.desc().append_child("channels")
        channels_ts.append_child("channel").append_child_value("label", "local_ts")
        channels_ts.append_child("channel").append_child_value("label", "frame_ts")
        self.outlet_ts = StreamOutlet(info_ts)

        # Gaze
        info_gaze = StreamInfo("Glasses3_Gaze", "Gaze", 4, 0, "float32", "g3_gaze")
        channels_gaze = info_gaze.desc().append_child("channels")
        channels_gaze.append_child("channel").append_child_value("label", "local_ts")
        channels_gaze.append_child("channel").append_child_value("label", "gaze_ts")
        channels_gaze.append_child("channel").append_child_value("label", "x-coordinate")
        channels_gaze.append_child("channel").append_child_value("label", "y-coordinate")
        self.outlet_gaze = StreamOutlet(info_gaze)

        logging.info("LSL outlets created.")

    async def stream(self):
        """Push data to LSL."""
        if not self.g3:
            raise RuntimeError("Not connected to Tobii Glasses 3.")

        logging.info("Starting RTSP stream...")
        streams = await self.g3.stream_rtsp(scene_camera=True, gaze=True)
        async with streams.scene_camera.decode() as decoded_stream, streams.gaze.decode() as decoded_gaze:
            try:
                while True:
                    ts = local_clock()

                    # Get video frame
                    frame, frame_timestamp = await decoded_stream.get()
                    image = frame.to_ndarray(format="bgr24")

                    # Get gaze data
                    gaze, gaze_timestamp = await decoded_gaze.get()
                    x, y = -1, -1  # default if gaze not available
                    if "gaze2d" in gaze:
                        gaze2d = gaze["gaze2d"]  # rational (x,y)
                        h, w = image.shape[:2]   # convert to pixel location
                        x, y = int(gaze2d[0] * w), int(gaze2d[1] * h)

                    # Push to LSL
                    self.outlet_ts.push_sample([ts, frame_timestamp], ts)
                    self.outlet_gaze.push_sample([ts, gaze_timestamp, x, y], ts)

                    # Show video
                    cv2.imshow("Video", image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            finally:
                self.close()

    def close(self):
        """Clean up resources."""
        if self.g3:
            self.g3.close()
            logging.info("Closed connection to Tobii Glasses 3.")
        cv2.destroyAllWindows()


def main():
    hostname = ""  # serial number or IP address
    streamer = G3toLSL(hostname)

    asyncio.run(streamer.connect())
    asyncio.run(streamer.stream())


if __name__ == "__main__":
    main()