import asyncio
import logging
import cv2

from pylsl import StreamInfo, StreamOutlet, local_clock
from g3pylib import connect_to_glasses # only works with python 3.10

logging.basicConfig(level=logging.INFO)
G3_HOSTNAME = "" # serial number of ip address 

async def stream_rtsp():
    # connect to Tobii G3
    g3 = await connect_to_glasses.with_hostname(G3_HOSTNAME, using_zeroconf=True)

    # LSL
    # video timestamp 
    info_ts = StreamInfo("Glasses3_VideoTS", "VideoTimestamps", 2, 0, "float32", "g3_ts")
    outlet_ts = StreamOutlet(info_ts)

    channels_ts = info_ts.desc().append_child("channels")
    channels_ts.append_child("channel").append_child_value("label", "local_ts")
    channels_ts.append_child("channel").append_child_value("label", "frame_ts")

    # eye tracker  
    info_gaze = StreamInfo("Glasses3_Gaze", "Gaze", 4, 0, "float32", "g3_gaze")
    outlet_gaze = StreamOutlet(info_gaze)

    channels_gaze = info_gaze.desc().append_child("channels")
    channels_gaze.append_child("channel").append_child_value("label", "local_ts")
    channels_gaze.append_child("channel").append_child_value("label", "gaze_ts")
    channels_gaze.append_child("channel").append_child_value("label", "x-coordinate")
    channels_gaze.append_child("channel").append_child_value("label", "y-coordinate")

    try:
        # get RTSP stream
        streams = await g3.stream_rtsp(scene_camera=True, gaze=True)
        async with streams.scene_camera.decode() as decoded_stream, streams.gaze.decode() as decoded_gaze:
            while True: 
                ts = local_clock()
                frame, frame_timestamp = await decoded_stream.get()
                image = frame.to_ndarray(format="bgr24")

                gaze, gaze_timestamp = await decoded_gaze.get()
                if "gaze2d" in gaze:
                    gaze2d = gaze["gaze2d"] # rational (x,y)
                    h, w = frame.shape[:2] # convert to pixel location
                    x, y = int(gaze2d[0] * w), int(gaze2d[1] * h)

                # push timestamp info to lsl
                outlet_ts.push_sample([ts, frame_timestamp], ts)

                # push gaze info to lsl
                outlet_gaze.push_sample([ts, gaze_timestamp, x, y], ts)

                # show video
                cv2.imshow("Video", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        g3.close()
        cv2.destroyAllWindows()


def main():
    asyncio.run(stream_rtsp())


if __name__ == "__main__":
    main()