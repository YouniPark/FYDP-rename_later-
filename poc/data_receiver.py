## Run "open /opt/homebrew/Cellar/labrecorder/1.16.5_9/LabRecorder/LabRecorder.app" in terminal first
# dummy_biosignal_receiver.py
from pylsl import StreamInlet, resolve_streams, resolve_byprop

print("Looking for DummyBio stream...")
streams = resolve_byprop("type", "EEG")
inlet = StreamInlet(streams[0])
print("Connected to DummyBio stream.")

while True:
    sample, timestamp = inlet.pull_sample()
    print(f"Received {sample[0]:.4f} at LSL time {timestamp:.5f}")