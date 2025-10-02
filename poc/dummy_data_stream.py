# Dummy bio signals generator to LSL -> get its time stamp -> mark it as start of fixation time
# Set up a dummy event marker and test with LabRecorder to see if they receive event marker

# dummy_biosignal_sender.py
from pylsl import StreamInfo, StreamOutlet
import time
import random

# Create dummy biosignal stream (1 channel, 100 Hz)
info = StreamInfo(name='DummyBio', type='EEG', channel_count=1,
                  nominal_srate=100, channel_format='float32', source_id='dummy_bio')
outlet = StreamOutlet(info)

print("DummyBio stream created. Sending samples...")

while True:
    sample = [random.random()]
    outlet.push_sample(sample)   # send random value
    time.sleep(0.01)             # ~100 Hz