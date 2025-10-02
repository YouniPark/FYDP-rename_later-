from pylsl import StreamInfo, StreamOutlet, local_clock
from pynput import keyboard

# Create marker stream
marker_info = StreamInfo(name='ManualMarkers', type='Markers',
                         channel_count=1, nominal_srate=0,
                         channel_format='string', source_id='manual_marker')
marker_outlet = StreamOutlet(marker_info)

print("ManualMarkers stream ready, started marker lsl. Press 'm' to send a marker, 'q' to quit.")

def on_press(key):
    try:
        if key.char == 's':
            marker_outlet.push_sample(["StartedMarker"])
        if key.char == 'm':  # send marker
            ts = local_clock()
            marker_outlet.push_sample(["ManualMarker"], timestamp=ts)
            print(f"Sent ManualMarker at LSL time {ts:.5f}")
        elif key.char == 'q':  # quit program
            print("Exiting...")
            return False
    except AttributeError:
        pass  # handle special keys

# start listening to keyboard
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
