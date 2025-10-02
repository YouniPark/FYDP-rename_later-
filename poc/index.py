from pylsl import StreamInfo, StreamOutlet, local_clock
import cv2

# Multiple Independent Streams (EEG, eye tracker, markers)
# LSL will automatically synchronizes data to the same global clock

info_eeg = StreamInfo()
outlet_eeg = StreamOutlet(info_eeg)

# Eye tracker gaze coordinate stream (# channels: x, y, z?) 2D or 3D?
# nominal_srate: sampling rate in Hz
info_gaze = StreamInfo(name = 'EyeTracker', type = 'EyeGaze', channel_count = 2,
                       nominal_srate = 60, channel_format, source_id)
outlet_gaze = StreamOutlet(info_gaze)

# Event marker stream (1 channel, string)
info_markers = StreamInfo('Markers', 'Events', 1, 0, 'string', 'marker')
outlet_markers = StreamOutlet(info_markers) 

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

ip_url = "" # IP URL of Tobii camera
cap = cv2.VideoCapture(ip_url) 

# Get Eye Tracker Output

# time, coordinate (x, y) 
time = 123 # change with real values
x_gaze = 123
y_gaze = 123

# Identifying Faces in the Video Stream

# Function to detect faces in the video stream and draw a bounding box around them:
def detect_bounding_box(vid):
    
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        # Print coordinates
        print(f"Face detected at X:{x}, Y:{y}, Width:{w}, Height:{h}")
        
    return faces

event_time = 0

while True:

    result, video_frame = cap.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(
        video_frame
    )  # apply the function we created to the video frame

    # --- Algorithm to mark event (Vivien) 
    # --- #

    cv2.imshow(
        "My Face Detection Project", video_frame
    )  # display the processed frame in a window named "My Face Detection Project"

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()