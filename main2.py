import cv2
from tracking import GazeTracking
import time

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

# Variables to track attentiveness
attentive_time = 0
distracted_time = 0
blink_count = 0
max_distracted_time = 3  # seconds before considering as distracted
max_attentive_time = 3    # seconds before considering as attentive

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    # Check the gaze state
    if gaze.is_blinking():
        blink_count += 1
        text = "Blinking"
    elif gaze.is_right():
        distracted_time += 1  # Increment distracted time
        attentive_time = 0  # Reset attentive time
        text = "Looking right"
    elif gaze.is_left():
        distracted_time += 1  # Increment distracted time
        attentive_time = 0  # Reset attentive time
        text = "Looking left"
    elif gaze.is_center():
        attentive_time += 1  # Increment attentive time
        distracted_time = 0  # Reset distracted time
        text = "Looking center"
    elif left_pupil is None and right_pupil is None:
        distracted_time += 1  # Increment distracted time
        attentive_time = 0
        text = "Distracted"
    # Determine attentiveness
    if attentive_time > max_attentive_time:
        attention_status = "Attentive"
    elif distracted_time > max_distracted_time:
        attention_status = "Distracted"
    else:
        attention_status = "Monitoring..."

    cv2.putText(frame, f"Status: {attention_status}", (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
