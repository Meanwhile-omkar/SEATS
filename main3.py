import cv2
import dlib
import numpy as np
from tracking import GazeTracking
import matplotlib.pyplot as plt

# Load pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

# 3D model points for the head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),     # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left Mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
])

# Camera matrix (assumed values)
camera_matrix = np.array([[640, 0, 320],
                           [0, 640, 240],
                           [0, 0, 1]], dtype="double")

dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

def get_head_pose(landmarks):
    # Get 2D image points from landmarks
    image_points = np.array([
        (landmarks[30].x, landmarks[30].y),   # Nose tip
        (landmarks[8].x, landmarks[8].y),     # Chin
        (landmarks[36].x, landmarks[36].y),   # Left eye left corner
        (landmarks[45].x, landmarks[45].y),   # Right eye right corner
        (landmarks[48].x, landmarks[48].y),   # Left Mouth corner
        (landmarks[54].x, landmarks[54].y)    # Right mouth corner
    ], dtype="double")

    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Calculate Euler angles (yaw, pitch, roll)
    pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]) * 180 / np.pi
    yaw = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2)) * 180 / np.pi
    roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * 180 / np.pi

    return yaw, pitch, roll

while True:
    # Get a new frame from the webcam
    _, frame = webcam.read()
    gaze.refresh(frame)

    # Analyze gaze and annotate the frame
    annotated_frame = gaze.annotated_frame()
    gaze_text = ""

    # Check gaze direction
    if gaze.is_blinking():
        gaze_text = "Blinking"
    elif gaze.is_right():
        gaze_text = "Looking right"
    elif gaze.is_left():
        gaze_text = "Looking left"
    elif gaze.is_center():
        gaze_text = "Looking center"

    # Annotate the gaze text
    cv2.putText(annotated_frame, gaze_text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    # Detect faces and landmarks
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)

        # Get head pose
        yaw, pitch, roll = get_head_pose(landmarks.parts())

        # Determine attentiveness
        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()

        if left_pupil is not None and right_pupil is not None:
            attention_status = "Attentive"
        else:
            # Check head position to determine if distracted
            # Add conditions for distraction based on head pose and visibility
            if abs(yaw) > 30 or abs(pitch) > 30:  # Adjust thresholds based on testing
                attention_status = "Distracted"
            else:
                attention_status = "Unknown"  # Assume unknown if not explicitly attentive or distracted
        
        # Annotate attention status
        cv2.putText(annotated_frame, f"Attention: {attention_status}", (90, 100), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    # Convert frame from BGR to RGB for Matplotlib
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Display the frame using Matplotlib
    plt.imshow(annotated_frame_rgb)
    plt.axis('off')  # Hide axes
    plt.pause(0.001)  # Pause to allow the plot to update

    

# Cleanup
webcam.release()
cv2.destroyAllWindows()
