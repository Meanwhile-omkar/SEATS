from flask import Flask, request, jsonify
import cv2
import dlib
import numpy as np
from tracking import GazeTracking
from fer import FER
import base64

app = Flask(__name__)

# Load pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Initialize GazeTracking
gaze = GazeTracking()

# Initialize emotion detector
emotion_detector = FER()

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

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get the frame from the request (base64 encoded image)
    data = request.json.get('frame', None)
    if data is None:
        return jsonify({'error': 'No frame provided'}), 400

    # Decode the image from base64
    img_data = base64.b64decode(data)
    np_img = np.frombuffer(img_data, dtype=np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Process the frame
    gaze.refresh(frame)

    # Analyze gaze and annotate the frame
    gaze_text = ""
    if gaze.is_blinking():
        gaze_text = "Blinking"
    elif gaze.is_right():
        gaze_text = "Looking right"
    elif gaze.is_left():
        gaze_text = "Looking left"
    elif gaze.is_center():
        gaze_text = "Looking center"

    # Detect faces and landmarks
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    attention_status = "Unknown"
    emotion_text = "No face detected"
    
    for face in faces:
        landmarks = predictor(gray, face)

        # Get head pose
        yaw, pitch, roll = get_head_pose(landmarks.parts())

        # Emotion detection
        emotions = emotion_detector.detect_emotions(frame)
        if emotions:
            dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
            emotion_text = f"Emotion: {dominant_emotion}"

        # Determine attentiveness
        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()

        if left_pupil is not None and right_pupil is not None:
            attention_status = "Attentive"
        else:
            if abs(yaw) > 30 or abs(pitch) > 30:
                attention_status = "Distracted"
            else:
                attention_status = "Unknown"

    # Return analysis result as JSON
    return jsonify({
        'attentionStatus': attention_status,
        'emotion': emotion_text,
        'gazeText': gaze_text
    })

# Ensure to replace '__seats__' with '__main__'
if __name__ == '__main__':
    app.run(debug=True)
