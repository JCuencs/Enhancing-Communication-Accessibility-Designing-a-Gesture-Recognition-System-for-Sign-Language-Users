import cv2
import numpy as np
import os
import time
import mediapipe as mp

# Initialize MediaPipe holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to perform MediaPipe detection on an image
def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image_rgb.flags.writeable = False                   # Set image to non-writeable
    results = model.process(image_rgb)                  # Make detection
    image_rgb.flags.writeable = True                    # Set image back to writeable
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # Convert back to BGR
    return image_bgr, results

# Function to draw landmarks for face, pose, and hands
def draw_landmarks(image, results):
    # Define drawing specs
    draw_spec_landmarks = mp_drawing.DrawingSpec(color=(15, 38, 208), thickness=1, circle_radius=2)
    draw_spec_connections = mp_drawing.DrawingSpec(color=(250, 246, 42), thickness=1, circle_radius=2)

    # Draw face landmarks
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  draw_spec_landmarks, draw_spec_connections)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  draw_spec_landmarks, draw_spec_connections)

    # Draw left hand landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  draw_spec_landmarks, draw_spec_connections)

    # Draw right hand landmarks
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  draw_spec_landmarks, draw_spec_connections)

# Function to extract keypoints from the results
def extract_keypoints(results):
    # Extract pose keypoints
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    
    # Extract face keypoints
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    
    # Extract left and right hand keypoints
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

    # Combine all keypoints
    return np.concatenate([pose, face, left_hand, right_hand])

# Data collection parameters
DATA_PATH = os.path.join('MP_Data') # path folder
actions = np.array(['A', 'B', 'C']) # action index
no_sequences = 30 # number of action sequences
sequence_length = 30 # number of frames per sequence

# Create directory structure
for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

# Capture video and process using MediaPipe holistic model
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            print(f'Starting collection for action {action}, sequence {sequence}...')
            
            # Add a 2-second pause before starting each sequence
            time.sleep(2)

            for frame_num in range(sequence_length):

                ret, frame = cap.read()
                if not ret:
                    break

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_landmarks(image, results)

                # Display info and feedback for data collection
                if frame_num == 0:
                    cv2.putText(image, f'STARTING DATA COLLECTION', (120, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Extract and save keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Display the frame
                cv2.imshow('Sign Language Data Collection Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()
