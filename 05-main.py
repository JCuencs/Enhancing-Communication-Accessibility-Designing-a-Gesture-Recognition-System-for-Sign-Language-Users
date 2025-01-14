import cv2
import mediapipe as mp
import numpy as np
from keras.api.models import load_model

# Load your pre-trained model and action classes
model = load_model('trained_model/asl_model.h5')
actions = ['A', 'B', 'C']
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

# Variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

# Initialize MediaPipe holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image_rgb.flags.writeable = False                   # Set image to non-writeable
    results = model.process(image_rgb)                  # Make detection
    image_rgb.flags.writeable = True                    # Set image back to writeable
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # Convert back to BGR
    return image_bgr, results

def draw_landmarks(image, results):
    # Define drawing specs
    draw_spec_landmarks = mp_drawing.DrawingSpec(color=(15, 38, 208), thickness=1, circle_radius=2)
    draw_spec_connections = mp_drawing.DrawingSpec(color=(250, 246, 42), thickness=1, circle_radius=2)

    # Draw landmarks for face, pose, and hands
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  draw_spec_landmarks, draw_spec_connections)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  draw_spec_landmarks, draw_spec_connections)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  draw_spec_landmarks, draw_spec_connections)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  draw_spec_landmarks, draw_spec_connections)

def extract_keypoints(results):
    # Extract keypoints from pose, face, and hands
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, left_hand, right_hand])

# Start video capture
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        draw_landmarks(image, results)
        
        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            prediction = actions[np.argmax(res)]
            print(f"Prediction: {prediction} with confidence: {res[np.argmax(res)]:.2f}")
            predictions.append(np.argmax(res))

            # Add to sentence if above threshold and not the same as previous sign
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            # Limit sentence length to the last 5 actions
            if len(sentence) > 5:
                sentence = sentence[-5:]

        # Draw sentence on the frame
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('Sign Language Translator', image)

        # Break with q
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
