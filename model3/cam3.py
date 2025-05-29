import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle

# Load model dan preprocessing
model = tf.keras.models.load_model("model3/model3.h5")

with open("model3/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model3/label_binarizer.pkl", "rb") as f:
    lb = pickle.load(f)

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Fungsi jarak Euclidean antar dua titik
def distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# Fitur distance antar titik (HARUS SAMA DENGAN collect_data.py!)
def extract_distance_features(landmarks):
    pairs = [
        (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),  # wrist ke ujung jari
        (4, 8), (8, 12), (12, 16), (16, 20),        # antar ujung jari
        (5, 9), (9, 13), (13, 17),                  # antar MCP
        (2, 6), (6, 10), (10, 14), (14, 18)         # antar IP/PIP
    ]
    features = [distance(landmarks[a], landmarks[b]) for a, b in pairs]
    return features

# Webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = frame.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = extract_distance_features(hand_landmarks.landmark)
            if len(features) == 16:
                features_np = np.array(features).reshape(1, -1)
                features_scaled = scaler.transform(features_np)
                prediction = model.predict(features_scaled, verbose=0)
                pred_index = np.argmax(prediction)
                label = lb.classes_[pred_index]
                confidence = prediction[0][pred_index] * 100

                cv2.putText(image, f"{label} ({confidence:.1f}%)", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.imshow("Hand Sign Recognition", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
