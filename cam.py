import cv2
import pickle, numpy as np, mediapipe as mp, tensorflow as tf

from utils.distance import extract_distance_features
from utils.config import scaler_name, binarizer_name, model_name, cameras

#--> Load model dan preprocessing
model = tf.keras.models.load_model(model_name)

with open(scaler_name, "rb") as f:
    scaler = pickle.load(f)

with open(binarizer_name, "rb") as f:
    lb = pickle.load(f)

#--> Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

#--> Webcam
cap = cv2.VideoCapture(cameras)

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