# ====== collect_data.py (refactored with distance features) ======
import cv2
import csv
import time
import mediapipe as mp
import numpy as np
import os

# Konfigurasi label yang akan dikumpulkan
LABEL = "F"  # Ganti sesuai huruf yang ingin direkam
SAVE_PATH = "model3/mediapipe_landmarks.csv"
SAMPLES = 300  # Jumlah data per huruf

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Fungsi jarak Euclidean antara dua titik
def distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# Fungsi untuk ekstraksi fitur berbasis jarak antar titik
# Menggunakan pasangan titik tertentu (misal ujung jari ke pangkal jari)
def extract_distance_features(landmarks):
    pairs = [
        (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),  # wrist ke ujung jari
        (4, 8), (8, 12), (12, 16), (16, 20),          # antar ujung jari
        (5, 9), (9, 13), (13, 17),                    # antar MCP
        (2, 6), (6, 10), (10, 14), (14, 18)           # antar IP/PIP
    ]
    features = [distance(landmarks[a], landmarks[b]) for a, b in pairs]
    return features

# Buka webcam
cap = cv2.VideoCapture(1)
data = []
saved = 0

print(f"Mulai merekam data untuk label '{LABEL}' dalam 3 detik...")
time.sleep(3)
print("Merekam...")

while saved < SAMPLES:
    ret, frame = cap.read()
    if not ret:
        continue

    image = frame.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            features = extract_distance_features(hand_landmarks.landmark)
            features.append(LABEL)
            data.append(features)
            saved += 1
            print(f"Tersimpan: {saved}/{SAMPLES}")

    cv2.imshow("Collecting Data", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Simpan ke CSV
if not os.path.exists(os.path.dirname(SAVE_PATH)):
    os.makedirs(os.path.dirname(SAVE_PATH))

header = [f"d{i}" for i in range(len(data[0]) - 1)] + ["label"]
with open(SAVE_PATH, mode='a', newline='') as f:
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow(header)
    writer.writerows(data)

print(f"✅ Data untuk label '{LABEL}' berhasil disimpan di {SAVE_PATH}")