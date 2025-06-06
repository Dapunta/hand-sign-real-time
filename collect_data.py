import os, time
import csv, cv2, mediapipe as mp

from utils.distance import extract_distance_features
from utils.config import dataset_name, cameras

#--> Konfigurasi data yg akan dicollect
label        : str = "S"          #--> Ganti sesuai huruf yang ingin direkam
dataset_path : str = dataset_name #--> Path dataset
samples      : int = 300          #--> Jumlah data per huruf

#--> Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

#--> Webcam
cap = cv2.VideoCapture(cameras)
data = []
saved = 0

print(f"Mulai merekam data untuk label '{label.upper()}' dalam 3 detik...")
time.sleep(3)
print("Merekam...")

while saved < samples:
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
            features.append(label.upper())
            data.append(features)
            saved += 1
            print(f"Tersimpan: {saved}/{samples}")

    cv2.imshow("Collecting Data", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
cap.release()
cv2.destroyAllWindows()

#--> Simpan ke CSV
if not os.path.exists(os.path.dirname(dataset_path)):
    os.makedirs(os.path.dirname(dataset_path))

header = [f"d{i}" for i in range(len(data[0]) - 1)] + ["label"]
with open(dataset_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow(header)
    writer.writerows(data)

print(f"✅ Data untuk label '{label.upper()}' berhasil disimpan di {dataset_path}")