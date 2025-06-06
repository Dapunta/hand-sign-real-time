import numpy as np

#--> Fungsi jarak Euclidean antara dua titik
def distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

#--> Fungsi untuk ekstraksi fitur berbasis jarak antar titik
#--> Menggunakan pasangan titik tertentu (misal ujung jari ke pangkal jari)
def extract_distance_features(landmarks):
    pairs = [
        (0, 4), (0, 8), (0, 12), (0, 16), (0, 20), #--> wrist ke ujung jari
        (4, 8), (8, 12), (12, 16), (16, 20),       #--> antar ujung jari
        (5, 9), (9, 13), (13, 17),                 #--> antar MCP
        (2, 6), (6, 10), (10, 14), (14, 18)        #--> antar IP/PIP
    ]
    features = [distance(landmarks[a], landmarks[b]) for a, b in pairs]
    return features