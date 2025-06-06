
# Real-Time Hand-Sign Language Recognition Using MediaPipe and MLP

## Deskripsi
Sistem ini merupakan aplikasi deteksi bahasa isyarat tangan (huruf alfabet) secara **real-time** menggunakan webcam, MediaPipe, dan Multi-Layer Perceptron (MLP). Semua pengenalan berbasis pada **fitur jarak antar landmark tangan**, sehingga lebih stabil meski tangan berpindah posisi di kamera.

---

## Latar Belakang
Komunikasi menggunakan bahasa isyarat sangat penting bagi penyandang tuna wicara dan tuna rungu. Namun, tidak semua orang memahami bahasa isyarat. Sistem ini hadir untuk menerjemahkan gestur tangan ke huruf secara otomatis dan real-time, membantu jembatan komunikasi antara penyandang disabilitas dan masyarakat umum.

---

## Hand Sign

![hand_sign](/assets/hand_sign.png)

---

## Tujuan
- Membangun sistem yang dapat mengenali dan menerjemahkan gesture tangan ke huruf secara real-time.
- Menghasilkan akurasi tinggi dengan metode machine learning berbasis MLP.
- Menjadi dasar sistem komunikasi berbasis gesture yang dapat dikembangkan lebih lanjut.

---

## Teknologi & Library yang Digunakan
- [MediaPipe](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html) (landmark deteksi tangan)
- [OpenCV](https://opencv.org/) (webcam & visualisasi)
- [TensorFlow/Keras](https://www.tensorflow.org/) (model MLP)
- [scikit-learn](https://scikit-learn.org/) (preprocessing & label encoder)
- [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [Pandas](https://pandas.pydata.org/)

---

## **Alur & Metode**

### 1. Data Collection (`collect_data.py`)
- Dataset yang digunakan adalah hasil generate sendiri *(gesture capture)*
- Label (misal: A, B, C, ...) diberikan manual saat collect.
   ```py
   #--> konfigurasi data yg akan dicollect
   label   : str = "S" #--> ganti sesuai huruf yang ingin direkam
   samples : int = 300 #--> jumlah data per huruf
   ```
- Menggunakan webcam dan MediaPipe, landmark tangan (21 titik) dideteksi.
- Dari landmark, dihitung **16 jarak Euclidean** antar titik-titik penting (wrist ke ujung jari, antar ujung jari, dsb).
   ```py
   #--> fungsi jarak euclidean antara dua titik
   def distance(p1, p2):
      return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
   ```
   ```py
   #--> fungsi untuk ekstraksi fitur berbasis jarak antar titik
   #--> menggunakan pasangan titik tertentu (misal ujung jari ke pangkal jari)
   def extract_distance_features(landmarks):
       pairs = [
           (0, 4), (0, 8), (0, 12), (0, 16), (0, 20), #--> wrist ke ujung jari
           (4, 8), (8, 12), (12, 16), (16, 20),       #--> antar ujung jari
           (5, 9), (9, 13), (13, 17),                 #--> antar MCP
           (2, 6), (6, 10), (10, 14), (14, 18)        #--> antar IP/PIP
       ]
       features = [distance(landmarks[a], landmarks[b]) for a, b in pairs]
       return features
   ```
   | Fitur | Pasangan Titik | Keterangan                                           |
   | ----- | -------------- | ---------------------------------------------------- |
   | `d0`  | (0, 4)         | Jarak dari **wrist ke ujung ibu jari**               |
   | `d1`  | (0, 8)         | Jarak dari **wrist ke ujung telunjuk**               |
   | `d2`  | (0, 12)        | Jarak dari **wrist ke ujung jari tengah**            |
   | `d3`  | (0, 16)        | Jarak dari **wrist ke ujung jari manis**             |
   | `d4`  | (0, 20)        | Jarak dari **wrist ke ujung kelingking**             |
   | `d5`  | (4, 8)         | Jarak dari **ujung ibu jari ke ujung telunjuk**      |
   | `d6`  | (8, 12)        | Jarak dari **ujung telunjuk ke ujung jari tengah**   |
   | `d7`  | (12, 16)       | Jarak dari **ujung jari tengah ke ujung jari manis** |
   | `d8`  | (16, 20)       | Jarak dari **ujung jari manis ke ujung kelingking**  |
   | `d9`  | (5, 9)         | Jarak antar **MCP telunjuk dan MCP jari tengah**     |
   | `d10` | (9, 13)        | Jarak antar **MCP jari tengah dan MCP jari manis**   |
   | `d11` | (13, 17)       | Jarak antar **MCP jari manis dan MCP kelingking**    |
   | `d12` | (2, 6)         | Jarak antar **IP ibu jari dan PIP telunjuk**         |
   | `d13` | (6, 10)        | Jarak antar **PIP telunjuk dan PIP jari tengah**     |
   | `d14` | (10, 14)       | Jarak antar **PIP jari tengah dan PIP jari manis**   |
   | `d15` | (14, 18)       | Jarak antar **PIP jari manis dan PIP kelingking**    |
- Ringkasan Kelompok Fitur:
   - d0 – d4: Jarak wrist ke tiap ujung jari (panjang dan arah jari)
   - d5 – d8: Jarak antar ujung jari (posisi relatif jari-jari)
   - d9 – d11: Jarak antar MCP (lebar tangan di pangkal jari)
   - d12 – d15: Jarak antar IP/PIP (kelengkungan atau tekukan jari)
- Visualisasi
   | &nbsp;                  | &nbsp;                  |
   | ----------------------- | ----------------------- |
   | ![ss1](/assets/ss1.png) | ![ss2](/assets/ss2.png) |
- Data disimpan ke `data/landmarks.csv`.

### 2. Training Model (`train.py`)
- Load data dari CSV, pisahkan fitur & label
- Normalisasi fitur *(`StandardScaler`)*
   - Agar semua fitur memiliki skala yang sama
   - Supaya neural network tidak berat sebelah terhadap fitur besar
- One-Hot Encoding label *(`LabelBinarizer`)*
   - Label teks seperti `A`, `B`, `C` diubah menjadi vektor biner *(one-hot)*
   - Dibutuhkan karena output dari model MLP berupa vektor probabilitas multi-kelas
- Train-Test Split
   - Membagi data menjadi **80% training** dan **20% testing**
- Arsitektur Model MLP (Multi-Layer Perceptron)
   ```py
   #--> MLP Model
   model = tf.keras.Sequential([
       tf.keras.layers.Dense(256, activation='relu', input_shape=(X.shape[1],)),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Dense(64, activation='relu'),
       tf.keras.layers.Dense(len(lb.classes_), activation='softmax')
   ])
   model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   ```
   - Dense(256) → Dropout → Dense(128) → Dropout → Dense(64) → Output
      - `Dense` : layer yang menghubungkan semua neuron antar layer
      - `Dropout` : menonaktifkan sebagian neuron secara acak saat training untuk mencegah overfitting
      - `ReLU` : fungsi aktivasi yang hanya meneruskan nilai positif dan mengabaikan nilai negatif
   - Softmax mengubah output menjadi probabilitas tiap kelas (hasil antara 0 dan 1, total = 1)
   - Compile Model
      - `Optimizer` : Adam --> adaptif, cepat konvergen
      - `Loss` : categorical_crossentropy --> digunakan saat label dalam format one-hot encoding
      - `Metric` : accuracy --> untuk memantau performa klasifikasi saat training dan evaluasi
- Model, scaler, dan label encoder disimpan untuk deployment
   - `model.h5` : format model TensorFlow
   - `scaler.pkl` : standar untuk normalisasi data baru
   - `label_binarizer.pkl` : konversi label teks saat prediksi
- Evaluasi : classification report, confusion matrix, akurasi akhir
   ![cfm](/assets/confusion_matrix.png)

### 3. Real-Time Prediction (`cam.py`)
- Webcam aktif, tangan dideteksi dengan MediaPipe.
- Landmark diolah menjadi 16 fitur jarak, distandarkan dengan scaler.
- Model melakukan prediksi kelas/huruf, hasil dan confidence (%) ditampilkan di layar webcam secara live.
   | &nbsp;                      | &nbsp;                      | &nbsp;                      |
   | --------------------------- | --------------------------- | --------------------------- |
   | ![test1](/assets/test1.png) | ![test2](/assets/test2.png) | ![test3](/assets/test3.png) |

---

## **Keunggulan**
- Fitur jarak antar titik membuat sistem lebih robust (tahan noise posisi tangan).
- Mudah di-extend ke banyak huruf atau gesture baru.
- Cepat dan ringan untuk real-time.
- Sumber data bisa dikembangkan mandiri (fleksibel).

---

## **Langkah Menjalankan Proyek**

1. **Kumpulkan data:**
   ```
   python collect_data.py
   ```
   Rekam gesture untuk setiap huruf, sesuaikan label.

2. **Latih model:**
   ```
   python train.py
   ```

3. **Jalankan real-time deteksi:**
   ```
   python cam.py
   ```

---

## Catatan
- Dataset dasar alphabet, bisa diperluas ke gesture lain.
- Untuk hasil terbaik, pastikan pose jari konsisten saat collect data.

---

## Referensi
- [Sign Language MNIST Dataset](https://www.kaggle.com/datamunge/sign-language-mnist) _(referensi awal, namun project ini dataset-nya hasil collect mandiri)_

---

> _Feel free to modify, extend, and share!_