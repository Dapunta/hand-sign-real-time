
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
- Menggunakan webcam dan MediaPipe, landmark tangan (21 titik) dideteksi.
- Dari landmark, dihitung **16 jarak Euclidean** antar titik-titik penting (wrist ke ujung jari, antar ujung jari, dsb).
- Label (misal: A, B, C, ...) diberikan manual saat collect.
- Data disimpan ke `model3/mediapipe_landmarks.csv`.

### 2. Training Model (`train.py`)
- Data dari CSV di-load, fitur dinormalisasi (StandardScaler), label di-one-hot encoding (`LabelBinarizer`).
- Data dibagi train-test (stratified, 80:20).
- Model **MLP** dilatih (Dense(256) → Dropout → Dense(128) → Dropout → Dense(64) → Output).
- Model, scaler, dan label encoder disimpan untuk deployment.
- Evaluasi: classification report, confusion matrix, akurasi akhir.

### 3. Real-Time Prediction (`cam.py`)
- Webcam aktif, tangan dideteksi dengan MediaPipe.
- Landmark diolah menjadi 16 fitur jarak, distandarkan dengan scaler.
- Model melakukan prediksi kelas/huruf, hasil dan confidence (%) ditampilkan di layar webcam secara live.

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

## Contoh Output
- Hasil prediksi muncul di tampilan webcam, contoh:  
  `A (99.1%)`

- Laporan evaluasi di terminal berupa confusion matrix dan classification report.

---

## Catatan
- Dataset dasar alphabet, bisa diperluas ke gesture lain.
- Untuk hasil terbaik, pastikan pose jari konsisten saat collect data.

---

## Referensi
- [Sign Language MNIST Dataset](https://www.kaggle.com/datamunge/sign-language-mnist) _(referensi awal, namun project ini dataset-nya hasil collect mandiri)_

---

> _Feel free to modify, extend, and share!_