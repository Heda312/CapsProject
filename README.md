# BISINDO Realtime Sign Language Recognition

READY TO USE!
Langsung di run saja "realtime_inference.py"
python realtime_inference.py

Project deteksi bahasa isyarat BISINDO secara realtime menggunakan:

- MediaPipe
- LSTM
- TensorFlow
- OpenCV

Program menggunakan webcam untuk mendeteksi gesture tangan dan mengubahnya menjadi hasil prediksi gesture.

---

# Requirements

- Python 3.11
- Webcam

Install dependency:

```bash
pip install -r requirements.txt
```

---

# Struktur Folder

```bash
project/
│
├── dataset_clean/
├── dataset_landmarks/
├── prepared_dataset/
├── models/
│
├── landmark_extraction_2hand.py
├── prepare_dataset.py
├── train_lstm.py
├── realtime_inference.py
│
├── requirements.txt
└── README.md
```

---

# Cara Menjalankan

notes: Jika mau langsung test bisa melanjutkan langkah dibawah, namun jika ingin dari tahap ambil dataset bisa di mulai dari awal prosesnya dengan menjalankan "collect_dataset.

## 1. Landmark Extraction

Mengubah video dataset menjadi landmark sequence.

```bash
python landmark_extraction_2hand.py
```

Output:

```bash
dataset_landmarks/
```

---

## 2. Prepare Dataset

Membuat dataset training, validation, dan testing.

```bash
python prepare_dataset.py
```

Output:

```bash
prepared_dataset/
```

---

## 3. Training Model

Melatih model LSTM menggunakan dataset landmark.

```bash
python train_lstm.py
```

Output:

```bash
models/final_lstm_model.keras
```

---

## 4. Realtime Inference

Menjalankan deteksi gesture realtime menggunakan webcam.

```bash
python realtime_inference.py
```

Tekan:

```bash
ESC
```

untuk keluar.

---

# Features

- 2-Hand Detection
- Landmark Normalization
- Mirror Augmentation
- Confidence Threshold
- Prediction Smoothing
- Realtime Webcam Detection

---

# Notes

- Semakin banyak variasi dataset, semakin stabil hasil realtime.
- Disarankan menggunakan:
  - lighting berbeda
  - background berbeda
  - tangan kiri & kanan
  - beberapa orang berbeda

Karena model AI ternyata tidak otomatis memahami bahwa semua manusia punya tangan yang bentuknya sedikit berbeda. Matematika memang sangat berbakat, tapi tetap perlu banyak contoh supaya tidak panik melihat gesture baru.
