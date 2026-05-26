# 🚀 TUTORIAL LENGKAP: Cara Menjalankan Sign Language Recognition Pipeline

## 📋 Daftar Isi
1. [Prerequisites](#prerequisites)
2. [Setup Awal](#setup-awal)
3. [Pipeline Lengkap Step-by-Step](#pipeline-lengkap-step-by-step)
4. [Detail Setiap Script](#detail-setiap-script)
5. [Troubleshooting](#troubleshooting)
6. [Monitoring Progress](#monitoring-progress)

---

## Prerequisites

### ✅ Yang Perlu Disiapkan:

1. **Python 3.8+**
   ```bash
   python --version  # Check versi Python
   ```

2. **Folder Dataset**
   - Letakkan file video dalam folder `dataset/`
   - Struktur harus seperti ini:
   ```
   dataset/
   ├── Abjad/
   │   ├── A/         (video .mp4 files)
   │   ├── B/         (video .mp4 files)
   │   └── ...Z/
   ├── Angka/
   │   ├── 0/         (video .mp4 files)
   │   └── ...9/
   ├── Custom/
   │   ├── ATM/
   │   ├── Kartu_ATM/
   │   ├── Saldo/
   │   ├── Transfer/
   │   └── Uang/
   └── Keuangan/
       ├── 10/, 20/, 50/, 100/, 500/, 1000/
       └── Ribu/, Juta/, Milyar/
   ```

3. **Storage Space**
   - Minimal 20GB untuk processing
   - Lebih bagus jika punya SSD (lebih cepat dari HDD)

---

## Setup Awal

### Step 1: Buka Command Prompt / PowerShell

**Windows:**
- Tekan `Win + R`
- Ketik `cmd` atau `powershell`
- Tekan Enter

**atau buka di VS Code:**
- Tekan `Ctrl + backtick` untuk buka terminal
- Atau: Terminal → New Terminal

### Step 2: Navigate ke Project Directory

```bash
# Change directory ke project folder
cd d:\Project_Capstone_DS

# Verify Anda di folder yang benar
dir  # Lihat list files (cleansing.py, extract_landmarks.py, etc harus ada)
```

### Step 3: Install Dependencies

```bash
# Install semua packages yang dibutuhkan
pip install -r requirements.txt

# Tunggu sampai selesai (bisa 5-15 menit pertama kali)
```

**Output yang diharapkan:**
```
Successfully installed opencv-python==4.8.1.78
Successfully installed mediapipe==0.10.8
Successfully installed numpy==1.24.3
Successfully installed tensorflow==2.14.0
...
Successfully installed requests==2.31.0
```

> **⚠️ Catatan**: TensorFlow besar (~1GB). Download pertama kali bisa lambat.

---

## Pipeline Lengkap Step-by-Step

### OPSI 1: Automatic (Recommended untuk pemula)

Jika menggunakan Windows, double-click file ini:
```
run_pipeline.bat
```

**Atau di terminal:**
```bash
.\run_pipeline.bat
```

Ini akan menjalankan semua 5 steps secara otomatis.

---

### OPSI 2: Manual Step-by-Step (Untuk kontrol penuh)

#### **STEP 1️⃣: CLEANSE DATASET**
```bash
python cleansing.py
```

**Apa yang dilakukan:**
- ✅ Validasi semua video
- ✅ Hapus video corrupt
- ✅ Copy video valid ke `cleansed_dataset/`
- ✅ Parallel processing dengan 4 threads

**Output:**
```
Processing label: A
Processing label: B
...
VALID   : Abjad/A/video1.mp4
VALID   : Abjad/A/video2.mp4
INVALID : Abjad/B/corrupted.mp4
...
=== CLEANING SUMMARY ===
✅ Valid videos:   850
❌ Invalid videos: 50
📊 Total:          900
```

**Waktu estimasi:**
- 100 video → 3-5 menit
- 1000 video → 30-50 menit

**Folder yang dihasilkan:**
- `cleansed_dataset/` ← Semua video valid ada di sini

---

#### **STEP 2️⃣: EXTRACT LANDMARKS**
```bash
python extract_landmarks.py
```

**Apa yang dilakukan:**
- ✅ Extract hand landmarks dari setiap frame
- ✅ Frame skipping (ambil 1 dari 3 frame)
- ✅ Resize video (480x360 untuk speed)
- ✅ Deteksi tangan dengan MediaPipe
- ✅ Save sebagai numpy arrays
- ✅ Parallel processing dengan 2 threads

**Output:**
```
============================================================
Starting Hand Landmarks Extraction
============================================================
Frame skip:      3
Sequence length: 40 frames per video
Frame resize:    1920x1080 → 480x360
Threading:       2 workers for parallel processing

Processing label: A
Processing label: B
...
✅ BERHASIL: cleansed_dataset/A/video1.mp4 (13 frames)
✅ BERHASIL: cleansed_dataset/A/video2.mp4 (15 frames)
...
============================================================
=== EXTRACTION SUMMARY ===
============================================================
📊 Total processed:  850
✅ Success:          843
❌ Failed:           7
============================================================
```

**Waktu estimasi:**
- 100 video (1 min each) → 30-45 menit
- 1000 video (1 min each) → 300-450 menit

**Folder yang dihasilkan:**
- `extracted_landmarks/` ← Semua .npy files ada di sini

**Setiap file .npy punya format:**
- Shape: `(num_frames, 126)`
- Contoh: `(13, 126)` = 13 frames dengan 126 features per frame
- 126 = 21 landmarks × 3 (x,y,z) × 2 tangan

---

#### **STEP 3️⃣: PREPARE DATASET**
```bash
python prepare_dataset.py
```

**Apa yang dilakukan:**
- ✅ Load semua .npy files dari extracted_landmarks/
- ✅ Pad/truncate sequences ke 40 frames
- ✅ Encode labels (A → 0, B → 1, dll)
- ✅ Convert ke one-hot encoding
- ✅ Split train/test (80/20 dengan stratification)
- ✅ Save dalam format numpy array

**Output:**
```
============================================================
Preparing Dataset for Training
============================================================

Loading landmarks...
Loading: 0
  Loaded 10 files...
  Loaded 20 files...
  ...
Loading: Z
  Loaded 850 files...

============================================================
=== Dataset Statistics ===
============================================================
X shape:           (850, 40, 126) (samples, sequence_length, features)
y shape:           (850,)
Total samples:     850
Total classes:     41
Memory usage:      170.00 MB

Samples per class:
  0              :   20 samples
  1              :   22 samples
  ...
  Z              :   21 samples

Encoding labels...
Classes: ['0' '1' '2' ... 'Z']
One-hot shape: (850, 41)

Splitting dataset...
Train set: (680, 40, 126)
Test set:  (170, 40, 126)

============================================================
PREPARATION COMPLETE!
============================================================
```

**Folder yang dihasilkan:**
- `prepared_dataset/` ← 4 file numpy (.npy)
  - `X_train.npy` ← Training data
  - `X_test.npy` ← Test data
  - `y_train.npy` ← Training labels
  - `y_test.npy` ← Test labels

**Waktu estimasi:**
- Untuk 850 samples → 1-2 menit

---

#### **STEP 4️⃣: TRAIN LSTM MODEL**
```bash
python train_lstm.py
```

**Apa yang dilakukan:**
- ✅ Load prepared dataset
- ✅ Build Bidirectional LSTM model
- ✅ Train dengan callbacks (early stopping, reduce LR, checkpoint)
- ✅ Evaluate pada test set
- ✅ Generate visualisasi (graphs, confusion matrix)

**Output:**
```
============================================================
Training LSTM Model for Sign Language Recognition
============================================================

✓ Mixed precision policy applied

Loading dataset...
✓ Dataset loaded successfully

Training set shape:   (680, 40, 126)
Test set shape:       (170, 40, 126)
Training labels shape: (680, 41)
Test labels shape:    (170, 41)
Number of classes:    41

Building model...

Model Summary:
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
bidirectional (Bidirectional) (None, 40, 64)           106496
dropout (Dropout)            (None, 40, 64)           0
bidirectional_1 (Bidirectional) (None, 128)            49664
dropout_1 (Dropout)          (None, 128)              0
dense (Dense)                (None, 32)               4128
dropout_2 (Dropout)          (None, 32)               0
dense_1 (Dense)              (None, 41)               1353
=================================================================
Total params: 161,641
Trainable params: 161,641
Non-trainable params: 0
_________________________________________________________________

============================================================
Starting training...
============================================================

Epoch 1/100
21/21 [==============================] - 5s 245ms/step
     - loss: 2.3421 - accuracy: 0.1765 - val_loss: 2.1523 - val_accuracy: 0.2235

Epoch 2/100
21/21 [==============================] - 4s 192ms/step
     - loss: 1.8734 - accuracy: 0.3456 - val_loss: 1.6234 - val_accuracy: 0.4118

...

Epoch 28/100
21/21 [==============================] - 4s 192ms/step
     - loss: 0.2134 - accuracy: 0.9265 - val_loss: 0.5432 - val_accuracy: 0.8235

Epoch 29/100
21/21 [==============================] - 4s 192ms/step
EarlyStopping: Stop training (val_loss tidak improve)

============================================================
Evaluating model on test set...
============================================================

✓ Test Accuracy: 0.8235 (82.35%)
✓ Test Loss:     0.5432

Saving model...
✓ Model saved to models/final_model.h5

Generating training history graphs...
✓ Training history saved to models/training_history.png

Generating confusion matrix...
✓ Confusion matrix saved to models/confusion_matrix.png

============================================================
Classification Report (Per Class)
============================================================

              precision    recall  f1-score   support

           0       0.92      0.89      0.90        18
           1       0.88      0.91      0.89        21
           ...
           Z       0.85      0.87      0.86        19

    accuracy                           0.82       170
   macro avg       0.85      0.84      0.84       170
weighted avg       0.82      0.82      0.82       170

============================================================
TRAINING COMPLETE!
============================================================
✓ Best model:         models/best_model.h5
✓ Final model:        models/final_model.h5
✓ Training history:   models/training_history.png
✓ Confusion matrix:   models/confusion_matrix.png

Your model is ready for inference! 🎉
```

**Waktu estimasi:**
- Per epoch → 30-60 detik (pada CPU menengah)
- Total training → 30-60 menit (biasanya 20-40 epochs)

**File yang dihasilkan:**
- `models/best_model.h5` ← Model terbaik (berdasarkan validation accuracy)
- `models/final_model.h5` ← Model final
- `models/training_history.png` ← Graph accuracy & loss
- `models/confusion_matrix.png` ← Confusion matrix visualization
- `models/label_encoder.pkl` ← Label encoder untuk inference

---

#### **STEP 5️⃣: (OPTIONAL) TEST MODEL**

Jika sudah punya `realtime_inference.py`:

```bash
python realtime_inference.py
```

Ini untuk test model dengan camera real-time.

---

## Detail Setiap Script

### 📄 cleansing.py
**File:** `cleansing.py`
**Fungsi:** Validasi dan pembersihan dataset

| Aspek | Detail |
|-------|--------|
| Input | `dataset/` folder dengan video .mp4 |
| Output | `cleansed_dataset/` dengan video valid |
| Validasi | Min frames (30), FPS > 0, not corrupted |
| Threading | 4 parallel threads |
| Time | 100 video = 5 min |

**Cara menjalankan:**
```bash
python cleansing.py
```

---

### 📄 extract_landmarks.py
**File:** `extract_landmarks.py`
**Fungsi:** Extract hand landmarks dari video

| Aspek | Detail |
|-------|--------|
| Input | `cleansed_dataset/` |
| Output | `extracted_landmarks/` (.npy files) |
| Format | (num_frames, 126) per video |
| Optimization | Frame skip 3, Resize 480x360 |
| Threading | 2 parallel threads |
| Time | 100 video (1 min each) = 45 min |

**Cara menjalankan:**
```bash
python extract_landmarks.py
```

---

### 📄 prepare_dataset.py
**File:** `prepare_dataset.py`
**Fungsi:** Persiapan data untuk training

| Aspek | Detail |
|-------|--------|
| Input | `extracted_landmarks/` |
| Output | `prepared_dataset/` (4 .npy files) |
| Processing | Pad to 40 frames, encode labels, split 80/20 |
| Output Shape | X_train: (680, 40, 126) |
| Time | 850 samples = 2 min |

**Cara menjalankan:**
```bash
python prepare_dataset.py
```

---

### 📄 train_lstm.py
**File:** `train_lstm.py`
**Fungsi:** Training LSTM model

| Aspek | Detail |
|-------|--------|
| Input | `prepared_dataset/` |
| Output | `models/` (best_model.h5, final_model.h5, png files) |
| Model | Bidirectional LSTM (2 layers) |
| Batch Size | 8 (CPU optimized) |
| Early Stopping | Patience 10 |
| Time | Per epoch = 45 sec, Total = 45 min |

**Cara menjalankan:**
```bash
python train_lstm.py
```

---

## Troubleshooting

### ❌ Error: "No module named 'tensorflow'"

**Solusi:**
```bash
# Reinstall TensorFlow
pip install --upgrade tensorflow

# Atau install specific version
pip install tensorflow==2.14.0
```

---

### ❌ Error: "No module named 'mediapipe'"

**Solusi:**
```bash
pip install mediapipe==0.10.8
```

---

### ❌ Error: "dataset directory not found"

**Solusi:**
1. Check folder structure: `d:\Project_Capstone_DS\dataset\`
2. Pastikan file video ada di dalam subfolder (A/, B/, 0/, dll)
3. Pastikan file video punya extension `.mp4`

---

### ❌ Error: "CUDA not available" (jika punya GPU)

**Solusi:**
- Tidak masalah, code akan otomatis use CPU
- TensorFlow akan berjalan lebih lambat tapi tetap work

---

### ❌ Error: "Memory usage too high" / Program crash

**Solusi:**
Kurangi batch size di `config.py`:
```python
BATCH_SIZE = 4  # dari 8
```

atau kurangi FRAME_SKIP:
```python
FRAME_SKIP = 5  # dari 3 (lebih sedikit frame)
```

---

### ❌ Extract landmarks sangat lambat

**Solusi:**

Increase FRAME_SKIP di `config.py`:
```python
FRAME_SKIP = 5  # Ambil 1 dari 5 frame (lebih cepat, kurang detail)
```

atau kurangi SEQUENCE_LENGTH:
```python
SEQUENCE_LENGTH = 20  # dari 40
```

---

## Monitoring Progress

### 📊 Cara Monitor Training

**Selama training, lihat:**

1. **Loss & Accuracy**
   ```
   Epoch 5/100
   21/21 [==============================] - 5s 245ms/step
        loss: 1.2345 - accuracy: 0.7234
        val_loss: 1.4567 - val_accuracy: 0.6891
   ```
   - ✅ Loss turun = model belajar
   - ✅ Accuracy naik = model lebih akurat

2. **Early Stopping**
   - Jika validation loss tidak improve → model stop automatically
   - Ini normal, bukan error!

3. **Learning Rate Reduction**
   ```
   Reduce LR by factor 0.5
   ```
   - Learning rate dibuat lebih kecil untuk fine-tuning

---

### 📈 Monitoring Resource Usage

**Buka Task Manager (Windows):**
1. Tekan `Ctrl + Shift + Esc`
2. Lihat Performance tab
3. Monitor CPU usage (target 70-90%)
4. Monitor Memory usage

---

### 📁 Monitoring Output Files

**Setiap step menghasilkan folder:**

```
Project_Capstone_DS/
├── cleansed_dataset/       ← Setelah step 1
├── extracted_landmarks/    ← Setelah step 2
├── prepared_dataset/       ← Setelah step 3
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   └── y_test.npy
└── models/                 ← Setelah step 4
    ├── best_model.h5
    ├── final_model.h5
    ├── training_history.png
    ├── confusion_matrix.png
    └── label_encoder.pkl
```

**Check file size untuk verify:**
```bash
# Windows PowerShell
Get-Item models/best_model.h5 | Format-List Length
```

---

## 🎯 Checklist - Sebelum Mulai

- [ ] Python 3.8+ sudah installed
- [ ] Folder `dataset/` exist dengan structure yang benar
- [ ] Terminal sudah di folder project
- [ ] Jalankan: `pip install -r requirements.txt`
- [ ] Punya minimal 20GB free storage
- [ ] CPU tidak terlalu panas (cool it down dulu)

---

## ✅ Checklist - Setelah Selesai

- [ ] `cleansed_dataset/` ada dan berisi video
- [ ] `extracted_landmarks/` ada dan berisi .npy files
- [ ] `prepared_dataset/` ada dan berisi 4 .npy files
- [ ] `models/` ada dan berisi:
  - [ ] `best_model.h5`
  - [ ] `final_model.h5`
  - [ ] `training_history.png`
  - [ ] `confusion_matrix.png`
  - [ ] `label_encoder.pkl`

---

## 🚀 Next Steps

Setelah model ready:
1. Use `realtime_inference.py` untuk real-time detection
2. Fine-tune model dengan lebih banyak data
3. Deploy ke production

---

## 📞 Support

Jika ada error:
1. Check error message di terminal
2. Look di section **Troubleshooting**
3. Google error message
4. Check file permissions

---

**Generated:** May 26, 2026  
**Optimized for:** CPU dengan kecepatan menengah ke lambat  
**Total Processing Time:** ~2-3 jam untuk full pipeline (100-1000 videos)
