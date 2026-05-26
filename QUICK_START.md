# ⚡ QUICK START GUIDE - Cara Cepat Menjalankan Pipeline

## 🎯 TL;DR (Too Long; Didn't Read)

**Jika Anda dalam hurry, ikuti ini:**

### Opsi A: Automatic (Rekomendasi)

**Di Windows:**
```
1. Double-click: run_pipeline.bat
2. Tunggu sampai selesai (~2-3 jam)
3. Model sudah ready di: models/
```

---

### Opsi B: Manual Command (Sedikit lebih longgar)

**Buka PowerShell/Command Prompt di folder project:**

```bash
# Step 1: Setup (hanya sekali)
pip install -r requirements.txt

# Step 2: Cleanse data
python cleansing.py

# Step 3: Extract landmarks
python extract_landmarks.py

# Step 4: Prepare dataset
python prepare_dataset.py

# Step 5: Train model
python train_lstm.py
```

---

## 📊 Timeline Estimasi

| Step | Deskripsi | Waktu |
|------|-----------|-------|
| 1️⃣ Setup | Install packages | 10 menit |
| 2️⃣ Cleanse | Validasi video | 5-50 menit |
| 3️⃣ Extract | Extract landmarks | 45-450 menit |
| 4️⃣ Prepare | Prepare dataset | 2 menit |
| 5️⃣ Train | Train model | 30-60 menit |
| **TOTAL** | **Semuanya** | **1-3 jam** |

---

## 🔧 Requirements

- Python 3.8+
- 20GB free storage
- Video dataset di folder `dataset/`
- Ketenangan (jangan interrupt process)

---

## ✅ Verify Setup

```bash
# Navigate ke folder project
cd d:\Project_Capstone_DS

# Check Python version
python --version

# Check required packages installed
python -c "import tensorflow, mediapipe, cv2; print('✓ All packages OK')"

# Check dataset
dir dataset\
```

---

## 📁 Input Folder Structure

```
dataset/
├── Abjad/
│   ├── A/  (50+ .mp4 files)
│   ├── B/
│   └── ...
├── Angka/
│   ├── 0/
│   └── ...
├── Custom/
│   └── ATM/
└── Keuangan/
    └── JUTA/
```

**Minimal requirements:**
- Minimal 1 video per label
- Better: 10-20+ video per label
- Setiap video minimal 30 frames (1 detik pada 30fps)

---

## 📈 Output Folder Structure

Setelah selesai, akan ada:

```
cleansed_dataset/          (dari step 1)
extracted_landmarks/       (dari step 2)
prepared_dataset/          (dari step 3)
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   └── y_test.npy
models/                    (dari step 4)
│   ├── best_model.h5         ← USE THIS untuk inference
│   ├── final_model.h5
│   ├── training_history.png  ← Accuracy graphs
│   ├── confusion_matrix.png  ← Performance matrix
│   └── label_encoder.pkl     ← Label decoder
```

---

## 🚨 Jika Ada Error

### Error: ModuleNotFoundError
```bash
pip install -r requirements.txt
```

### Error: dataset not found
- Check folder exists: `d:\Project_Capstone_DS\dataset\`
- Check video files ada di subfolder (A/, B/, 0/, dll)

### Error: Out of Memory
- Kurangi batch size di `config.py`:
  ```python
  BATCH_SIZE = 4  # dari 8
  ```

### Error: Training terlalu lambat
- Increase FRAME_SKIP di `config.py`:
  ```python
  FRAME_SKIP = 5  # dari 3
  ```

---

## 📊 Monitor Progress

**Terminal akan show:**
```
Epoch 1/100
21/21 [==============================] - 5s 245ms/step
     loss: 2.3421 - accuracy: 0.1765 
     val_loss: 2.1523 - val_accuracy: 0.2235
```

**Tanda-tanda baik:**
- ✅ Loss berkurang seiring epoch
- ✅ Accuracy meningkat
- ✅ Val_loss dan val_accuracy stabil/meningkat

**Tanda-tanda masalah:**
- ❌ Loss naik atau flat (tidak belajar)
- ❌ Program crash (out of memory)
- ❌ Val_loss meningkat drastis (overfitting)

---

## 🎯 Final Checklist

Sebelum jalankan:
- [ ] Python installed
- [ ] Folder dataset/ exist
- [ ] Video ada di subfolder (A/, B/, dll)
- [ ] Internet good (untuk download TensorFlow)
- [ ] Minimal 20GB free storage

Setelah selesai:
- [ ] `models/best_model.h5` exist
- [ ] `models/training_history.png` exist
- [ ] `models/confusion_matrix.png` exist
- [ ] Console show "TRAINING COMPLETE!"

---

## 🎓 Apa Yang Terjadi Di Setiap Step?

### Step 1: Cleansing
- Buka setiap video
- Check apakah corrupted
- Check apakah punya minimal frame
- Copy video baik ke `cleansed_dataset/`

### Step 2: Extract Landmarks
- Baca video dengan frame skipping
- Resize frame (lebih cepat)
- Deteksi tangan dengan MediaPipe
- Extract 126 features per frame (21 landmarks × 3 coord × 2 tangan)
- Save sebagai numpy array (.npy file)

### Step 3: Prepare Dataset
- Load semua .npy files
- Pad/truncate ke 40 frames
- Encode labels (A→0, B→1, dll)
- Convert ke one-hot format
- Split train/test 80/20

### Step 4: Train Model
- Load training data
- Build LSTM model (2 layers)
- Train dengan epoch sampai convergence
- Early stop jika val_loss plateau
- Save best model

---

## 💡 Tips & Tricks

### Untuk Accelerate:
```python
# Di config.py, increase ini:
FRAME_SKIP = 5          # Ambil lebih sedikit frame
SEQUENCE_LENGTH = 20    # Sequence lebih pendek
BATCH_SIZE = 4          # Batch lebih kecil (tapi lebih lambat per epoch)
```

### Untuk Better Accuracy:
```python
# Di config.py, decrease ini:
FRAME_SKIP = 1          # Ambil semua frame (lebih detail)
SEQUENCE_LENGTH = 60    # Sequence lebih panjang (lebih info)
BATCH_SIZE = 16         # Batch lebih besar (lebih stable)
```

### Untuk Monitor Training:
1. Buka Task Manager (Ctrl+Shift+Esc)
2. Lihat CPU % usage (target 70-90%)
3. Lihat Memory % usage (warning jika >80%)

---

## 📞 Jika Stuck

1. **Baca error message** di console
2. **Google error** atau cari di Stack Overflow
3. **Check folder structure** (cleansed_dataset/, extracted_landmarks/, dll)
4. **Try reduce batch size** atau frame skip
5. **Restart dari awal** (delete output folders dan try again)

---

## 🚀 Next: Using The Model

Setelah training selesai:

```python
# Load model
from tensorflow.keras.models import load_model
import joblib
import numpy as np

model = load_model('models/best_model.h5')
encoder = joblib.load('models/label_encoder.pkl')

# Predict (assuming X_new shape: (1, 40, 126))
prediction = model.predict(X_new)
predicted_class = np.argmax(prediction)
predicted_label = encoder.inverse_transform([predicted_class])

print(f"Predicted: {predicted_label[0]}")
```

---

## ⏱️ Estimated Times by Dataset Size

| Videos | Cleanse | Extract | Train | Total |
|--------|---------|---------|-------|-------|
| 100    | 5 min   | 45 min  | 30 min | 1.5h  |
| 500    | 25 min  | 225 min | 40 min | 5h    |
| 1000   | 50 min  | 450 min | 50 min | 10h   |

*Times berubah sesuai CPU speed. Laptop lambat bisa 2-3x lebih lama.*

---

**Version:** 1.0  
**Last Updated:** May 26, 2026  
**Status:** Ready for Production ✅
