"""
=============================================================
    PREPARE_DATASET.PY - Persiapan Dataset
=============================================================

FUNGSI:
    Persiapan data untuk training LSTM model
    - Load landmarks dari extracted_landmarks/
    - Pad/truncate sequences ke SEQUENCE_LENGTH (40 frames)
    - Encode labels (string → numeric)
    - Convert ke one-hot encoding
    - Split train/test (80/20)
    - Save dalam format numpy array

INPUT:
    extracted_landmarks/ folder dengan .npy files
    Format: (num_frames, 126)

PROCESSING:
    1. Load setiap .npy file
    2. Pad/truncate ke 40 frames
    3. Stack sebagai dataset (num_samples, 40, 126)
    4. Encode labels dengan LabelEncoder
    5. Convert ke one-hot encoding
    6. Split 80/20 train/test dengan stratification
    7. Save 4 files (.npy)

OUTPUT:
    prepared_dataset/
    ├── X_train.npy (3200, 40, 126)
    ├── X_test.npy (800, 40, 126)
    ├── y_train.npy (3200, 41)
    └── y_test.npy (800, 41)

    models/
    └── label_encoder.pkl (untuk inference nanti)

=============================================================
"""

import os
import joblib
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import *

print("=" * 60)
print("Preparing Dataset for Training")
print("=" * 60)
print("")


def pad_sequence(sequence):
    """
    Pad atau truncate sequence ke SEQUENCE_LENGTH yang fixed
    
    PENJELASAN:
        - Semua video bisa punya durasi berbeda
        - LSTM butuh input dengan shape yang sama
        - Solution: Pad short sequences dengan zeros
        - Truncate long sequences
    
    LOGIC:
        if len(sequence) >= 40:
            return sequence[:40]  # Ambil 40 frames pertama
        else:
            return pad dengan zeros  # Tambah zeros di belakang
    
    INPUT:
        sequence: numpy array dengan shape (num_frames, 126)
    
    OUTPUT:
        numpy array dengan shape (40, 126)
    """
    # Jika sequence lebih panjang atau sama dengan SEQUENCE_LENGTH
    if len(sequence) >= SEQUENCE_LENGTH:
        # Truncate: ambil SEQUENCE_LENGTH frame pertama
        # (discard frame yg lebih dari ini)
        return sequence[:SEQUENCE_LENGTH]

    # Jika sequence lebih pendek dari SEQUENCE_LENGTH
    # Buat padding array berisi zeros
    # Shape: (40 - current_length, 126)
    padding = np.zeros((
        SEQUENCE_LENGTH - len(sequence),
        126
    ), dtype=np.float32)

    # Stack padding di belakang sequence
    # Result: (current_length + padding_length, 126) = (40, 126)
    return np.vstack([sequence, padding]).astype(np.float32)


# ============================
# STEP 1: LOAD LANDMARKS
# ============================

print("Loading landmarks...")
print("")

# Initialize lists untuk features dan labels
X = []  # Features: array of sequences
y = []  # Labels: class label for setiap sequence
count = 0

# Iterate setiap label folder
for label in sorted(os.listdir(LANDMARK_DIR)):

    label_path = os.path.join(LANDMARK_DIR, label)

    # Skip jika bukan folder
    if not os.path.isdir(label_path):
        continue

    print(f"Loading: {label}")

    # Iterate setiap file .npy dalam label folder
    for file_name in sorted(os.listdir(label_path)):

        # Skip jika bukan .npy file
        if not file_name.endswith('.npy'):
            continue

        file_path = os.path.join(label_path, file_name)

        try:
            # Load numpy array dari file
            # Shape: (num_frames, 126)
            sequence = np.load(file_path, allow_pickle=False)
            
            # Ensure float32 untuk memory efficiency
            # float32 lebih hemat memory daripada float64
            sequence = sequence.astype(np.float32)
            
            # Pad atau truncate ke SEQUENCE_LENGTH (40 frames)
            sequence = pad_sequence(sequence)

            # Add ke dataset
            X.append(sequence)  # Add sequence features
            y.append(label)     # Add corresponding label
            count += 1
            
            # Print progress setiap 10 files
            if count % 10 == 0:
                print(f"  Loaded {count} files...")
                
        except Exception as e:
            # Jika ada error loading file, skip dan continue
            print(f"  ⚠ Error loading {file_path}: {e}")
            continue

# Convert lists ke numpy arrays
# X shape: (num_samples, 40, 126)
# y shape: (num_samples,)
X = np.array(X, dtype=np.float32)
y = np.array(y)

# ============================
# STEP 2: PRINT DATASET INFO
# ============================

print(f"\n{'='*60}")
print(f"=== Dataset Statistics ===")
print(f"{'='*60}")
print(f"X shape:           {X.shape} (samples, sequence_length, features)")
print(f"y shape:           {y.shape} (samples,)")
print(f"Total samples:     {len(X)}")
print(f"Total classes:     {len(np.unique(y))}")
print(f"Memory usage:      {X.nbytes / (1024**2):.2f} MB")

# Count samples per class
unique_labels, counts = np.unique(y, return_counts=True)
print(f"\nSamples per class:")
for label, count in zip(unique_labels, counts):
    print(f"  {label:15} : {count:4d} samples")
print("")

# ============================
# STEP 3: ENCODE LABELS
# ============================

print("Encoding labels...")

# LabelEncoder: Convert string labels → numeric (0, 1, 2, ...)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print(f"Classes: {encoder.classes_}")
print(f"Encoding: {dict(zip(encoder.classes_, range(len(encoder.classes_))))}")

# Convert ke one-hot encoding
# Contoh: class 0 → [1, 0, 0, ..., 0]
#         class 1 → [0, 1, 0, ..., 0]
#         class 41 → [0, 0, 0, ..., 1]
y_categorical = to_categorical(y_encoded)

print(f"One-hot shape: {y_categorical.shape}")
print("")

# ============================
# STEP 4: SAVE ENCODER
# ============================

print("Saving label encoder...")

# Save encoder untuk inference nanti
# Di inference: model predict numeric → decode dengan encoder
joblib.dump(
    encoder,
    os.path.join(MODEL_DIR, 'label_encoder.pkl')
)
print(f"✓ Encoder saved to {os.path.join(MODEL_DIR, 'label_encoder.pkl')}")
print("")

# ============================
# STEP 5: SPLIT TRAIN/TEST
# ============================

print("Splitting dataset...")

# Train/Test split: 80% train, 20% test
# stratify=y_encoded: Ensure class distribution sama di train dan test
# Contoh: jika original 10% class A, train juga 10% class A
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,      # 20% untuk test
    random_state=42,    # Fixed random seed untuk reproducibility
    stratify=y_encoded  # Stratified split (maintain class distribution)
)

print(f"Train set: {X_train.shape}")
print(f"Test set:  {X_test.shape}")
print("")

# ============================
# STEP 6: SAVE PREPARED DATA
# ============================

print("Saving prepared data...")

# Save training features
np.save(os.path.join(PREPARED_DIR, 'X_train.npy'), X_train)
# Save test features
np.save(os.path.join(PREPARED_DIR, 'X_test.npy'), X_test)
# Save training labels (one-hot encoded)
np.save(os.path.join(PREPARED_DIR, 'y_train.npy'), y_train)
# Save test labels (one-hot encoded)
np.save(os.path.join(PREPARED_DIR, 'y_test.npy'), y_test)

print(f"✓ Training features:  {os.path.join(PREPARED_DIR, 'X_train.npy')}")
print(f"✓ Test features:      {os.path.join(PREPARED_DIR, 'X_test.npy')}")
print(f"✓ Training labels:    {os.path.join(PREPARED_DIR, 'y_train.npy')}")
print(f"✓ Test labels:        {os.path.join(PREPARED_DIR, 'y_test.npy')}")
print("")

# ============================
# SUMMARY
# ============================

print("=" * 60)
print("PREPARATION COMPLETE!")
print("=" * 60)
print("✓ Dataset prepared and saved successfully!")
print("✓ Ready for LSTM training (train_lstm.py)")
print("=" * 60)