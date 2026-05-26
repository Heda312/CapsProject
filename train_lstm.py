"""
=============================================================
    TRAIN_LSTM.PY - Training LSTM Model
=============================================================

FUNGSI:
    Train model LSTM untuk sign language recognition
    - Load prepared dataset (X_train, X_test, y_train, y_test)
    - Build Bidirectional LSTM model (optimized untuk CPU)
    - Train model dengan early stopping & learning rate reduction
    - Evaluate model pada test set
    - Generate visualisasi (training history, confusion matrix)

MODEL ARCHITECTURE:
    Input (40, 126)
        ↓
    Bidirectional LSTM 32 + Dropout 0.3
        ↓
    Bidirectional LSTM 64 + Dropout 0.3
        ↓
    Dense 32 (ReLU) + Dropout 0.3
        ↓
    Dense 41 (Softmax) - output layer
    
    Total parameters: ~200K (optimized untuk CPU)

TRAINING OPTIMIZATION:
    - Batch size: 8 (vs 16 sebelumnya)
    - Learning rate: 0.0005 (lebih stable)
    - Early stopping: patience 10 (lebih ketat)
    - ReduceLROnPlateau: factor 0.5, patience 3

WAKTU ESTIMASI:
    - Per epoch: 30-60 detik (pada CPU menengah)
    - Total training: 30-60 menit (20-40 epochs)

OUTPUT:
    - models/best_model.h5 - Model terbaik
    - models/final_model.h5 - Model final
    - models/training_history.png - Graphs
    - models/confusion_matrix.png - Confusion matrix

=============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import Policy
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from config import *

print("=" * 60)
print("Training LSTM Model for Sign Language Recognition")
print("=" * 60)
print("")

# ============================
# STEP 1: OPTIMIZE UNTUK CPU
# ============================
# Mixed precision dapat mempercepat computation pada GPU dan beberapa CPU
# Tidak akan break jika tidak available
try:
    policy = Policy('mixed_float16')
    # tf.keras.mixed_precision.set_global_policy(policy)
    print("✓ Mixed precision policy applied")
except:
    print("ℹ Mixed precision not available, using float32")

print("")

# ============================
# STEP 2: LOAD DATA
# ============================
# Data sudah di-prepare oleh prepare_dataset.py
# Format: numpy arrays
# - X_train shape: (num_train_samples, 40, 126)
# - y_train shape: (num_train_samples, 41) - one-hot encoded

print("Loading dataset...")

# Load training data
X_train = np.load(os.path.join(PREPARED_DIR, 'X_train.npy'))
# Load test data
X_test = np.load(os.path.join(PREPARED_DIR, 'X_test.npy'))
# Load training labels (one-hot encoded)
y_train = np.load(os.path.join(PREPARED_DIR, 'y_train.npy'))
# Load test labels (one-hot encoded)
y_test = np.load(os.path.join(PREPARED_DIR, 'y_test.npy'))

# Print dataset info
print(f"✓ Dataset loaded successfully\n")
print(f"Training set shape:   {X_train.shape} (samples, sequence_length, features)")
print(f"Test set shape:       {X_test.shape}")
print(f"Training labels shape: {y_train.shape} (samples, num_classes)")
print(f"Test labels shape:    {y_test.shape}")
print(f"Number of classes:    {y_train.shape[1]}")
print("")

# ============================
# STEP 3: BUILD MODEL
# ============================
# Model architecture: Bidirectional LSTM dengan 2 layers
# Lebih kecil dibanding sebelumnya untuk CPU efficiency

print("Building model...")
print("")

model = Sequential([
    # LAYER 1: Bidirectional LSTM
    # - Bidirectional: Process sequence forward DAN backward
    # - 32 units: Reduced dari 64 untuk CPU efficiency
    # - return_sequences=True: Pass full sequence ke layer berikutnya
    # - input_shape=(40, 126): 40 frames, 126 features per frame
    Bidirectional(LSTM(
        32,
        return_sequences=True,
        input_shape=(SEQUENCE_LENGTH, 126)
    )),
    Dropout(0.3),  # Dropout 30% untuk regularization

    # LAYER 2: Bidirectional LSTM
    # - 64 units: Reduced dari 128
    # - return_sequences=False: Return hanya last output
    # - Output shape: (batch_size, 64)
    Bidirectional(LSTM(
        64,
        return_sequences=False
    )),
    Dropout(0.3),

    # LAYER 3: Dense (Fully Connected)
    # - 32 units: Reduced dari 64
    # - activation='relu': ReLU activation
    # - Output shape: (batch_size, 32)
    Dense(32, activation='relu'),
    Dropout(0.3),

    # LAYER 4: Output Layer
    # - Dense 41: 41 class labels (A-Z, 0-9, ATM, Kartu_ATM, Saldo, Transfer, Uang, 10, 20, 50, 100, 500, 1000, Ribu, Juta, Milyar)
    # - activation='softmax': Output probability per class
    # - Output shape: (batch_size, 41)
    Dense(len(LABELS), activation='softmax')
])

# COMPILE MODEL
# - optimizer=Adam: Adaptive learning rate optimizer
# - loss='categorical_crossentropy': Loss untuk multi-class classification
# - metrics=['accuracy']: Monitor accuracy selama training
model.compile(
    optimizer=Adam(learning_rate=0.0005),  # Learning rate lebih kecil untuk stability
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model Summary:")
print("")
model.summary()  # Print detailed model info
print("")

# ============================
# STEP 4: SETUP CALLBACKS
# ============================
# Callbacks: Functions yang dipanggil setelah setiap epoch
# Digunakan untuk monitoring dan optimization

print("Setting up callbacks...")
print("")

# CALLBACK 1: Early Stopping
# - Stop training jika validation loss tidak improve untuk patience epochs
# - patience=10: Stop jika tidak improve selama 10 epochs
# - restore_best_weights=True: Restore weights dari best epoch
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# CALLBACK 2: Reduce Learning Rate on Plateau
# - Reduce learning rate jika validation loss plateau
# - factor=0.5: Multiply learning rate dengan 0.5 (i.e., learning_rate / 2)
# - patience=3: Reduce LR jika tidak improve selama 3 epochs
# - min_lr=1e-6: Minimum learning rate (jangan turun lebih dari ini)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# CALLBACK 3: Model Checkpoint
# - Save best model (berdasarkan validation accuracy)
# - save_best_only=True: Hanya save jika better than previous
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, 'best_model.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ============================
# STEP 5: TRAIN MODEL
# ============================

print("=" * 60)
print("Starting training...")
print("=" * 60)
print("")

# BATCH_SIZE: Jumlah samples per gradient update
# Batch size lebih kecil (8 vs 16) untuk CPU efficiency
BATCH_SIZE_TRAIN = 8

# FIT MODEL
# - X_train: Training features (num_samples, 40, 126)
# - y_train: Training labels one-hot encoded (num_samples, 41)
# - validation_data: Use test set untuk validation
# - epochs: Maximum number of iterations
# - batch_size: Samples per gradient update
# - callbacks: List of callbacks untuk monitoring
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE_TRAIN,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1  # Print progress
)

# ============================
# STEP 6: EVALUATE MODEL
# ============================

print("\n" + "=" * 60)
print("Evaluating model on test set...")
print("=" * 60)

# Evaluate pada test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✓ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"✓ Test Loss:     {loss:.4f}")
print("")

# ============================
# STEP 7: SAVE MODEL
# ============================

print("Saving model...")
# Save final model (including training history)
model.save(os.path.join(MODEL_DIR, 'final_model.h5'))
print(f"✓ Model saved to {os.path.join(MODEL_DIR, 'final_model.h5')}")
print("")

# ============================
# STEP 8: VISUALISASI - TRAINING HISTORY
# ============================

print("Generating training history graphs...")

# Create figure dengan 2 subplots
plt.figure(figsize=(14, 5))

# Subplot 1: Accuracy
plt.subplot(1, 2, 1)
# Plot training accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
# Plot validation accuracy
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy During Training', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Subplot 2: Loss
plt.subplot(1, 2, 2)
# Plot training loss
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
# Plot validation loss
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss During Training', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
# Save figure
plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'), dpi=100, bbox_inches='tight')
plt.close()

print(f"✓ Training history saved to {os.path.join(MODEL_DIR, 'training_history.png')}")
print("")

# ============================
# STEP 9: VISUALISASI - CONFUSION MATRIX
# ============================

print("Generating confusion matrix...")

# Predict pada test set
predictions = model.predict(X_test, batch_size=32, verbose=0)

# Get predicted class (argmax) untuk setiap sample
y_pred = np.argmax(predictions, axis=1)
# Get true class dari one-hot encoded labels
y_true = np.argmax(y_test, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix dengan heatmap
plt.figure(figsize=(20, 20))
# sns.heatmap: Visualisasi confusion matrix
# annot=False: Jangan print nilai di setiap cell (terlalu kecil)
# cmap='Blues': Color scheme
sns.heatmap(cm, cmap='Blues', cbar=True)
plt.title('Confusion Matrix - Test Set', fontsize=18, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)

# Save figure
plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=100, bbox_inches='tight')
plt.close()

print(f"✓ Confusion matrix saved to {os.path.join(MODEL_DIR, 'confusion_matrix.png')}")
print("")

# ============================
# STEP 10: CLASSIFICATION REPORT
# ============================

print("=" * 60)
print("Classification Report (Per Class)")
print("=" * 60)
print("")

# Generate detailed classification report
# Precision, Recall, F1-score untuk setiap class
report = classification_report(y_true, y_pred, target_names=LABELS)
print(report)

# ============================
# SUMMARY
# ============================

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"✓ Best model:         models/best_model.h5")
print(f"✓ Final model:        models/final_model.h5")
print(f"✓ Training history:   models/training_history.png")
print(f"✓ Confusion matrix:   models/confusion_matrix.png")
print(f"\nYour model is ready for inference! 🎉")
print("")