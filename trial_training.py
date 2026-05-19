import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os

# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading Prepared Dataset...\n")

X_train = np.load("prepared_dataset/X_train.npy")
y_train = np.load("prepared_dataset/y_train.npy")

X_val = np.load("prepared_dataset/X_val.npy")
y_val = np.load("prepared_dataset/y_val.npy")

X_test = np.load("prepared_dataset/X_test.npy")
y_test = np.load("prepared_dataset/y_test.npy")

labels = np.load(
    "prepared_dataset/labels.npy",
    allow_pickle=True
).item()

print("Train Shape :", X_train.shape)
print("Val Shape   :", X_val.shape)
print("Test Shape  :", X_test.shape)

# ==========================================
# MODEL CONFIG
# ==========================================

NUM_CLASSES = y_train.shape[1]

# ==========================================
# BUILD LSTM MODEL
# ==========================================

print("\nBuilding LSTM Model...\n")

model = Sequential([

    LSTM(
        64,
        return_sequences=True,
        input_shape=(70, 126)
    ),

    Dropout(0.2),

    LSTM(128),

    Dropout(0.2),

    Dense(64, activation="relu"),

    Dense(NUM_CLASSES, activation="softmax")

])

# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# EARLY STOPPING
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# ==========================================
# TRAIN MODEL
# ==========================================

print("\nStarting Trial Training...\n")

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=50,

    batch_size=16,

    callbacks=[early_stopping],

    verbose=1
)

# ==========================================
# EVALUATE MODEL
# ==========================================

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(f"\nTest Accuracy : {accuracy:.4f}")
print(f"Test Loss     : {loss:.4f}")

# ==========================================
# PREDICTION REPORT
# ==========================================

y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred_classes
    )
)

# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

model.save("models/trial_lstm_model.h5")

print("\nModel Saved:")
print("models/trial_lstm_model.h5")

# ==========================================
# PLOT TRAINING
# ==========================================

plt.figure(figsize=(12, 5))

# accuracy

plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# loss

plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Train Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.show()

print("\n====================================")
print("TRIAL TRAINING COMPLETED")
print("====================================")