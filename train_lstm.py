import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)

from keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# ==========================================
# LOAD DATASET
# ==========================================

X_train = np.load("prepared_dataset/X_train.npy")
y_train = np.load("prepared_dataset/y_train.npy")

X_val = np.load("prepared_dataset/X_val.npy")
y_val = np.load("prepared_dataset/y_val.npy")

X_test = np.load("prepared_dataset/X_test.npy")
y_test = np.load("prepared_dataset/y_test.npy")

# ==========================================
# CONFIG
# ==========================================

INPUT_SHAPE = (70, 126)

NUM_CLASSES = y_train.shape[1]

# ==========================================
# MODEL
# ==========================================

model = Sequential([

    LSTM(
        128,
        return_sequences=True,
        input_shape=INPUT_SHAPE
    ),

    BatchNormalization(),

    Dropout(0.3),

    LSTM(256),

    BatchNormalization(),

    Dropout(0.3),

    Dense(128, activation="relu"),

    Dropout(0.3),

    Dense(NUM_CLASSES, activation="softmax")

])

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

model.summary()

# ==========================================
# CALLBACKS
# ==========================================

os.makedirs("models", exist_ok=True)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "models/best_lstm_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ==========================================
# TRAINING
# ==========================================

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=100,

    batch_size=16,

    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ]
)

# ==========================================
# SAVE FINAL MODEL
# ==========================================

model.save("models/final_lstm_model.keras")

# ==========================================
# EVALUATION
# ==========================================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nTest Accuracy:", accuracy)

# ==========================================
# PLOT
# ==========================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Accuracy")

plt.legend([
    "Train",
    "Validation"
])

plt.subplot(1,2,2)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Loss")

plt.legend([
    "Train",
    "Validation"
])

plt.tight_layout()

plt.show()