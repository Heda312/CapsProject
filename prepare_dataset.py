import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# ==========================================
# CONFIG
# ==========================================

DATASET_DIR = "dataset_landmarks"

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

# ==========================================
# LOAD DATASET
# ==========================================

X = []
y = []

label_map = {}
current_label = 0

print("\nLoading Dataset...\n")

for category in os.listdir(DATASET_DIR):

    category_path = os.path.join(DATASET_DIR, category)

    if not os.path.isdir(category_path):
        continue

    for label in os.listdir(category_path):

        label_path = os.path.join(category_path, label)

        if not os.path.isdir(label_path):
            continue

        print(f"Processing Label: {label}")

        label_map[label] = current_label

        npy_files = [
            f for f in os.listdir(label_path)
            if f.endswith(".npy")
        ]

        for npy_file in npy_files:

            npy_path = os.path.join(
                label_path,
                npy_file
            )

            sequence = np.load(npy_path)

            X.append(sequence)
            y.append(current_label)

        current_label += 1

X = np.array(X)
y = np.array(y)

print(X.shape)
print(y.shape)

# ==========================================
# ONE HOT
# ==========================================

y = to_categorical(y)

# ==========================================
# SPLIT
# ==========================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=(TEST_SIZE + VALIDATION_SIZE),
    random_state=42,
    shuffle=True
)

val_ratio = VALIDATION_SIZE / (
    TEST_SIZE + VALIDATION_SIZE
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=val_ratio,
    random_state=42,
    shuffle=True
)

# ==========================================
# SAVE
# ==========================================

os.makedirs("prepared_dataset", exist_ok=True)

np.save("prepared_dataset/X_train.npy", X_train)
np.save("prepared_dataset/y_train.npy", y_train)

np.save("prepared_dataset/X_val.npy", X_val)
np.save("prepared_dataset/y_val.npy", y_val)

np.save("prepared_dataset/X_test.npy", X_test)
np.save("prepared_dataset/y_test.npy", y_test)

np.save(
    "prepared_dataset/labels.npy",
    label_map
)

print("\nDONE.")