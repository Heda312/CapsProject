import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# ==========================================
# CONFIG
# ==========================================

DATASET_DIR = "dataset_clean"
OUTPUT_DIR = "dataset_landmarks"

SEQUENCE_LENGTH = 70

ENABLE_MIRROR_AUGMENTATION = True

# ==========================================
# MEDIAPIPE
# ==========================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# CREATE OUTPUT
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# NORMALIZATION
# ==========================================

def normalize_hand(hand_data):

    hand_data = np.array(hand_data)

    # wrist landmark
    wrist_x = hand_data[0]
    wrist_y = hand_data[1]
    wrist_z = hand_data[2]

    normalized = []

    for i in range(0, len(hand_data), 3):

        normalized.extend([
            hand_data[i] - wrist_x,
            hand_data[i+1] - wrist_y,
            hand_data[i+2] - wrist_z
        ])

    return normalized

# ==========================================
# MIRROR AUGMENTATION
# ==========================================

def mirror_landmarks(landmarks):

    mirrored = landmarks.copy()

    for i in range(0, len(mirrored), 3):

        # mirror x coordinate
        mirrored[i] = -mirrored[i]

    return mirrored

# ==========================================
# EXTRACT LANDMARKS
# ==========================================

def extract_landmarks(frame):

    image_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(image_rgb)

    all_landmarks = []

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            hand_data = []

            for lm in hand_landmarks.landmark:

                hand_data.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            # NORMALIZATION
            hand_data = normalize_hand(hand_data)

            all_landmarks.extend(hand_data)

    # padding
    while len(all_landmarks) < 126:
        all_landmarks.append(0.0)

    return np.array(all_landmarks[:126])

# ==========================================
# SAMPLE FRAMES
# ==========================================

def sample_frames(total_frames, seq_length):

    if total_frames < seq_length:

        indices = list(range(total_frames))

        while len(indices) < seq_length:
            indices.append(indices[-1])

        return indices

    return np.linspace(
        0,
        total_frames - 1,
        seq_length,
        dtype=int
    )

# ==========================================
# PROCESS
# ==========================================

print("\nStarting Landmark Extraction...\n")

for category in os.listdir(DATASET_DIR):

    category_path = os.path.join(
        DATASET_DIR,
        category
    )

    if not os.path.isdir(category_path):
        continue

    for label in os.listdir(category_path):

        label_path = os.path.join(
            category_path,
            label
        )

        if not os.path.isdir(label_path):
            continue

        print(f"\nProcessing: {category}/{label}")

        output_label_path = os.path.join(
            OUTPUT_DIR,
            category,
            label
        )

        os.makedirs(
            output_label_path,
            exist_ok=True
        )

        videos = [
            f for f in os.listdir(label_path)
            if f.endswith(".mp4")
        ]

        for video_file in tqdm(videos):

            video_path = os.path.join(
                label_path,
                video_file
            )

            cap = cv2.VideoCapture(video_path)

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            frame_indices = sample_frames(
                total_frames,
                SEQUENCE_LENGTH
            )

            sequence_data = []

            current_frame = 0
            target_index = 0

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                if current_frame == frame_indices[target_index]:

                    frame = cv2.resize(
                        frame,
                        (512, 512)
                    )

                    landmarks = extract_landmarks(frame)

                    sequence_data.append(landmarks)

                    target_index += 1

                    if target_index >= len(frame_indices):
                        break

                current_frame += 1

            cap.release()

            sequence_array = np.array(sequence_data)

            if sequence_array.shape != (SEQUENCE_LENGTH, 126):

                print(f"Rejected: {video_file}")
                continue

            # ==================================
            # SAVE ORIGINAL
            # ==================================

            output_filename = (
                os.path.splitext(video_file)[0]
                + ".npy"
            )

            output_path = os.path.join(
                output_label_path,
                output_filename
            )

            np.save(output_path, sequence_array)

            # ==================================
            # SAVE MIRROR
            # ==================================

            if ENABLE_MIRROR_AUGMENTATION:

                mirrored_sequence = np.array([

                    mirror_landmarks(frame)

                    for frame in sequence_array

                ])

                mirror_filename = (
                    os.path.splitext(video_file)[0]
                    + "_mirror.npy"
                )

                mirror_path = os.path.join(
                    output_label_path,
                    mirror_filename
                )

                np.save(
                    mirror_path,
                    mirrored_sequence
                )

print("\nDONE.")