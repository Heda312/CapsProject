import cv2
import numpy as np
from collections import deque, Counter

from keras.models import load_model

import mediapipe as mp

# ==========================================
# CONFIG
# ==========================================

SEQUENCE_LENGTH = 70

CONFIDENCE_THRESHOLD = 0.65

SMOOTHING_WINDOW = 5

MODEL_PATH = "models/final_lstm_model.keras"

LABELS_PATH = "prepared_dataset/labels.npy"

# ==========================================
# LOAD MODEL
# ==========================================

model = load_model(MODEL_PATH)

label_map = np.load(
    LABELS_PATH,
    allow_pickle=True
).item()

reverse_label_map = {
    v: k for k, v in label_map.items()
}

# ==========================================
# MEDIAPIPE
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=2,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5
)

# ==========================================
# BUFFER
# ==========================================

sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

prediction_buffer = deque(
    maxlen=SMOOTHING_WINDOW
)

# ==========================================
# NORMALIZATION
# ==========================================

def normalize_hand(hand_data):

    hand_data = np.array(hand_data)

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
# LANDMARK EXTRACTION
# ==========================================

def extract_landmarks(frame):

    image_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(image_rgb)

    all_landmarks = []

    hand_detected = False

    if results.multi_hand_landmarks:

        hand_detected = True

        for hand_landmarks in results.multi_hand_landmarks:

            hand_data = []

            for lm in hand_landmarks.landmark:

                hand_data.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            hand_data = normalize_hand(hand_data)

            all_landmarks.extend(hand_data)

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    while len(all_landmarks) < 126:
        all_landmarks.append(0.0)

    return np.array(all_landmarks[:126]), hand_detected

# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

stable_prediction = "NO_SIGN"

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    landmarks, hand_detected = extract_landmarks(frame)

    # ======================================
    # NO HAND
    # ======================================

    if not hand_detected:

        stable_prediction = "NO_SIGN"

        cv2.putText(

            frame,

            stable_prediction,

            (20, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0,0,255),

            2
        )

        cv2.imshow(
            "BISINDO Realtime",
            frame
        )

        if cv2.waitKey(1) & 0xFF == 27:
            break

        continue

    # ======================================
    # BUFFER
    # ======================================

    sequence_buffer.append(landmarks)

    if len(sequence_buffer) == SEQUENCE_LENGTH:

        input_data = np.expand_dims(
            np.array(sequence_buffer),
            axis=0
        )

        prediction = model.predict(
            input_data,
            verbose=0
        )[0]

        predicted_index = np.argmax(prediction)

        confidence = prediction[predicted_index]

        predicted_label = reverse_label_map[
            predicted_index
        ]

        # ==================================
        # CONFIDENCE THRESHOLD
        # ==================================

        if confidence >= CONFIDENCE_THRESHOLD:

            prediction_buffer.append(
                predicted_label
            )

            # ==============================
            # SMOOTHING
            # ==============================

            most_common = Counter(
                prediction_buffer
            ).most_common(1)

            stable_prediction = most_common[0][0]

        else:

            stable_prediction = "NO_SIGN"

        # ==================================
        # DISPLAY
        # ==================================

        cv2.putText(

            frame,

            f"Prediction: {stable_prediction}",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0,255,0),

            2
        )

        cv2.putText(

            frame,

            f"Confidence: {confidence:.2f}",

            (20, 80),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0,255,255),

            2
        )

    cv2.imshow(
        "BISINDO Realtime",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()