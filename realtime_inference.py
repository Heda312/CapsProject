import cv2
import joblib
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model
from config import *


model = load_model('models/best_model.h5')
encoder = joblib.load('models/label_encoder.pkl')


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

sentence = []
last_prediction = ""
cooldown = 0

def extract_landmarks(results):
    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    if results.multi_hand_landmarks and results.multi_handedness:

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            label = handedness.classification[0].label

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks)
            
            if label == 'Left':
                left_hand = landmarks
            else:
                right_hand = landmarks

    return np.concatenate([left_hand, right_hand])

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    landmarks = extract_landmarks(results)

    sequence_buffer.append(landmarks)

    if len(sequence_buffer) == SEQUENCE_LENGTH:

        input_data = np.expand_dims(sequence_buffer, axis=0)

        prediction = model.predict(input_data, verbose=0)[0]

        confidence = np.max(prediction)
        predicted_index = np.argmax(prediction)

        predicted_label = encoder.inverse_transform(
            [predicted_index]
        )[0]

        prediction_buffer.append(predicted_label)

        stable_prediction = max(
            set(prediction_buffer),
            key=prediction_buffer.count
        )

        if confidence > PREDICTION_THRESHOLD:

            if cooldown == 0:

                if stable_prediction != last_prediction:

                    sentence.append(stable_prediction)
                    last_prediction = stable_prediction
                    cooldown = COOLDOWN_FRAMES

        if cooldown > 0:
            cooldown -= 1

        cv2.putText(
            frame,
            f'Prediction: {stable_prediction}',
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f'Confidence: {confidence:.2f}',
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )
    sentence_text = ' '.join(sentence)

    cv2.putText(
        frame,
        sentence_text,
        (10, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow('Realtime Sign Language', frame)

    key = cv2.waitKey(1)

    if key == ord('c'):
        sentence = []

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()