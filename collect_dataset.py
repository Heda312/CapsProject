import cv2
import os
import time
import string

# ======================================
# CONFIG
# ======================================

DATASET_PATH = "dataset"

FRAME_WIDTH = 224
FRAME_HEIGHT = 224

FPS = 20
VIDEO_DURATION = 5

COUNTDOWN = 3
VIDEOS_PER_LABEL = 15

# ======================================
# CREATE LABEL FOLDERS
# ======================================

labels = list(string.ascii_uppercase)

for label in labels:
    os.makedirs(os.path.join(DATASET_PATH, label), exist_ok=True)

# ======================================
# OPEN CAMERA
# ======================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera tidak bisa dibuka.")
    exit()

print("\n=== SIGN LANGUAGE DATASET COLLECTOR ===")

# ======================================
# MAIN PROGRAM
# ======================================

while True:

    print("\n==============================")
    print(" PILIH LABEL DATASET ")
    print("==============================")
    print("A - Z")
    print("Ketik EXIT untuk keluar")
    print("==============================")

    selected_label = input("Masukkan label: ").upper()

    if selected_label == "EXIT":
        break

    if selected_label not in labels:
        print("Label tidak valid.")
        continue

    print(f"\n[INFO] Mengambil dataset label {selected_label}")

    vid_num = 1

    while vid_num <= VIDEOS_PER_LABEL:

        # ======================================
        # WAIT SCREEN
        # ======================================

        while True:

            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            cv2.putText(
                frame,
                f"Label : {selected_label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Video : {vid_num}/{VIDEOS_PER_LABEL}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "SPACE = Start Recording",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Q = Back to Menu",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 100, 255),
                2
            )

            cv2.imshow("Dataset Collector", frame)

            key = cv2.waitKey(1)

            # START RECORD
            if key == ord(' '):
                break

            # BACK MENU
            elif key == ord('q'):
                break

        if key == ord('q'):
            break

        # ======================================
        # COUNTDOWN
        # ======================================

        for count in range(COUNTDOWN, 0, -1):

            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            cv2.putText(
                frame,
                f"Recording in {count}",
                (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 255),
                4
            )

            cv2.imshow("Dataset Collector", frame)

            cv2.waitKey(1000)

        # ======================================
        # VIDEO SETUP
        # ======================================

        filename = f"{selected_label}_{vid_num:02d}.mp4"

        filepath = os.path.join(
            DATASET_PATH,
            selected_label,
            filename
        )

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(
            filepath,
            fourcc,
            FPS,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        print(f"[RECORDING] {filename}")

        start_time = time.time()

        canceled = False

        # ======================================
        # RECORD LOOP
        # ======================================

        while True:

            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            resized = cv2.resize(
                frame,
                (FRAME_WIDTH, FRAME_HEIGHT)
            )

            display = resized.copy()

            elapsed = round(time.time() - start_time, 1)

            cv2.putText(
                display,
                f"Label : {selected_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Video : {vid_num}/{VIDEOS_PER_LABEL}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Time : {elapsed}/{VIDEO_DURATION}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 100, 255),
                2
            )

            cv2.putText(
                display,
                "C = Cancel Take",
                (10, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            # SAVE FRAME
            out.write(resized)

            cv2.imshow("Dataset Collector", display)

            key = cv2.waitKey(1)

            # CANCEL TAKE
            if key == ord('c'):

                canceled = True
                print("[CANCELED TAKE]")
                break

            # AUTO STOP
            if elapsed >= VIDEO_DURATION:
                break

        out.release()

        # ======================================
        # HANDLE RESULT
        # ======================================

        if canceled:

            if os.path.exists(filepath):
                os.remove(filepath)

            print(f"[DELETED] {filepath}")
            print(f"[RETAKE] Video {vid_num}")

            # vid_num tidak bertambah

        else:

            print(f"[SAVED] {filepath}")

            # lanjut video berikutnya
            vid_num += 1

# ======================================
# CLEANUP
# ======================================

cap.release()
cv2.destroyAllWindows()

print("\n=== DATASET COLLECTION FINISHED ===")