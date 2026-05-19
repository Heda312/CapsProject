import cv2
import os
import time
import string

# ======================================
# CONFIG
# ======================================

DATASET_PATH = "dataset"

FRAME_WIDTH = 512
FRAME_HEIGHT = 512

FPS = 30
VIDEO_DURATION = 5

COUNTDOWN = 3
VIDEOS_PER_LABEL = 15

# ======================================
# LABELS
# ======================================

alphabet_labels = list(string.ascii_uppercase)

number_labels = [
    "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9"
]

financial_labels = [
    "10",
    "20",
    "50",
    "100",
    "500",
    "1000",
    "RIBU",
    "JUTA",
    "MILYAR"
]

# ======================================
# CREATE CATEGORY FOLDERS
# ======================================

alphabet_path = os.path.join(
    DATASET_PATH,
    "alphabet"
)

number_path = os.path.join(
    DATASET_PATH,
    "number"
)

financial_path = os.path.join(
    DATASET_PATH,
    "financial"
)

custom_path = os.path.join(
    DATASET_PATH,
    "custom"
)

os.makedirs(alphabet_path, exist_ok=True)
os.makedirs(number_path, exist_ok=True)
os.makedirs(financial_path, exist_ok=True)
os.makedirs(custom_path, exist_ok=True)

# ======================================
# CREATE LABEL FOLDERS
# ======================================

for label in alphabet_labels:
    os.makedirs(
        os.path.join(alphabet_path, label),
        exist_ok=True
    )

for label in number_labels:
    os.makedirs(
        os.path.join(number_path, label),
        exist_ok=True
    )

for label in financial_labels:
    os.makedirs(
        os.path.join(financial_path, label),
        exist_ok=True
    )

# ======================================
# DELETE DATASET FUNCTION
# ======================================

def delete_dataset_file(selected_path, selected_label):

    files = sorted([
        f for f in os.listdir(selected_path)
        if f.endswith(".mp4")
    ])

    if len(files) == 0:

        print("\n[INFO] Tidak ada dataset.")
        return

    print("\n=== LIST DATASET ===")

    for file in files:
        print(file)

    try:

        delete_num = int(
            input(
                "\nMasukkan nomor dataset yang ingin dihapus: "
            )
        )

    except:
        print("Input harus angka.")
        return

    target_file = f"{selected_label}_{delete_num:02d}.mp4"

    target_path = os.path.join(
        selected_path,
        target_file
    )

    if not os.path.exists(target_path):

        print("[ERROR] File tidak ditemukan.")
        return

    # DELETE FILE
    os.remove(target_path)

    print(f"[DELETED] {target_file}")

    # ======================================
    # RENAME FILES
    # ======================================

    updated_files = sorted([
        f for f in os.listdir(selected_path)
        if f.endswith(".mp4")
    ])

    for idx, old_file in enumerate(updated_files, start=1):

        old_path = os.path.join(
            selected_path,
            old_file
        )

        new_file = f"{selected_label}_{idx:02d}.mp4"

        new_path = os.path.join(
            selected_path,
            new_file
        )

        os.rename(old_path, new_path)

    print("[INFO] Renumber selesai.")

# ======================================
# GET NEXT VIDEO NUMBER
# ======================================

def get_next_video_number(selected_path):

    existing_files = [
        f for f in os.listdir(selected_path)
        if f.endswith(".mp4")
    ]

    if len(existing_files) == 0:
        return 1

    existing_numbers = []

    for file in existing_files:

        try:

            number = int(
                file.split("_")[-1]
                .replace(".mp4", "")
            )

            existing_numbers.append(number)

        except:
            pass

    if len(existing_numbers) == 0:
        return 1

    return max(existing_numbers) + 1

# ======================================
# OPEN CAMERA
# ======================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera tidak bisa dibuka.")
    exit()

print("\n=== SIGN LANGUAGE DATASET COLLECTOR ===")

# ======================================
# MAIN LOOP
# ======================================

while True:

    print("\n===================================")
    print(" PILIH MODE DATASET ")
    print("===================================")
    print("1. Alphabet (A-Z)")
    print("2. Number (0-9)")
    print("3. Financial")
    print("4. Custom Label")
    print("5. Delete Dataset")
    print("6. Exit")
    print("===================================")

    mode = input("Pilih mode: ")

    # ======================================
    # EXIT
    # ======================================

    if mode == "6":
        break

    # ======================================
    # ALPHABET
    # ======================================

    elif mode == "1":

        print("\n=== ALPHABET LABEL ===")

        for label in alphabet_labels:
            print(label, end=" ")

        print()

        selected_label = input(
            "\nMasukkan label alphabet: "
        ).upper()

        if selected_label not in alphabet_labels:
            print("Label tidak valid.")
            continue

        selected_path = os.path.join(
            alphabet_path,
            selected_label
        )

    # ======================================
    # NUMBER
    # ======================================

    elif mode == "2":

        print("\n=== NUMBER LABEL ===")

        for label in number_labels:
            print(label, end=" ")

        print()

        selected_label = input(
            "\nMasukkan label angka: "
        ).upper()

        if selected_label not in number_labels:
            print("Label tidak valid.")
            continue

        selected_path = os.path.join(
            number_path,
            selected_label
        )

    # ======================================
    # FINANCIAL
    # ======================================

    elif mode == "3":

        print("\n=== FINANCIAL LABEL ===")

        for label in financial_labels:
            print(label, end=" ")

        print()

        selected_label = input(
            "\nMasukkan label financial: "
        ).upper()

        if selected_label not in financial_labels:
            print("Label tidak valid.")
            continue

        selected_path = os.path.join(
            financial_path,
            selected_label
        )

    # ======================================
    # CUSTOM LABEL
    # ======================================

    elif mode == "4":

        selected_label = input(
            "\nMasukkan custom label: "
        ).upper()

        if selected_label == "":
            print("Label kosong.")
            continue

        selected_path = os.path.join(
            custom_path,
            selected_label
        )

        os.makedirs(
            selected_path,
            exist_ok=True
        )

    # ======================================
    # DELETE DATASET
    # ======================================

    elif mode == "5":

        print("\n=== DELETE DATASET ===")
        print("1. Alphabet")
        print("2. Number")
        print("3. Financial")
        print("4. Custom")

        delete_mode = input(
            "\nPilih kategori: "
        )

        # ALPHABET
        if delete_mode == "1":

            selected_label = input(
                "Masukkan label alphabet: "
            ).upper()

            selected_path = os.path.join(
                alphabet_path,
                selected_label
            )

        # NUMBER
        elif delete_mode == "2":

            selected_label = input(
                "Masukkan label number: "
            ).upper()

            selected_path = os.path.join(
                number_path,
                selected_label
            )

        # FINANCIAL
        elif delete_mode == "3":

            selected_label = input(
                "Masukkan label financial: "
            ).upper()

            selected_path = os.path.join(
                financial_path,
                selected_label
            )

        # CUSTOM
        elif delete_mode == "4":

            selected_label = input(
                "Masukkan custom label: "
            ).upper()

            selected_path = os.path.join(
                custom_path,
                selected_label
            )

        else:

            print("Kategori tidak valid.")
            continue

        if not os.path.exists(selected_path):

            print("[ERROR] Folder tidak ditemukan.")
            continue

        delete_dataset_file(
            selected_path,
            selected_label
        )

        continue

    else:

        print("Pilihan tidak valid.")
        continue

    # ======================================
    # AUTO CONTINUE VIDEO NUMBER
    # ======================================

    vid_num = get_next_video_number(
        selected_path
    )

    print(f"\n[INFO] Mengambil dataset: {selected_label}")
    print(f"[INFO] Mulai dari video ke-{vid_num}")

    # ======================================
    # START COLLECTION
    # ======================================

    while True:

        # ======================================
        # LIMIT CHECK
        # ======================================

        if vid_num > VIDEOS_PER_LABEL:

            print("\n[INFO] Dataset sudah penuh.")
            break

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

            cv2.imshow(
                "Dataset Collector",
                frame
            )

            key = cv2.waitKey(1)

            # START
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

            cv2.imshow(
                "Dataset Collector",
                frame
            )

            cv2.waitKey(1000)

        # ======================================
        # VIDEO SETUP
        # ======================================

        filename = f"{selected_label}_{vid_num:02d}.mp4"

        filepath = os.path.join(
            selected_path,
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

            elapsed = round(
                time.time() - start_time,
                1
            )

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

            cv2.imshow(
                "Dataset Collector",
                display
            )

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

        else:

            print(f"[SAVED] {filepath}")

            # refresh nomor terbaru
            vid_num = get_next_video_number(
                selected_path
            )

# ======================================
# CLEANUP
# ======================================

cap.release()
cv2.destroyAllWindows()

print("\n=== DATASET COLLECTION FINISHED ===")