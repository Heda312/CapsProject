import os
import cv2
import shutil
import mediapipe as mp
from tqdm import tqdm

# =========================================================
# CONFIGURATION
# =========================================================

DATASET_DIR = "dataset_raw"

CLEAN_DATASET_DIR = "dataset_clean"
REJECTED_DATASET_DIR = "dataset_rejected"

VALID_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

# Dataset Rules
MIN_FRAMES = 60
MAX_FRAMES = 300

MIN_BRIGHTNESS = 40

MIN_HAND_DETECTION_RATIO = 0.5

MIN_RESOLUTION_WIDTH = 320
MIN_RESOLUTION_HEIGHT = 240

# =========================================================
# MEDIAPIPE INITIALIZATION
# =========================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================================================
# CREATE OUTPUT FOLDERS
# =========================================================

os.makedirs(CLEAN_DATASET_DIR, exist_ok=True)
os.makedirs(REJECTED_DATASET_DIR, exist_ok=True)

# =========================================================
# VALIDATION FUNCTION
# =========================================================

def validate_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False, "Cannot Open Video"

    total_frames = 0
    hand_detected_frames = 0
    brightness_values = []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resolution Validation
    if width < MIN_RESOLUTION_WIDTH or height < MIN_RESOLUTION_HEIGHT:
        cap.release()
        return False, "Low Resolution"

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        total_frames += 1

        # Brightness Check
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        brightness_values.append(brightness)

        # Hand Detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_detected_frames += 1

    cap.release()

    # Frame Validation
    if total_frames < MIN_FRAMES:
        return False, "Too Few Frames"

    if total_frames > MAX_FRAMES:
        return False, "Too Many Frames"

    # Brightness Validation
    avg_brightness = sum(brightness_values) / len(brightness_values)

    if avg_brightness < MIN_BRIGHTNESS:
        return False, "Too Dark"

    # Hand Detection Validation
    detection_ratio = hand_detected_frames / total_frames

    if detection_ratio < MIN_HAND_DETECTION_RATIO:
        return False, "Hand Not Detected Properly"

    return True, "Valid"

# =========================================================
# COPY FUNCTION WITH FOLDER STRUCTURE
# =========================================================

def copy_video_with_structure(src_path, base_output_dir):

    # Relative path from dataset_raw
    relative_path = os.path.relpath(src_path, DATASET_DIR)

    # Full destination path
    dst_path = os.path.join(base_output_dir, relative_path)

    # Create folders automatically
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # Copy video
    shutil.copy2(src_path, dst_path)

# =========================================================
# SUMMARY
# =========================================================

summary = {
    "valid": 0,
    "rejected": 0
}

# =========================================================
# MAIN PROCESS
# =========================================================

print("\nStarting Dataset Validation & Cleansing...\n")

# Read category folders
for category in os.listdir(DATASET_DIR):

    category_path = os.path.join(DATASET_DIR, category)

    if not os.path.isdir(category_path):
        continue

    print(f"\nProcessing Category: {category}")

    # Recursive video search
    video_files = []

    for root, dirs, files in os.walk(category_path):

        for file in files:

            if os.path.splitext(file)[1].lower() in VALID_EXTENSIONS:

                full_path = os.path.join(root, file)

                video_files.append(full_path)

    # =====================================================
    # PROCESS EACH VIDEO
    # =====================================================

    for video_path in tqdm(video_files):

        try:

            is_valid, reason = validate_video(video_path)

            # =================================================
            # VALID VIDEO
            # =================================================

            if is_valid:

                copy_video_with_structure(
                    video_path,
                    CLEAN_DATASET_DIR
                )

                summary["valid"] += 1

            # =================================================
            # REJECTED VIDEO
            # =================================================

            else:

                copy_video_with_structure(
                    video_path,
                    REJECTED_DATASET_DIR
                )

                summary["rejected"] += 1

                print(f"\nRejected: {video_path} --> {reason}")

        except Exception as e:

            copy_video_with_structure(
                video_path,
                REJECTED_DATASET_DIR
            )

            summary["rejected"] += 1

            print(f"\nError: {video_path}")
            print(str(e))

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n====================================")
print("DATASET VALIDATION COMPLETED")
print("====================================")
print(f"Valid Videos    : {summary['valid']}")
print(f"Rejected Videos : {summary['rejected']}")
print("====================================")

hands.close()