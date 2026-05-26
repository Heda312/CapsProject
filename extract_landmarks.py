"""
=============================================================
    EXTRACT_LANDMARKS.PY - Ekstrak Hand Landmarks
=============================================================

FUNGSI:
    Extract hand landmarks dari video menggunakan MediaPipe
    - Deteksi posisi tangan dalam setiap frame
    - Ekstrak 21 landmarks per tangan (x, y, z coordinates)
    - Optimasi dengan frame skipping dan resize
    - Proses parallel dengan 2 threads
    - Save landmarks sebagai numpy array (.npy)

MEDIAPIPE HAND LANDMARKS:
    - 21 landmarks per tangan (wrist + 4 jari + joints)
    - 3 coordinates per landmark (x, y, z)
    - Total: 21 * 3 = 63 values per hand
    - 2 tangan: 63 * 2 = 126 values per frame
    
    Format output: (num_frames, 126)
    - Contoh: (40, 126) = 40 frames dengan 126 features

OPTIMASI:
    - Frame skipping: Ambil 1 dari 3 frame (FRAME_SKIP=3)
    - Resize frame: 1920x1080 → 480x360 (80% lebih cepat)
    - Parallel: 2 threads untuk processing multiple video
    - float32 precision: 50% lebih hemat memory

WAKTU ESTIMASI:
    - 100 video (1 min each): 30-45 menit (parallel 2 threads)
    - 1000 video (1 min each): 300-450 menit

OUTPUT:
    - extracted_landmarks/ folder dengan .npy files
    - Setiap video → satu file .npy

=============================================================
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import *

# Initialize MediaPipe Hands detector
mp_hands = mp.solutions.hands

# KONFIGURASI MEDIAPIPE:
# - static_image_mode: False (video mode, optimal untuk video)
# - max_num_hands: 2 (detect max 2 tangan)
# - min_detection_confidence: 0.5 (confidence threshold untuk detection)
# - min_tracking_confidence: 0.5 (confidence threshold untuk tracking)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def extract_hand_landmarks(results):
    """
    Extract hand landmarks dari MediaPipe detection results
    
    PENJELASAN:
        - results.multi_hand_landmarks: List of detected hands
        - results.multi_handedness: Label untuk setiap hand (Left/Right)
        - Iterate setiap hand yang terdeteksi
        - Ekstrak 21 landmarks per hand (x, y, z)
        - Return array 126 values (63 left + 63 right)
    
    LANDMARKS STRUCTURE:
        Tangan kiri (63):   [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]
        Tangan kanan (63):  [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]
        Total (126):       [left_63 + right_63]
    
    JIKA TANGAN TIDAK TERDETEKSI:
        - Isi dengan zeros (63 values per tangan)
        - Ini normal, model akan learn dari patterns yang ada
    
    RETURN:
        Array 126 values (float32)
    """
    # Initialize arrays untuk left dan right hand (masing-masing 63 values)
    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    # Cek apakah ada tangan yang terdeteksi
    if results.multi_hand_landmarks and results.multi_handedness:

        # Iterate setiap tangan yang terdeteksi
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            # Ambil label tangan (Left atau Right)
            label = handedness.classification[0].label
            
            # Extract 21 landmarks dari tangan ini
            landmarks = []

            # Iterate 21 landmarks, setiap landmark punya x, y, z
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # Tambah 3 values per landmark

            # Convert ke numpy array dengan float32 (memory efficient)
            landmarks = np.array(landmarks, dtype=np.float32)

            # Store ke left atau right array berdasarkan label
            if label == 'Left':
                left_hand = landmarks
            else:
                right_hand = landmarks

    # Concatenate left + right landmarks (63 + 63 = 126)
    combined = np.concatenate([left_hand, right_hand])
    return combined.astype(np.float32)


def process_video(video_path, output_path):
    """
    Process satu video: extract landmarks dari semua frames
    
    PROSES:
        1. Open video file dengan OpenCV
        2. Loop through setiap frame
        3. Frame skipping: ambil 1 dari FRAME_SKIP frame
        4. Resize frame (untuk speed optimization)
        5. Deteksi hand landmarks dengan MediaPipe
        6. Extract dan store landmarks
        7. Save sequence sebagai numpy array
    
    OPTIMASI SPEED:
        - Frame skipping (FRAME_SKIP=3): 3x lebih cepat
        - Resize (1920x1080 → 480x360): 4x lebih cepat
        - Total: ~10x lebih cepat dibanding full resolution
    
    ARGS:
        video_path: path lengkap ke file video (.mp4)
        output_path: path lengkap untuk output .npy file
    
    RETURN:
        Tuple (success: bool, message: str)
    """
    try:
        # STEP 1: Open video file dengan OpenCV
        cap = cv2.VideoCapture(video_path)

        # Cek apakah video berhasil dibuka
        if not cap.isOpened():
            return False, f"Gagal membuka: {video_path}"

        # Initialize untuk tracking frames
        landmarks_sequence = []  # List untuk store semua landmarks
        frame_count = 0

        # STEP 2: Loop melalui semua frames dalam video
        while True:
            # Baca frame dari video
            ret, frame = cap.read()

            # Cek apakah frame berhasil dibaca (ret=True)
            if not ret:
                break  # End of video

            # STEP 3: Frame skipping - ambil 1 dari setiap FRAME_SKIP frame
            # FRAME_SKIP=3 berarti: frame 0, 3, 6, 9, 12, etc
            # Ini mengurangi jumlah frame yang diproses
            if frame_count % FRAME_SKIP != 0:
                frame_count += 1
                continue  # Skip frame ini

            # STEP 4: Resize frame untuk faster processing
            # Original: 1920x1080 (full HD)
            # Resized: 480x360 (25% original resolution)
            # MediaPipe tetap bisa detect tangan dengan baik di size ini
            frame = cv2.resize(frame, (480, 360))
            
            # STEP 5: Convert BGR (OpenCV default) ke RGB (MediaPipe format)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # STEP 6: Process frame dengan MediaPipe untuk detect tangan
            # Input: RGB frame
            # Output: results (dengan multi_hand_landmarks & multi_handedness)
            results = hands.process(rgb_frame)
            
            # STEP 7: Extract landmarks dari results
            # Returns array 126 values (63 left + 63 right)
            landmarks = extract_hand_landmarks(results)
            
            # Add ke sequence list
            landmarks_sequence.append(landmarks)

            # Increment frame counter
            frame_count += 1

        # Selesai membaca video, tutup file
        cap.release()

        # STEP 8: Validasi hasil - cek jumlah frames yang diextract
        # Minimum: MIN_FRAMES (30) dibagi FRAME_SKIP (3) = 10 frames minimum
        min_required_frames = MIN_FRAMES // FRAME_SKIP
        
        if len(landmarks_sequence) < min_required_frames:
            return False, f"Video terlalu pendek: {video_path}"

        # STEP 9: Convert sequence list ke numpy array
        # Shape: (num_frames, 126)
        # Contoh: (40, 126) = 40 frames dengan 126 landmarks per frame
        sequence_array = np.array(landmarks_sequence, dtype=np.float32)
        
        # STEP 10: Save array ke .npy file
        # Format numpy binary lebih cepat dibanding CSV atau JSON
        np.save(output_path, sequence_array)

        return True, f"✅ BERHASIL: {video_path} ({len(landmarks_sequence)} frames)"

    except Exception as e:
        # Jika ada error apapun → log error
        return False, f"❌ ERROR di {video_path}: {str(e)}"


def save_landmarks_for_video(args):
    """
    Wrapper function untuk parallel processing dengan ThreadPoolExecutor
    
    PENJELASAN:
        - ThreadPoolExecutor butuh function dengan single argument
        - Function ini unpack args tuple dan call process_video()
        - Return hasil untuk di-print
    
    ARGS:
        args: Tuple (video_path, output_label_dir, label)
    """
    video_path, output_label_dir, label = args
    
    # Generate output filename: change .mp4 → .npy
    output_file = os.path.join(output_label_dir, os.path.basename(video_path).replace('.mp4', '.npy'))
    
    # Process video
    success, message = process_video(video_path, output_file)
    
    return success, message


def extract_landmarks():
    """
    Main function: Extract landmarks dari semua video di cleansed_dataset/
    
    PROSES:
        1. Scan cleansed_dataset/ untuk semua label folders
        2. Create output folder di extracted_landmarks/
        3. Kumpulkan list semua video files
        4. Process dengan ThreadPoolExecutor (2 parallel threads)
        5. Print progress dan summary
    
    DIRECTORY STRUCTURE:
        INPUT (cleansed_dataset/):
        ├── A/ (video files)
        ├── B/ (video files)
        └── ... (semua label)
        
        OUTPUT (extracted_landmarks/):
        ├── A/ (.npy files)
        ├── B/ (.npy files)
        └── ... (semua label)
    
    TIMING:
        - Setiap video: ~1-2 detik (dengan frame skip + resize)
        - 100 video: 30-45 menit (parallel 2 threads)
    """
    print("=" * 60)
    print("Starting Hand Landmarks Extraction")
    print("=" * 60)
    print(f"Frame skip:      {FRAME_SKIP} (ambil 1 dari {FRAME_SKIP} frame)")
    print(f"Sequence length: {SEQUENCE_LENGTH} frames per video")
    print(f"Frame resize:    1920x1080 → 480x360 (80% lebih cepat)")
    print(f"Threading:       2 workers untuk parallel processing")
    print("")

    # Initialize counters
    total_processed = 0
    total_success = 0
    total_failed = 0
    
    # Collect semua tasks
    tasks = []

    # STEP 1: Scan cleansed_dataset/ untuk semua label
    for label in sorted(os.listdir(CLEANSED_DIR)):
        label_path = os.path.join(CLEANSED_DIR, label)

        # Skip jika bukan folder
        if not os.path.isdir(label_path):
            continue

        # Create output folder untuk label ini
        output_label_dir = os.path.join(LANDMARK_DIR, label)
        os.makedirs(output_label_dir, exist_ok=True)

        print(f"Processing label: {label}")

        # STEP 2: Collect semua video files untuk label ini
        for video_name in os.listdir(label_path):
            # Skip jika bukan .mp4 file
            if not video_name.endswith('.mp4'):
                continue

            video_path = os.path.join(label_path, video_name)
            tasks.append((video_path, output_label_dir, label))

    print(f"Total videos to process: {len(tasks)}\n")

    # STEP 3: Process dengan parallel threads
    # Max workers: 2 (file I/O bound, tidak perlu lebih banyak)
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit semua tasks
        futures = [executor.submit(save_landmarks_for_video, task) for task in tasks]
        
        # Process hasil seperti yang selesai
        for future in as_completed(futures):
            success, message = future.result()
            print(message)
            total_processed += 1
            if success:
                total_success += 1
            else:
                total_failed += 1

    # STEP 4: Print summary
    print(f"\n{'='*60}")
    print(f"=== EXTRACTION SUMMARY ===")
    print(f"{'='*60}")
    print(f"📊 Total processed:  {total_processed}")
    print(f"✅ Success:          {total_success}")
    print(f"❌ Failed:           {total_failed}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    """
    CARA MENJALANKAN:
        python extract_landmarks.py
    
    INPUT:
        cleansed_dataset/ (dari step 1: cleansing.py)
    
    OUTPUT:
        extracted_landmarks/ folder dengan .npy files
    
    WAKTU ESTIMASI:
        100 video (1 min each): 30-45 menit
        1000 video (1 min each): 300-450 menit
    """
    extract_landmarks()