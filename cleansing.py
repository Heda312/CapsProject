"""
=============================================================
    CLEANSING.PY - Pembersihan Dataset Video
=============================================================

FUNGSI:
    Membersihkan dataset dengan validasi video quality
    - Menghapus video yang corrupt/corrupted
    - Memvalidasi minimum frame count
    - Proses parallel dengan multiple threads (4 workers)
    - Copy video valid ke folder cleansed_dataset

OPTIMASI:
    - ThreadPoolExecutor: Process 4 video simultaneously
    - Error handling: Robust handling untuk corrupt files
    - Progress tracking: Display status per file
    - Batch processing: Collect tasks sebelum processing

WAKTU ESTIMASI:
    - Dataset 100 video: 3-5 menit (parallel 4 threads)
    - Dataset 1000 video: 30-50 menit

OUTPUT:
    - cleansed_dataset/ folder dengan video valid
    - Console: Progress dan summary statistics

=============================================================
"""

import os
import cv2
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import *


def is_video_valid(video_path):
    """
    Validasi apakah video file valid
    
    PENJELASAN:
        - Open video dengan OpenCV VideoCapture
        - Cek apakah file bisa dibuka (not corrupted)
        - Hitung total frames dalam video
        - Validasi FPS (frames per second)
        - Close video file untuk free resources
    
    RETURN:
        True jika valid, False jika invalid/corrupt
    
    MIN_FRAMES: Ditetapkan di config.py (default: 30)
    """
    try:
        # Buka file video dengan OpenCV
        cap = cv2.VideoCapture(video_path)

        # Cek apakah video berhasil dibuka (not corrupted)
        if not cap.isOpened():
            return False

        # Ambil metadata: jumlah frame dalam video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ambil metadata: FPS (frame per second)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Tutup video file untuk free memory
        cap.release()

        # Validasi: minimum frames dan fps yang reasonable
        # Jika frame < minimum atau FPS invalid → reject
        if total_frames < MIN_FRAMES or fps <= 0:
            return False

        return True
    except Exception as e:
        # Jika ada error apapun → video invalid
        print(f"Error checking {video_path}: {e}")
        return False


def process_video_file(args):
    """
    Proses satu video file dan copy ke output folder jika valid
    
    PENJELASAN:
        - Fungsi ini dijalankan oleh ThreadPoolExecutor (parallel)
        - Validasi video menggunakan is_video_valid()
        - Jika valid: copy ke output folder
        - Return status (valid/invalid) dan message
    
    ARGS:
        Tuple berisi:
        - video_path: path lengkap ke file video
        - output_label_dir: folder destination untuk video valid
        - category: kategori video (Abjad/Angka/Custom/Keuangan)
        - label: label spesifik (A/B/C/0/1/ATM/Rp10/etc)
        - video_name: nama file video
    
    RETURN:
        Tuple (is_valid: bool, message: str)
    """
    video_path, output_label_dir, category, label, video_name = args
    
    try:
        # Validasi video menggunakan fungsi is_video_valid()
        if is_video_valid(video_path):
            # Jika valid: copy file ke output folder
            shutil.copy(video_path, output_label_dir)
            return (True, f"VALID   : {category}/{label}/{video_name}")
        else:
            # Jika invalid: skip video
            return (False, f"INVALID : {category}/{label}/{video_name}")
    except Exception as e:
        # Jika ada error saat copy → log error
        return (False, f"ERROR   : {category}/{label}/{video_name} - {str(e)}")


def cleanse_dataset():
    """
    Main function: Membersihkan semua dataset dengan parallel processing
    
    PROSES:
        1. Scan folder dataset/ untuk semua kategori dan label
        2. Kumpulkan list semua video files
        3. Proses dengan ThreadPoolExecutor (4 parallel threads)
        4. Copy video valid ke cleansed_dataset/
        5. Print summary di akhir
    
    DIRECTORY STRUCTURE:
        dataset/
        ├── Abjad/
        │   ├── A/ (video files)
        │   ├── B/ (video files)
        │   └── ...
        ├── Angka/
        │   ├── 0/ (video files)
        │   └── ...
        ├── Custom/
        │   └── ATM/ (video files)
        └── Keuangan/
            └── JUTA/ (video files)
    
    OUTPUT:
        cleansed_dataset/
        ├── A/ (video files yang valid)
        ├── B/ (video files yang valid)
        └── ... (semua label)
    """
    total_valid = 0
    total_invalid = 0
    
    # STEP 1: Collect semua tasks (video files yang akan diproses)
    # Ini lebih efficient daripada submit satu per satu
    tasks = []
    
    # Iterate kategori (Abjad, Angka, Custom, Keuangan)
    for category in sorted(os.listdir(DATASET_DIR)):
        category_path = os.path.join(DATASET_DIR, category)

        # Skip jika bukan folder
        if not os.path.isdir(category_path):
            continue

        # Iterate label dalam kategori (A, B, C, 0, 1, ATM, etc)
        for label in sorted(os.listdir(category_path)):

            label_path = os.path.join(category_path, label)

            # Skip jika bukan folder
            if not os.path.isdir(label_path):
                continue

            # Buat output folder untuk label ini di cleansed_dataset/
            output_label_dir = os.path.join(CLEANSED_DIR, label)
            os.makedirs(output_label_dir, exist_ok=True)

            print(f"Processing label: {label}")

            # Iterate semua video files dalam label folder
            for video_name in os.listdir(label_path):

                # Skip jika bukan .mp4 file
                if not video_name.endswith('.mp4'):
                    continue

                # Get full path ke video
                video_path = os.path.join(label_path, video_name)
                
                # Tambah ke tasks list (akan diproses parallel nanti)
                tasks.append((video_path, output_label_dir, category, label, video_name))

    # STEP 2: Process semua tasks dengan parallel threads
    # Max workers: 4 (I/O bound operation, tidak CPU intensive)
    print(f"\nStarting parallel processing with 4 workers...")
    print(f"Total videos to process: {len(tasks)}\n")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit semua tasks ke executor
        futures = [executor.submit(process_video_file, task) for task in tasks]
        
        # Proses hasil seperti yang selesai (tidak blocking)
        for future in as_completed(futures):
            is_valid, message = future.result()
            print(message)
            if is_valid:
                total_valid += 1
            else:
                total_invalid += 1

    # STEP 3: Print summary
    print(f"\n{'='*50}")
    print(f"=== CLEANING SUMMARY ===")
    print(f"{'='*50}")
    print(f"✅ Valid videos:   {total_valid}")
    print(f"❌ Invalid videos: {total_invalid}")
    print(f"📊 Total:          {total_valid + total_invalid}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    """
    CARA MENJALANKAN:
        python cleansing.py
    
    WAKTU ESTIMASI:
        100 video  → 3-5 menit
        1000 video → 30-50 menit
    
    OUTPUT:
        cleansed_dataset/ folder dengan video yang valid
    """
    cleanse_dataset()