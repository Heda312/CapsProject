import os

# ==========================================
# KONFIGURASI
# ==========================================

BASE_DIR = "Dataset"

# ==========================================
# LABEL DATASET
# ==========================================

# ----------------------
# ABJAD A-Z
# ----------------------
label_abjad = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z"
]

# ----------------------
# ANGKA 0-9
# ----------------------
label_angka = [
    "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9"
]

# ----------------------
# CUSTOM
# ----------------------
label_custom = [
    "ATM",
    "Kartu_ATM",
    "Saldo",
    "Transfer",
    "Uang"
]

# ----------------------
# KEUANGAN
# ----------------------
label_keuangan = [
    "10",
    "20",
    "50",
    "100",
    "500",
    "1000",
    "Ribu",
    "Juta",
    "Milyar"
]

# ==========================================
# FUNGSI PEMBUAT FOLDER
# ==========================================

def buat_folder_dataset(kategori, labels):
    """
    Struktur folder:
    
    Dataset/
        Kategori/
            Label/
    """

    for label in labels:

        folder_path = os.path.join(
            BASE_DIR,
            kategori,
            label
        )

        os.makedirs(folder_path, exist_ok=True)

        print(f"[OK] {folder_path}")


# ==========================================
# EKSEKUSI
# ==========================================

buat_folder_dataset("Abjad", label_abjad)
buat_folder_dataset("Angka", label_angka)
buat_folder_dataset("Custom", label_custom)
buat_folder_dataset("Keuangan", label_keuangan)

print("\n====================================")
print("SEMUA FOLDER DATASET BERHASIL DIBUAT")
print("====================================")