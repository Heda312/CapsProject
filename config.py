import os

# =========================
# DATASET
# =========================
DATASET_DIR = "dataset"
CLEANSED_DIR = "cleansed_dataset"
LANDMARK_DIR = "extracted_landmarks"
PREPARED_DIR = "prepared_dataset"
MODEL_DIR = "models"

# =========================
# VIDEO OPTIMIZATION
# =========================
# Frame skip: mengambil frame setiap N frame untuk faster processing
# Lebih tinggi = lebih cepat tapi kurang detail
FRAME_SKIP = 3  

# Sequence length: jumlah frame per sequence
SEQUENCE_LENGTH = 40

# Minimum frames per video (sebelum frame skip)
MIN_FRAMES = 30

# =========================
# MODEL TRAINING (OPTIMIZED FOR CPU)
# =========================
# Batch size yang lebih kecil untuk CPU
BATCH_SIZE = 8  

# Jumlah epoch
EPOCHS = 100

# Learning rate untuk optimizer
LEARNING_RATE = 0.0005

# =========================
# REALTIME INFERENCE
# =========================
PREDICTION_THRESHOLD = 0.70
SMOOTHING_WINDOW = 10
COOLDOWN_FRAMES = 20

# =========================
# LABELS (Sign Language + Finance)
# =========================
LABELS = [
    # A-Z
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',

    # 0-9
    '0','1','2','3','4','5','6','7','8','9',

    # Custom
    'ATM',
    'Kartu_ATM',
    'Saldo',
    'Transfer',
    'Uang',

    # Financial
    '10',
    '20',
    '50',
    '100',
    '500',
    '1000',
    'Ribu',
    'Juta',
    'Milyar'
]

# Create directories
os.makedirs(CLEANSED_DIR, exist_ok=True)
os.makedirs(LANDMARK_DIR, exist_ok=True)
os.makedirs(PREPARED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)