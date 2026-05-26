@echo off
REM Quick Start Script untuk Training Model (Windows)

echo.
echo ================================
echo Sign Language Recognition Pipeline
echo Optimized for CPU
echo ================================
echo.

REM Check if directories exist
if not exist "dataset" (
    echo Error: 'dataset' folder tidak ditemukan!
    pause
    exit /b 1
)

echo [1/5] Installing dependencies...
pip install -r requirements.txt
echo.

echo [2/5] Cleaning dataset (Parallel processing)...
python cleansing.py
echo.

echo [3/5] Extracting landmarks (Frame skip + Resize + Parallel)...
python extract_landmarks.py
echo.

echo [4/5] Preparing dataset...
python prepare_dataset.py
echo.

echo [5/5] Training LSTM model (Optimized for CPU)...
python train_lstm.py
echo.

echo =====================================
echo Pipeline Complete!
echo =====================================
echo.
echo Output files:
echo   - models/best_model.h5
echo   - models/final_model.h5
echo   - models/training_history.png
echo   - models/confusion_matrix.png
echo.
pause
