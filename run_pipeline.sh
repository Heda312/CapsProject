#!/bin/bash
# Quick Start Script untuk Training Model

echo "================================"
echo "Sign Language Recognition Pipeline"
echo "Optimized for CPU"
echo "================================"
echo ""

# Check if directories exist
if [ ! -d "dataset" ]; then
    echo "❌ Error: 'dataset' folder tidak ditemukan!"
    exit 1
fi

echo "📦 Step 1: Installing dependencies..."
pip install -r requirements.txt
echo ""

echo "🧹 Step 2: Cleaning dataset (Parallel processing)..."
python cleansing.py
echo ""

echo "🔍 Step 3: Extracting landmarks (Frame skip + Resize + Parallel)..."
python extract_landmarks.py
echo ""

echo "📝 Step 4: Preparing dataset..."
python prepare_dataset.py
echo ""

echo "🤖 Step 5: Training LSTM model (Optimized for CPU)..."
python train_lstm.py
echo ""

echo "✅ Pipeline complete!"
echo ""
echo "Output files:"
echo "  - models/best_model.h5"
echo "  - models/final_model.h5"
echo "  - models/training_history.png"
echo "  - models/confusion_matrix.png"
echo ""
