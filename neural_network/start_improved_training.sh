#!/bin/bash
# Quick Start - Improved Training with Regularization
# This script starts training with all the improvements to combat overfitting

cd /u50/koenin1/ML_PROJECT/Mono_Duo_Depth/neural_network

echo "============================================"
echo "IMPROVED TRAINING - QUICK START"
echo "============================================"
echo ""
echo "This script will train a new model with:"
echo "  ✓ Dropout (0.2) in decoder layers"
echo "  ✓ Weight decay (1e-5) for L2 regularization"
echo "  ✓ Lower learning rate (5e-5) for stable fine-tuning"
echo "  ✓ Learning rate scheduling (ReduceLROnPlateau)"
echo "  ✓ Gradient clipping (max_norm=1.0)"
echo "  ✓ Early stopping (patience=15 epochs)"
echo ""
echo "Starting from: trained_25000_flyingthings_stereo_model.pth"
echo "Dataset: All training folders in ../dataset/train"
echo ""
echo "Expected improvements over v3:"
echo "  - Smaller train/val gap (<3.0 vs 5.6)"
echo "  - Better validation EPE (<15 px vs 19 px)"
echo "  - Automatic stopping when converged"
echo ""
echo "============================================"
echo ""

# Check if pretrained model exists
if [ ! -f "trained_25000_flyingthings_stereo_model.pth" ]; then
    echo "❌ ERROR: Pre-trained model not found!"
    echo "   Expected: trained_25000_flyingthings_stereo_model.pth"
    echo "   Please ensure the model is in the neural_network directory."
    exit 1
fi

# Check if calibration file exists
if [ ! -f "../camera_scripts/manual_calibration_refined.npz" ]; then
    echo "❌ ERROR: Calibration file not found!"
    echo "   Expected: ../camera_scripts/manual_calibration_refined.npz"
    exit 1
fi

# Check if dataset exists
if [ ! -d "../dataset/train" ]; then
    echo "❌ ERROR: Training dataset not found!"
    echo "   Expected: ../dataset/train/"
    exit 1
fi

echo "✓ All prerequisites found. Starting training..."
echo ""

# Run improved training script
python3 train_improved.py \
    --dataset_dir ../dataset/train \
    --calib_file ../camera_scripts/manual_calibration_refined.npz \
    --pretrained_model trained_25000_flyingthings_stereo_model.pth \
    --output_dir ../results \
    --epochs 50 \
    --lr 5e-5 \
    --dropout_p 0.2 \
    --weight_decay 1e-5 \
    --max_grad_norm 1.0 \
    --lr_patience 5 \
    --early_stop_patience 15 \
    --freeze_encoder

echo ""
echo "============================================"
echo "Training completed!"
echo "Check results in: ../results/v*_finetune/"
echo "  - best_finetuned_model.pth (best model checkpoint)"
echo "  - metrics.png (training curves)"
echo "============================================"
