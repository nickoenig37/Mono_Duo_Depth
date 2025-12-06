#!/bin/bash
# Conservative improved training - no dropout, just LR and regularization fixes

cd /u50/koenin1/ML_PROJECT/Mono_Duo_Depth/neural_network

python3 train_improved.py \
    --dataset_dir ../dataset/train \
    --calib_file ../camera_scripts/manual_calibration_refined.npz \
    --pretrained_model trained_25000_flyingthings_stereo_model.pth \
    --epochs 50 \
    --lr 5e-5 \
    --dropout_p 0.0 \
    --weight_decay 1e-5 \
    --early_stop_patience 15
