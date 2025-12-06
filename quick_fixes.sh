#!/bin/bash
# Quick Actions to Improve Training
# Run these commands in order to address overfitting

cd /u50/koenin1/ML_PROJECT/Mono_Duo_Depth/neural_network

echo "============================================"
echo "QUICK FIXES FOR OVERFITTING"
echo "============================================"
echo ""

# 1. Copy best model (epoch 8) to a safe location
echo "1. Backing up best model (epoch 8)..."
cp results/v3_finetune/best_finetuned_model.pth results/v3_finetune/epoch_8_backup.pth
echo "   ✓ Saved to: results/v3_finetune/epoch_8_backup.pth"
echo ""

# 2. Test with current best model
echo "2. Current model performance:"
echo "   EPE: ~12.1 px (median: 7.6 px)"
echo "   Status: Overfitted after epoch 8"
echo "   Recommendation: Use epoch 8 checkpoint for inference"
echo ""

# 3. Show what to do next
echo "3. Recommended next steps:"
echo ""
echo "   Option A: Resume training with lower LR (RECOMMENDED)"
echo "   --------------------------------------------------"
echo "   python3 train_fine_tune.py \\"
echo "       --resume results/v3_finetune/epoch_8_backup.pth \\"
echo "       --lr 1e-5 \\"
echo "       --epochs 30 \\"
echo "       --output_dir results/v4_finetune"
echo ""
echo "   Option B: Add regularization and retrain from scratch"
echo "   ----------------------------------------------------"
echo "   1. Edit train_fine_tune.py:"
echo "      - Add weight_decay=1e-5 to optimizer"
echo "      - Add dropout to model"
echo "      - Add LR scheduler"
echo "   2. python3 train_fine_tune.py --output_dir results/v4_finetune"
echo ""
echo "   Option C: Just use the current model (epoch 8)"
echo "   ---------------------------------------------"
echo "   For rough depth estimation, current model is acceptable."
echo "   EPE=12px gives ~0.8m error at 2m distance."
echo ""

echo "4. Check visualizations:"
echo "   ls results/v3_finetune/visualizations/"
echo "   # Copy to local machine to view:"
echo "   # scp user@ada:'/u50/koenin1/ML_PROJECT/Mono_Duo_Depth/results/v3_finetune/visualizations/*.png' ."
echo ""

echo "5. Monitor future training:"
echo "   - Watch for train/val EPE gap (<5px is good)"
echo "   - Stop if val EPE doesn't improve for 10 epochs"
echo "   - Target: EPE <10px (good), EPE <5px (excellent)"
echo ""

echo "============================================"
echo "For detailed analysis, see: TRAINING_DIAGNOSIS.md"
echo "============================================"
