# Quick Reference - Improved Training

## What Was Changed

### 1. Model Architecture (`depth_estimation_net.py`)
- ✅ Added `dropout_p` parameter to `SiameseStereoNet.__init__()`
- ✅ Added `nn.Dropout2d(p=dropout_p)` after each decoder conv layer (3 locations)
- Default dropout: 20% (p=0.2)

### 2. New Training Script (`train_improved.py`)
Created a new training script with all improvements:
- ✅ Dropout support (configurable via `--dropout_p`)
- ✅ Weight decay / L2 regularization (`--weight_decay`)
- ✅ Lower default learning rate (`--lr 5e-5`)
- ✅ Learning rate scheduling (`ReduceLROnPlateau`)
- ✅ Gradient clipping (`--max_grad_norm`)
- ✅ Early stopping (`--early_stop_patience`)
- ✅ Enhanced metrics plotting (includes LR curve)

## How to Train

### Option 1: Quick Start (Recommended)
```bash
cd /u50/koenin1/ML_PROJECT/Mono_Duo_Depth/neural_network
./start_improved_training.sh
```

### Option 2: Custom Parameters
```bash
cd /u50/koenin1/ML_PROJECT/Mono_Duo_Depth/neural_network

python3 train_improved.py \
    --dataset_dir ../dataset/train \
    --calib_file ../camera_scripts/manual_calibration_refined.npz \
    --pretrained_model trained_25000_flyingthings_stereo_model.pth \
    --epochs 50 \
    --lr 5e-5 \
    --dropout_p 0.2 \
    --weight_decay 1e-5 \
    --early_stop_patience 15
```

### Option 3: Experiment with Different Settings

**Less regularization (if model underperforms):**
```bash
python3 train_improved.py --dropout_p 0.1 --lr 7e-5 --weight_decay 5e-6
```

**More regularization (if still overfitting):**
```bash
python3 train_improved.py --dropout_p 0.3 --lr 3e-5 --weight_decay 5e-5
```

## All Available Arguments

```
Data:
  --dataset_dir         Path to training dataset (default: ../dataset/train)
  --calib_file          Path to calibration file (default: ../camera_scripts/manual_calibration_refined.npz)
  --output_dir          Output directory (default: ../results)

Model:
  --pretrained_model    Pre-trained model path (default: trained_25000_flyingthings_stereo_model.pth)
  --freeze_encoder      Freeze encoder layers (default: True)
  --dropout_p           Dropout probability (default: 0.2, range: 0.0-0.5)

Training:
  --epochs              Max epochs (default: 50)
  --lr                  Learning rate (default: 5e-5)
  --weight_decay        L2 regularization (default: 1e-5)
  --max_grad_norm       Gradient clipping (default: 1.0, 0=disabled)

Regularization:
  --lr_patience         LR scheduler patience (default: 5)
  --early_stop_patience Early stopping patience (default: 15)
```

## Why Lower Learning Rate?

### The Simple Explanation
**High LR (1e-4)**: Makes big weight changes → destroys pre-trained knowledge → overfits  
**Low LR (5e-5)**: Makes small weight refinements → preserves pre-trained knowledge → generalizes

### The Detailed Explanation
Your model starts with **good weights** from FlyingThings pre-training. These weights already know:
- How to detect edges and textures
- Basic depth relationships
- Feature extraction

**Fine-tuning** means making **small adjustments** to these weights for RealSense data.

**High learning rate** treats it like training from scratch:
- Large weight updates (e.g., 0.5 → 0.7)
- Destroys learned patterns
- Overshoots optimal solutions
- Model "forgets" FlyingThings knowledge

**Low learning rate** makes gentle refinements:
- Small weight updates (e.g., 0.5 → 0.51)
- Preserves learned patterns
- Converges to local minimum smoothly
- Adapts pre-trained knowledge to RealSense data

### Analogy
Think of tuning a guitar:
- **High LR**: Turning the peg too fast → breaks the string or goes way out of tune
- **Low LR**: Turning gently → fine-tunes to perfect pitch

Your model is already "in tune" from pre-training. You just need gentle adjustments!

### Mathematical Perspective
Gradient descent: `weight_new = weight_old - lr * gradient`

- **lr = 1e-4**: If gradient = 1.0, weight changes by 0.0001 (0.01%)
- **lr = 5e-5**: If gradient = 1.0, weight changes by 0.00005 (0.005%)

The smaller LR makes updates **2x gentler**, reducing overfitting.

## Expected Results

### v3 Training (Before Improvements)
- Train Loss: 10.4, Val Loss: 16.0 → Gap = 5.6 ❌ Overfitting
- Train EPE: 14.0, Val EPE: 19.0 → Gap = 5.0 ❌ Poor generalization
- Best epoch: 8, then plateau for 42 epochs ❌ Wasted computation

### Improved Training (Expected)
- Train/Val gap: < 3.0 ✅ Less overfitting
- Val EPE: 12-15 px ✅ Better performance
- Early stopping at ~20-30 epochs ✅ Efficient training
- Smooth LR reductions when plateau ✅ Adaptive optimization

## Monitoring Training

Watch the terminal output for:

**Good signs ✅:**
- Train and val losses both decreasing
- Train/val gap stays small (<3.0)
- EPE dropping below 15 px
- LR reductions followed by improvement

**Warning signs ⚠️:**
- Val loss increases while train decreases (overfitting)
- No improvement for >10 epochs (may need manual stop)
- EPE > 20 px after 20 epochs (check data quality)

**Great signs 🎉:**
- Val loss < 15.0
- EPE < 12 px
- Early stopping triggers naturally
- "Saved new best model" messages

## Files Created

1. **`depth_estimation_net.py`** (modified)
   - Added dropout parameter and layers

2. **`train_improved.py`** (new)
   - Complete training script with all improvements

3. **`start_improved_training.sh`** (new)
   - Quick start script with recommended settings

4. **`IMPROVED_TRAINING_EXPLANATION.md`** (new)
   - Detailed explanation of all changes

5. **`TRAINING_QUICK_REFERENCE.md`** (this file)
   - Quick lookup for commands and settings

## Troubleshooting

**Q: Training is too slow**  
A: Lower `--early_stop_patience` to 10, or reduce `--epochs` to 30

**Q: Still overfitting (train/val gap > 5)**  
A: Increase `--dropout_p` to 0.3, increase `--weight_decay` to 5e-5

**Q: Model performs worse than before**  
A: Try less regularization: `--dropout_p 0.1 --weight_decay 5e-6`

**Q: Validation loss increases**  
A: Lower learning rate: `--lr 3e-5`

**Q: Want to resume from checkpoint**  
A: Add `--pretrained_model ../results/v5_finetune/best_finetuned_model.pth`

## Quick Comparison

| Aspect | v3 Training | Improved Training |
|--------|-------------|-------------------|
| Dropout | None | 0.2 (20%) |
| Weight Decay | 1e-4 | 1e-5 |
| Learning Rate | 1e-4 | 5e-5 (2x lower) |
| LR Scheduling | Basic | ReduceLROnPlateau |
| Gradient Clip | No | Yes (1.0) |
| Early Stopping | No | Yes (15 epochs) |
| Val EPE | ~19 px | Target: 12-15 px |
| Overfitting | High (gap=5.6) | Target: Low (gap<3.0) |

## Next Steps After Training

1. **Check results:**
   ```bash
   ls ../results/v*_finetune/
   # Look for: best_finetuned_model.pth, metrics.png
   ```

2. **Visualize predictions:**
   ```bash
   python3 visualize_predictions.py
   # Check: results/v*_finetune/visualizations/
   ```

3. **Run diagnostics:**
   ```bash
   python3 quick_diag.py
   # Check EPE and disparity statistics
   ```

4. **If results are good:**
   - Use model for inference
   - Collect more diverse data
   - Train longer with even lower LR

5. **If results are still poor:**
   - Check TRAINING_DIAGNOSIS.md for more ideas
   - Try different regularization settings
   - Consider collecting better quality data

## Summary

You're now training with:
- **Dropout** → Prevents memorization
- **Weight Decay** → Encourages simple solutions  
- **Lower LR** → Gentle, stable fine-tuning
- **LR Scheduling** → Adapts to training progress
- **Early Stopping** → Stops at optimal point

This should significantly reduce overfitting and improve validation performance! 🚀

---
For detailed explanations, see: `IMPROVED_TRAINING_EXPLANATION.md`
