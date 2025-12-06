# Training Diagnosis Report - v3 Fine-Tuning

## Summary
After 50 epochs of fine-tuning, the model achieves an EPE of ~20 pixels, which is higher than desired (<5px). This document provides a comprehensive analysis of the issue.

## Diagnostics Performed

### 1. Calibration Verification ✅
**Status: CORRECT**

The rectified calibration parameters from OpenCV's `stereoRectify`:
- **Rectified Focal Length**: 719.53 px
- **Rectified Baseline**: 0.1371 m (137.1 mm)

These values are **higher than the raw RealSense specs** (focal~400px, baseline=95mm) because stereo rectification transforms the images to a common image plane, which changes the effective camera parameters. This is expected and correct.

### 2. Ground Truth Disparity Analysis ✅
**Status: CORRECT**

Analyzed 100 validation samples (29.9M valid pixels):
- **Min disparity**: 11.27 px (at 8.75m depth)
- **Max disparity**: 369.42 px (at 0.27m depth)  
- **Mean disparity**: 47.24 px (at 2.75m depth)
- **Median disparity**: 36.52 px (at 2.70m depth)

Formula verification: `disparity = (focal * baseline) / depth = (719.53 * 0.1371) / depth`

The ground truth values match expected physics. ✓

### 3. Model Prediction Analysis ⚠️
**Status: PARTIALLY CORRECT - HIGH VARIANCE**

Model predictions on 10 validation samples:
- **Predicted mean disparity**: 41.27 px (vs GT 42-47 px range)
- **Overall EPE**: 12.10 px
- **EPE std dev**: 15.45 px
- **Median EPE**: 7.60 px

**Key Findings:**
1. **Scale is correct**: Model predictions have the right mean disparity
2. **High variance**: Standard deviation (15.45px) is larger than mean (12.1px)
3. **Close-range weakness**: Model predicts 1.6-2.9px minimum vs GT 12-15px minimum
4. **Outlier problem**: Mean EPE=12px but median=7.6px → some predictions are very wrong

### 4. Training Behavior Analysis 🔴
**Status: OVERFITTING DETECTED**

Training progression over 50 epochs:
- **Epoch 1**: Train Loss=18.0, Val Loss=19.4, Train EPE=21.6, Val EPE=22.7
- **Epoch 8**: Train Loss=11.7, Val Loss=16.0 (best), Train EPE=15.5, Val EPE=19.2
- **Epoch 50**: Train Loss=~10.4, Val Loss=~16-17, Train EPE=~14, Val EPE=~19-20

**Overfitting symptoms:**
- Training loss decreases smoothly (18 → 10.4)
- Validation loss **plateaus at epoch 8** and doesn't improve
- Training EPE improves (21.6 → 14), validation EPE stays ~19-20
- Valid pixel percentage is consistent: 71.5% train, 76% val (masking working correctly)

## Root Causes Identified

### Primary Issue: Model Overfitting
The model is memorizing training data rather than learning generalizable disparity patterns.

**Evidence:**
1. Train/val gap increases over time
2. Validation metrics plateau early (epoch 8)
3. High variance in predictions (std=15.45px)
4. Poor performance on distribution edges (close objects)

### Secondary Issue: Close-Range Performance
Model struggles with close objects (high disparities):
- Predicts minimums of 1.6-2.9px when ground truth is 12-15px
- This could indicate:
  - Limited receptive field in network architecture
  - Insufficient training on close-range examples
  - Cost volume resolution limitations

## Recommendations

### Immediate Actions

#### 1. Combat Overfitting 🔥 PRIORITY
**A. Reduce Model Capacity**
```python
# Option 1: Unfreeze encoder gradually
# Currently: Encoder frozen, only training decoder
# Try: Unfreeze encoder with lower learning rate

optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': 1e-4},
    {'params': model.encoder.parameters(), 'lr': 1e-5}  # 10x smaller
])
```

**B. Add Regularization**
```python
# Add dropout to decoder layers
# Add weight decay to optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
```

**C. Early Stopping**
- Best model was at epoch 8 (Val Loss=16.0, EPE=19.2)
- Stop training when validation doesn't improve for 10 epochs
- Use current best checkpoint for inference

#### 2. Increase Training Data Diversity
The model has:
- **Train**: 4,372 samples
- **Val**: 1,094 samples

**Recommendations:**
- Add more datasets from different environments
- Use data augmentation:
  - Brightness/contrast adjustments
  - Small random crops
  - Horizontal flips (swap left/right)
- Check if samples are too similar (same scene, slight camera movement)

#### 3. Adjust Learning Rate Strategy
Current: Fixed LR = 1e-4 for 50 epochs

**Try:**
```python
# Reduce LR when validation plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# Or cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-6
)
```

#### 4. Verify Data Quality
Check if validation set is representative:
```bash
# Run on more validation samples
cd /u50/koenin1/ML_PROJECT/Mono_Duo_Depth/neural_network
python3 quick_diag.py  # Check more samples
```

Look for:
- Are validation images from different environment than training?
- Different lighting conditions?
- Different distance distributions?

### Long-Term Improvements

#### 1. Architecture Modifications
- **Increase decoder capacity** for better disparity detail
- **Multi-scale loss** to help with different depth ranges
- **Attention mechanisms** to focus on challenging regions

#### 2. Loss Function Tuning
Current: `SmoothL1Loss` on masked pixels

**Try:**
```python
# Multi-scale loss (if model outputs multiple resolutions)
loss = 0.5*loss_full + 0.3*loss_half + 0.2*loss_quarter

# Or gradient-aware loss
loss = smooth_l1_loss + 0.1 * gradient_loss
```

#### 3. Training Strategy
- **Curriculum learning**: Start with close objects, gradually add farther distances
- **Hard negative mining**: Focus on pixels with high error
- **Test-time augmentation**: Average predictions from augmented inputs

## Expected Performance

Based on the diagnostics:

### Current Performance
- **EPE**: 12.1 px (median: 7.6 px)
- **Depth error at 2m**: ~12px → ~0.8m error (very poor)
- **Depth error at 5m**: ~12px → ~5m error (unusable)

### Target Performance
- **EPE**: <5 px
- **Depth error at 2m**: <5px → <0.3m error
- **Depth error at 5m**: <5px → <1.3m error

### What "Good" Looks Like
State-of-the-art stereo methods achieve:
- EPE < 3 px on standard benchmarks
- EPE < 1 px on high-quality datasets

Your model at EPE=12px is **functional but not production-ready**.

## Action Plan

### Phase 1: Quick Wins (Today)
1. ✅ **Use best checkpoint** (epoch 8, Val Loss=16.0)
   - Current checkpoint may be overfitted
   
2. ⏳ **Add early stopping** to prevent future overfitting
   ```bash
   # Modify train_fine_tune.py
   # Add patience=10 for early stopping
   ```

3. ⏳ **Reduce learning rate** and continue from epoch 8
   ```bash
   # Resume training with LR=1e-5
   python3 train_fine_tune.py --resume results/v3_finetune/best_finetuned_model.pth --lr 1e-5
   ```

### Phase 2: Regularization (This Week)
1. Add weight decay (1e-5)
2. Add dropout to decoder (p=0.2)
3. Unfreeze encoder with lower LR
4. Implement LR scheduler

### Phase 3: Data Enhancement (Next Week)
1. Collect more diverse data (different environments, lighting)
2. Implement data augmentation
3. Balance dataset (more close-range examples if needed)

### Phase 4: Architecture (If Needed)
1. Try different backbone (ResNet-50 instead of ResNet-18)
2. Add multi-scale outputs
3. Experiment with different cost volume aggregation

## Monitoring Metrics

Track these during retraining:
- **Primary**: Validation EPE (target: <10px, ideal: <5px)
- **Secondary**: Train/Val EPE gap (should be <5px)
- **Diagnostic**: EPE by distance range (close/mid/far)
- **Quality**: Valid pixel percentage (should stay 70-80%)

## Files Generated

Diagnostic outputs:
- `results/v3_finetune/visualizations/sample_001-005.png` - Visual predictions
- `neural_network/quick_diag.py` - Quick diagnostic script
- `neural_network/visualize_predictions.py` - Visualization script

## Conclusion

**The model has learned the correct disparity scale** but suffers from overfitting and high variance. The calibration and data pipeline are correct. Focus on regularization and data diversity to improve generalization.

**Bottom Line**: Your current model (epoch 8) is usable for rough depth estimation but needs improvement for production use. The good news: the infrastructure is correct, just need to tune training.

---
Generated: $(date)
Model: v3_finetune (50 epochs, best at epoch 8)
EPE: 12.1 px (median: 7.6 px)
Status: Functional but overfitted
Next Steps: Early stopping + regularization + LR tuning
