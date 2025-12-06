#!/usr/bin/env python3
"""Quick diagnostic to check disparity values and model predictions"""

import torch
import numpy as np
import sys
import os
import argparse
from dataset_loaders.custom_loader_rectified import CustomLoaderRectified
from depth_estimation_net import SiameseStereoNet

# Parse arguments
parser = argparse.ArgumentParser(description="Quick diagnostic for model evaluation")
parser.add_argument("--model_path", type=str, default=None,
                   help="Path to model checkpoint (auto-detects latest if not specified)")
parser.add_argument("--dataset_dir", type=str, 
                   default="/u50/koenin1/ML_PROJECT/Mono_Duo_Depth/dataset/val/sneakythodelink_all640x480",
                   help="Path to validation dataset")
parser.add_argument("--calib_file", type=str,
                   default="/u50/koenin1/ML_PROJECT/Mono_Duo_Depth/camera_scripts/manual_calibration_refined.npz",
                   help="Path to calibration file")
args = parser.parse_args()

# Auto-detect latest model if not specified
if args.model_path is None:
    results_dir = "/u50/koenin1/ML_PROJECT/Mono_Duo_Depth/results"
    versions = sorted([d for d in os.listdir(results_dir) if d.startswith('v') and d.endswith('_finetune')])
    if versions:
        latest_version = versions[-1]
        args.model_path = os.path.join(results_dir, latest_version, "best_finetuned_model.pth")
        print(f"Auto-detected latest model: {latest_version}")
    else:
        print("ERROR: No model found! Please specify --model_path")
        sys.exit(1)

dataset_dir = args.dataset_dir
calib_file = args.calib_file
model_path = args.model_path

print("=" * 60)
print("DISPARITY AND MODEL DIAGNOSTICS")
print("=" * 60)
print(f"\nModel: {model_path}")
print(f"Dataset: {dataset_dir}")
print()

# 1. Load dataset
print("\n1. Loading validation dataset...")
val_dataset = CustomLoaderRectified(
    root_dir=dataset_dir,
    calib_file=calib_file,
    target_size=(480, 640),
    split='val'
)
print(f"✓ Loaded {len(val_dataset)} validation samples")
print(f"  Rectified focal: {val_dataset.f_rect:.2f} px")
print(f"  Rectified baseline: {val_dataset.B_rect:.4f} m")

# 2. Check ground truth disparity statistics
print("\n2. Ground truth disparity statistics (first 100 samples)...")
gt_disparities = []
for i in range(min(100, len(val_dataset))):
    sample = val_dataset[i]
    disp = sample['disparity'].squeeze().numpy()
    valid_disp = disp[disp > 0]
    if len(valid_disp) > 0:
        gt_disparities.extend(valid_disp.flatten().tolist())

gt_disparities = np.array(gt_disparities)
print(f"  Valid pixels: {len(gt_disparities):,}")
print(f"  Min disparity: {gt_disparities.min():.2f} px")
print(f"  Max disparity: {gt_disparities.max():.2f} px")
print(f"  Mean disparity: {gt_disparities.mean():.2f} px")
print(f"  Median disparity: {np.median(gt_disparities):.2f} px")
print(f"  95th percentile: {np.percentile(gt_disparities, 95):.2f} px")

# Convert to depth for sanity check
f_B = val_dataset.f_rect * val_dataset.B_rect
depths = f_B / gt_disparities
print(f"\n  Corresponding depths:")
print(f"    Min depth: {depths.min():.2f} m (from max disp)")
print(f"    Max depth: {depths.max():.2f} m (from min disp)")
print(f"    Mean depth: {depths.mean():.2f} m")
print(f"    Median depth: {np.median(depths):.2f} m")

# 3. Load model and check predictions
print("\n3. Loading model and checking predictions...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseStereoNet()

if os.path.exists(model_path):
    print(f"  Loading weights from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    # Remove DataParallel wrapper if present
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
else:
    print(f"  WARNING: Model file not found, using random weights!")

model.to(device)
model.eval()

# 4. Run inference on a few samples
print("\n4. Model predictions on first 10 samples...")
pred_disparities = []
errors = []

with torch.no_grad():
    for i in range(min(10, len(val_dataset))):
        sample = val_dataset[i]
        left = sample['left'].unsqueeze(0).to(device)
        right = sample['right'].unsqueeze(0).to(device)
        gt_disp = sample['disparity'].squeeze().numpy()
        
        pred_disp = model(left, right).squeeze().cpu().numpy()
        
        valid_mask = gt_disp > 0
        if valid_mask.sum() > 0:
            pred_valid = pred_disp[valid_mask]
            gt_valid = gt_disp[valid_mask]
            
            pred_disparities.extend(pred_valid.flatten().tolist())
            errors.extend(np.abs(pred_valid - gt_valid).flatten().tolist())
            
            print(f"\n  Sample {i+1}:")
            print(f"    GT:   min={gt_valid.min():.1f}, max={gt_valid.max():.1f}, mean={gt_valid.mean():.1f}")
            print(f"    Pred: min={pred_valid.min():.1f}, max={pred_valid.max():.1f}, mean={pred_valid.mean():.1f}")
            print(f"    EPE:  {np.abs(pred_valid - gt_valid).mean():.2f} px")

pred_disparities = np.array(pred_disparities)
errors = np.array(errors)

print("\n5. Overall prediction statistics:")
print(f"  Predicted disparity range: [{pred_disparities.min():.2f}, {pred_disparities.max():.2f}] px")
print(f"  Predicted mean disparity: {pred_disparities.mean():.2f} px")
print(f"  Overall EPE: {errors.mean():.2f} px")
print(f"  EPE std: {errors.std():.2f} px")
print(f"  EPE median: {np.median(errors):.2f} px")

# 6. Check if predictions are in reasonable range
print("\n6. Diagnosis:")
if pred_disparities.mean() < 5:
    print("  ⚠️  ISSUE: Predictions are too small (mean < 5 px)")
    print("       → Model might be predicting nearly zero everywhere")
    print("       → Check if model converged to trivial solution")
elif pred_disparities.mean() > gt_disparities.mean() * 2:
    print("  ⚠️  ISSUE: Predictions are too large (2x ground truth)")
    print("       → Possible scale mismatch in training")
elif errors.mean() > 10:
    print("  ⚠️  ISSUE: High EPE (>10 px)")
    if abs(pred_disparities.mean() - gt_disparities.mean()) / gt_disparities.mean() > 0.3:
        print("       → Systematic bias detected")
        print(f"       → GT mean: {gt_disparities.mean():.2f}, Pred mean: {pred_disparities.mean():.2f}")
        print("       → Model learned wrong disparity scale")
    else:
        print("       → No systematic bias, but high variance")
        print("       → Model may need more training or better regularization")
else:
    print("  ✓ Predictions look reasonable!")
    print(f"    EPE = {errors.mean():.2f} px is acceptable")

print("\n" + "=" * 60)
