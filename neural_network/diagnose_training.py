#!/usr/bin/env python3
"""
Comprehensive diagnostic script to debug training issues.
Checks calibration, disparity scale, rectification quality, and data statistics.
"""

import sys
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_loaders.custom_loader_rectified import CustomLoaderRectified

def check_calibration(calib_file):
    """Check calibration parameters."""
    print("\n" + "="*60)
    print("1. CALIBRATION CHECK")
    print("="*60)
    
    if not os.path.exists(calib_file):
        print(f"❌ Calibration file not found: {calib_file}")
        return None
    
    calib = np.load(calib_file)
    print(f"✓ Loaded calibration from: {calib_file}")
    print(f"\nAvailable keys: {list(calib.keys())}")
    
    # Check intrinsics
    if 'K1' in calib and 'K2' in calib:
        K1 = calib['K1']
        K2 = calib['K2']
        
        fx1, fy1 = K1[0,0], K1[1,1]
        fx2, fy2 = K2[0,0], K2[1,1]
        
        print(f"\nLeft Camera Intrinsics (K1):")
        print(f"  Focal length X: {fx1:.2f} px")
        print(f"  Focal length Y: {fy1:.2f} px")
        print(f"  Principal point: ({K1[0,2]:.2f}, {K1[1,2]:.2f})")
        
        print(f"\nRight Camera Intrinsics (K2):")
        print(f"  Focal length X: {fx2:.2f} px")
        print(f"  Focal length Y: {fy2:.2f} px")
        print(f"  Principal point: ({K2[0,2]:.2f}, {K2[1,2]:.2f})")
        
        # Check focal length mismatch
        fx_diff = abs(fx1 - fx2) / fx1 * 100
        fy_diff = abs(fy1 - fy2) / fy1 * 100
        
        if fx_diff > 5 or fy_diff > 5:
            print(f"\n⚠️  WARNING: Large focal length mismatch!")
            print(f"   X difference: {fx_diff:.1f}%")
            print(f"   Y difference: {fy_diff:.1f}%")
        else:
            print(f"\n✓ Focal lengths match well (diff < 5%)")
    
    # Check baseline
    if 'T' in calib:
        T = calib['T']
        baseline = abs(T[0])  # Horizontal baseline
        print(f"\nStereo Baseline:")
        print(f"  Translation: {T}")
        print(f"  Horizontal baseline: {baseline*1000:.2f} mm ({baseline:.4f} m)")
        
        # Typical baselines: 50-150mm for stereo cameras
        if baseline < 0.03 or baseline > 0.20:
            print(f"⚠️  WARNING: Unusual baseline (typical: 50-150mm)")
        else:
            print(f"✓ Baseline looks reasonable")
    
    # Check rectification
    if 'R1' in calib and 'R2' in calib:
        R1 = calib['R1']
        R2 = calib['R2']
        print(f"\n✓ Rectification matrices present")
        print(f"  R1 determinant: {np.linalg.det(R1):.4f} (should be ~1.0)")
        print(f"  R2 determinant: {np.linalg.det(R2):.4f} (should be ~1.0)")
    
    return calib


def check_dataset_statistics(dataset_dir, calib_file):
    """Check disparity statistics from the dataset."""
    print("\n" + "="*60)
    print("2. DATASET DISPARITY STATISTICS")
    print("="*60)
    
    try:
        dataset = CustomLoaderRectified(
            root_dir=dataset_dir,
            calib_file=calib_file,
            target_size=(480, 640),
            split='train',
            split_ratio=0.8
        )
        
        print(f"✓ Loaded dataset with {len(dataset)} samples")
        
        # Sample a few disparities
        disparities = []
        num_samples = min(20, len(dataset))
        
        print(f"\nAnalyzing {num_samples} samples...")
        
        for i in range(num_samples):
            sample = dataset[i]
            disp = sample['disparity'].numpy()
            
            # Filter out invalid values
            valid_disp = disp[np.isfinite(disp) & (disp > 0)]
            
            if len(valid_disp) > 0:
                disparities.append(valid_disp)
        
        if disparities:
            all_disp = np.concatenate(disparities)
            
            print(f"\n📊 Disparity Statistics (from {num_samples} samples):")
            print(f"  Min:        {all_disp.min():.2f} pixels")
            print(f"  Max:        {all_disp.max():.2f} pixels")
            print(f"  Mean:       {all_disp.mean():.2f} pixels")
            print(f"  Median:     {np.median(all_disp):.2f} pixels")
            print(f"  Std:        {all_disp.std():.2f} pixels")
            print(f"  25th %ile:  {np.percentile(all_disp, 25):.2f} pixels")
            print(f"  75th %ile:  {np.percentile(all_disp, 75):.2f} pixels")
            print(f"  99th %ile:  {np.percentile(all_disp, 99):.2f} pixels")
            
            # Check if in reasonable range for FlyingThings3D pre-trained model
            print(f"\n📋 Range Analysis:")
            if all_disp.max() > 192:
                print(f"⚠️  WARNING: Max disparity ({all_disp.max():.0f}) > 192")
                print(f"   FlyingThings3D model was trained on 0-192 range")
                print(f"   Your disparities might be on wrong scale!")
            else:
                print(f"✓ Disparities in FlyingThings3D range (0-192)")
            
            if all_disp.mean() < 5:
                print(f"⚠️  WARNING: Mean disparity very low ({all_disp.mean():.1f})")
                print(f"   This suggests objects are very far away (>2m)")
            elif all_disp.mean() > 50:
                print(f"⚠️  WARNING: Mean disparity very high ({all_disp.mean():.1f})")
                print(f"   This suggests objects are very close (<0.5m)")
            else:
                print(f"✓ Mean disparity looks reasonable")
            
            # Estimate depth range
            calib = np.load(calib_file)
            if 'K1' in calib and 'T' in calib:
                focal = calib['K1'][0, 0]
                baseline = abs(calib['T'][0])
                
                min_depth = (baseline * focal) / all_disp.max()
                max_depth = (baseline * focal) / all_disp.min()
                mean_depth = (baseline * focal) / all_disp.mean()
                
                print(f"\n🎯 Estimated Depth Range:")
                print(f"  Min depth:  {min_depth:.2f} m")
                print(f"  Max depth:  {max_depth:.2f} m")
                print(f"  Mean depth: {mean_depth:.2f} m")
                
                if max_depth > 10:
                    print(f"⚠️  WARNING: Max depth > 10m (RealSense unreliable)")
                if min_depth < 0.3:
                    print(f"⚠️  WARNING: Min depth < 0.3m (RealSense minimum)")
            
            return all_disp
        else:
            print("❌ No valid disparities found in samples!")
            return None
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_model_predictions(model_path, dataset_dir, calib_file):
    """Check what the model is actually predicting."""
    print("\n" + "="*60)
    print("3. MODEL PREDICTION CHECK")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    try:
        from depth_estimation_net import SiameseStereoNet
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SiameseStereoNet().to(device)
        
        state_dict = torch.load(model_path, map_location=device)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"✓ Loaded model from: {model_path}")
        
        # Load a sample
        dataset = CustomLoaderRectified(
            root_dir=dataset_dir,
            calib_file=calib_file,
            target_size=(480, 640),
            split='val',
            split_ratio=0.8
        )
        
        if len(dataset) == 0:
            print("❌ No validation samples found!")
            return
        
        print(f"✓ Loaded validation dataset with {len(dataset)} samples")
        
        # Get predictions on a few samples
        predictions = []
        ground_truths = []
        
        num_samples = min(10, len(dataset))
        print(f"\nRunning inference on {num_samples} samples...")
        
        with torch.no_grad():
            for i in range(num_samples):
                sample = dataset[i]
                left = sample['left'].unsqueeze(0).to(device)
                right = sample['right'].unsqueeze(0).to(device)
                gt_disp = sample['disparity'].numpy()
                
                pred = model(left, right).squeeze().cpu().numpy()
                
                # Filter valid pixels
                valid_mask = np.isfinite(gt_disp) & (gt_disp > 0)
                
                if valid_mask.sum() > 0:
                    predictions.append(pred[valid_mask])
                    ground_truths.append(gt_disp[valid_mask])
        
        if predictions:
            all_pred = np.concatenate(predictions)
            all_gt = np.concatenate(ground_truths)
            
            print(f"\n📊 Prediction Statistics:")
            print(f"  Ground Truth:")
            print(f"    Min:  {all_gt.min():.2f}  Max:  {all_gt.max():.2f}  Mean:  {all_gt.mean():.2f}")
            print(f"  Predictions:")
            print(f"    Min:  {all_pred.min():.2f}  Max:  {all_pred.max():.2f}  Mean:  {all_pred.mean():.2f}")
            
            # Check for scale mismatch
            pred_mean = all_pred.mean()
            gt_mean = all_gt.mean()
            scale_ratio = pred_mean / gt_mean
            
            print(f"\n🔍 Scale Analysis:")
            print(f"  Prediction/GT ratio: {scale_ratio:.2f}x")
            
            if scale_ratio < 0.5:
                print(f"⚠️  ISSUE: Predictions are {1/scale_ratio:.1f}x too small!")
                print(f"   Model is predicting much lower disparities than ground truth")
            elif scale_ratio > 2.0:
                print(f"⚠️  ISSUE: Predictions are {scale_ratio:.1f}x too large!")
                print(f"   Model is predicting much higher disparities than ground truth")
            else:
                print(f"✓ Prediction scale looks reasonable")
            
            # Check correlation
            correlation = np.corrcoef(all_pred, all_gt)[0, 1]
            print(f"\n📈 Correlation: {correlation:.3f}")
            
            if correlation < 0.3:
                print(f"❌ Very low correlation - model predictions are random!")
            elif correlation < 0.6:
                print(f"⚠️  Low correlation - model is struggling to learn pattern")
            else:
                print(f"✓ Good correlation - model is learning the pattern")
            
            # Compute EPE
            epe = np.abs(all_pred - all_gt).mean()
            print(f"\n🎯 End Point Error (EPE): {epe:.2f} pixels")
            
            if epe > 15:
                print(f"❌ Very high EPE - indicates major problem")
            elif epe > 5:
                print(f"⚠️  High EPE - model needs more training or has scale issue")
            else:
                print(f"✓ Reasonable EPE")
                
    except Exception as e:
        print(f"❌ Error during prediction check: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all diagnostics."""
    print("\n" + "="*60)
    print("COMPREHENSIVE TRAINING DIAGNOSTICS")
    print("="*60)
    
    # Paths
    calib_file = "../camera_scripts/manual_calibration_refined.npz"
    dataset_dir = "../dataset"
    model_path = "../results/v2_finetune/best_finetuned_model.pth"
    
    # Run checks
    calib = check_calibration(calib_file)
    
    if calib is not None:
        disparities = check_dataset_statistics(dataset_dir, calib_file)
        check_model_predictions(model_path, dataset_dir, calib_file)
    
    print("\n" + "="*60)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*60)
    
    print("""
Based on the diagnostics above, look for:

1. ❌ CRITICAL ISSUES:
   - Focal length mismatch > 5%
   - Baseline outside 50-150mm range
   - Max disparity > 192 pixels
   - Prediction/GT scale ratio < 0.5 or > 2.0
   - Correlation < 0.3
   - EPE > 20 pixels

2. ⚠️  WARNING SIGNS:
   - Mean disparity < 5 or > 50 pixels
   - Depth range outside 0.3-10m
   - Correlation < 0.6
   - EPE > 10 pixels

3. ✓ GOOD SIGNS:
   - Disparities in 0-192 range
   - Depth range 0.5-5m
   - Correlation > 0.6
   - EPE < 10 pixels

NEXT STEPS:
- If scale mismatch found: Check disparity conversion formula
- If low correlation: Check rectification quality
- If high EPE but good correlation: Need more training epochs
- If everything looks good but EPE high: Dataset may be too small
    """)


if __name__ == "__main__":
    main()
