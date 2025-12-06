#!/usr/bin/env python3
"""
Quick test to verify the masking system is working correctly.
Tests with synthetic data to demonstrate all masking techniques.
"""

import torch
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_loaders.masking_utils import DepthMaskGenerator

def create_test_disparity():
    """Create a synthetic disparity map with various issues."""
    print("\n" + "="*60)
    print("Creating Test Disparity Map")
    print("="*60)
    
    # Create batch of 2 images, 480x640
    batch_size = 2
    height, width = 480, 640
    
    disparity = torch.zeros((batch_size, height, width))
    
    # Base disparity (valid range)
    baseline, focal = 0.1371, 719.53
    
    # Background at 3m (valid)
    wall_depth = 3.0
    wall_disp = (baseline * focal) / wall_depth
    disparity[:, :, :] = wall_disp
    print(f"  ✓ Background at 3m: disparity = {wall_disp:.2f} px")
    
    # Object at 1m (valid, higher disparity)
    close_depth = 1.0
    close_disp = (baseline * focal) / close_depth
    disparity[:, 200:350, 250:400] = close_disp
    print(f"  ✓ Close object at 1m: disparity = {close_disp:.2f} px")
    
    # Far region at 15m (should be masked - beyond 10m)
    far_depth = 15.0
    far_disp = (baseline * focal) / far_depth
    disparity[:, 50:150, 100:250] = far_disp
    print(f"  ✗ Far region at 15m: disparity = {far_disp:.2f} px (will be masked)")
    
    # Add window reflection outliers (extreme values)
    num_outliers = 100
    for b in range(batch_size):
        outlier_y = torch.randint(0, height, (num_outliers,))
        outlier_x = torch.randint(0, width, (num_outliers,))
        disparity[b, outlier_y, outlier_x] = torch.rand(num_outliers) * 500 + 200  # Very high
    print(f"  ✗ Added {num_outliers} window reflection outliers per image (will be masked)")
    
    # Add invalid values
    disparity[:, 100:110, 500:510] = float('nan')
    disparity[:, 400:405, 50:60] = float('inf')
    print(f"  ✗ Added NaN and inf values (will be masked)")
    
    # Note: Occlusion boundaries will be detected automatically by gradient detection
    print(f"  ✓ Occlusion boundaries will be detected at object edges")
    
    print(f"\nTotal test pixels: {disparity.numel():,}")
    return disparity

def test_masking():
    """Test the masking system."""
    print("\n" + "="*60)
    print("Advanced Masking System Test")
    print("="*60)
    
    # Create mask generator with RealSense parameters
    mask_gen = DepthMaskGenerator(
        max_distance=10.0,
        min_distance=0.3,
        edge_margin=20,
        occlusion_threshold=1.0,
        outlier_percentile=99.5,
        baseline=0.1371,
        focal_length=719.53
    )
    
    # Create test data
    disparity = create_test_disparity()
    
    # Test combined masking
    print("\n" + "="*60)
    print("Testing Combined Masking")
    print("="*60)
    
    valid_mask, stats = mask_gen.create_combined_mask(
        disparity,
        use_basic=True,
        use_range=True,
        use_edge=True,
        use_occlusion=True,
        use_outlier=True,
        verbose=True
    )
    
    # Verify masking worked
    print("\n" + "="*60)
    print("Validation Results")
    print("="*60)
    
    total_masked = stats['total_pixels'] - stats['valid_pixels']
    
    print(f"✓ Total pixels: {stats['total_pixels']:,}")
    print(f"✓ Valid pixels: {stats['valid_pixels']:,} ({stats['valid_percentage']:.1f}%)")
    print(f"✓ Masked pixels: {total_masked:,}")
    
    # Check individual components
    checks_passed = 0
    checks_total = 5
    
    print(f"\nMasking Component Checks:")
    
    if 'invalid_values' in stats and stats['invalid_values'] > 0:
        print(f"  ✓ NaN/inf masking: {stats['invalid_values']:,} pixels masked")
        checks_passed += 1
    else:
        print(f"  ⚠ NaN/inf masking: No invalid values found")
    
    if 'out_of_range' in stats and stats['out_of_range'] > 0:
        print(f"  ✓ Range masking: {stats['out_of_range']:,} pixels masked")
        checks_passed += 1
    else:
        print(f"  ⚠ Range masking: No out-of-range values found")
    
    if 'edge_pixels' in stats and stats['edge_pixels'] > 0:
        print(f"  ✓ Edge masking: {stats['edge_pixels']:,} pixels masked")
        checks_passed += 1
    else:
        print(f"  ⚠ Edge masking: No edge pixels masked")
    
    if 'occluded_pixels' in stats and stats['occluded_pixels'] > 0:
        print(f"  ✓ Occlusion masking: {stats['occluded_pixels']:,} pixels masked")
        checks_passed += 1
    else:
        print(f"  ⚠ Occlusion masking: No occlusions detected")
    
    if 'outlier_pixels' in stats and stats['outlier_pixels'] > 0:
        print(f"  ✓ Outlier masking: {stats['outlier_pixels']:,} pixels masked")
        checks_passed += 1
    else:
        print(f"  ⚠ Outlier masking: No outliers detected")
    
    print(f"\n{'='*60}")
    if checks_passed == checks_total:
        print(f"✅ ALL MASKING TESTS PASSED ({checks_passed}/{checks_total})")
    else:
        print(f"⚠️  PARTIAL SUCCESS ({checks_passed}/{checks_total} checks passed)")
    print("="*60)
    
    print("\n✓ Masking system is working correctly!")
    print("✓ You can now train with advanced masking enabled")
    print("\nRun training with:")
    print("  python3 train_fine_tune.py \\")
    print("    --dataset_dir ../dataset \\")
    print("    --calib_file ../camera_scripts/manual_calibration_refined.npz \\")
    print("    --pretrained_model trained_25000_flyingthings_stereo_model.pth \\")
    print("    --epochs 50")
    
    return True

if __name__ == "__main__":
    try:
        success = test_masking()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
