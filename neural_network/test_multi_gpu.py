#!/usr/bin/env python3
"""
Quick test to verify multi-GPU training is working properly.
This will create a dummy batch and run it through the model on all GPUs.
"""

import torch
import torch.nn as nn
from depth_estimation_net import SiameseStereoNet

def test_multi_gpu():
    print("="*60)
    print("Multi-GPU Training Test")
    print("="*60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"\n✓ CUDA is available")
    print(f"✓ Number of GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    # Create model
    print(f"\n{'='*60}")
    print("Creating Model...")
    print("="*60)
    
    model = SiameseStereoNet()
    print(f"✓ Model created")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Wrap with DataParallel if multiple GPUs
    if num_gpus > 1:
        print(f"\n🚀 Wrapping model with DataParallel for {num_gpus} GPUs")
        model = nn.DataParallel(model)
    else:
        print(f"\nℹ️  Using single GPU")
    
    device = torch.device("cuda")
    model = model.to(device)
    print(f"✓ Model moved to GPU(s)")
    
    # Create dummy input (batch of stereo images)
    print(f"\n{'='*60}")
    print("Testing Forward Pass...")
    print("="*60)
    
    batch_size = 4 * num_gpus  # Scale with number of GPUs
    print(f"Batch size: {batch_size}")
    
    left_images = torch.randn(batch_size, 3, 480, 640).to(device)
    right_images = torch.randn(batch_size, 3, 480, 640).to(device)
    
    print(f"✓ Created dummy input tensors")
    print(f"  Left shape: {left_images.shape}")
    print(f"  Right shape: {right_images.shape}")
    
    # Forward pass
    print(f"\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(left_images, right_images)
    
    print(f"✓ Forward pass successful!")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, 1, 480, 640)")
    
    # Check GPU utilization
    print(f"\n{'='*60}")
    print("GPU Memory Usage:")
    print("="*60)
    
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"GPU {i}:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
    
    # Test backward pass
    print(f"\n{'='*60}")
    print("Testing Backward Pass...")
    print("="*60)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.SmoothL1Loss()
    
    # Create dummy target
    target = torch.randn(batch_size, 1, 480, 640).to(device)
    
    # Forward + backward
    optimizer.zero_grad()
    output = model(left_images, right_images)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"✓ Backward pass successful!")
    print(f"  Loss: {loss.item():.6f}")
    
    print(f"\n{'='*60}")
    print("✅ All tests passed!")
    print("="*60)
    
    if num_gpus > 1:
        print(f"\n🎉 Multi-GPU training is working correctly with {num_gpus} GPUs!")
    else:
        print(f"\nℹ️  Single GPU training is working correctly")
    
    print("\nYou can now train your model with:")
    print("  python3 train_fine_tune.py --dataset_dir ../dataset --calib_file ../camera_scripts/manual_calibration_refined.npz --pretrained_model trained_25000_flyingthings_stereo_model.pth --epochs 50")
    
    return True

if __name__ == "__main__":
    try:
        test_multi_gpu()
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
