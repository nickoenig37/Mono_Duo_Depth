#!/usr/bin/env python3
"""
Installation verification script for Mono_Duo_Depth project.
Run this script to verify all dependencies are properly installed.
"""

import sys

def check_package(package_name, import_name=None):
    """Check if a package can be imported and return its version."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name:20s} {version}")
        return True
    except ImportError as e:
        print(f"✗ {package_name:20s} NOT FOUND - {e}")
        return False

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        print(f"\n{'='*50}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️  WARNING: CUDA not available. Training will be slow on CPU.")
        print(f"{'='*50}\n")
        return True
    except Exception as e:
        print(f"Error checking CUDA: {e}")
        return False

def check_model_imports():
    """Check if custom model code can be imported."""
    print(f"\n{'='*50}")
    print("Checking Custom Model Imports:")
    print(f"{'='*50}")
    
    try:
        sys.path.insert(0, '/u50/koenin1/ML_PROJECT/Mono_Duo_Depth/neural_network')
        
        from depth_estimation_net import SiameseStereoNet
        print("✓ depth_estimation_net.SiameseStereoNet")
        
        from dataset_loaders.stereo_preprocessor import StereoPreprocessor
        print("✓ dataset_loaders.stereo_preprocessor.StereoPreprocessor")
        
        from dataset_loaders.flying_things_loader import FlyingThingsLoader
        print("✓ dataset_loaders.flying_things_loader.FlyingThingsLoader")
        
        from dataset_loaders.custom_loader import CustomLoader
        print("✓ dataset_loaders.custom_loader.CustomLoader")
        
        # Try instantiating the model
        model = SiameseStereoNet()
        params = sum(p.numel() for p in model.parameters())
        print(f"\n✓ Model instantiation successful!")
        print(f"  Total parameters: {params:,}")
        
        return True
    except Exception as e:
        print(f"✗ Error importing custom modules: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print(f"\n{'='*50}")
    print("Mono_Duo_Depth Installation Verification")
    print(f"{'='*50}\n")
    
    all_ok = True
    
    # Check core packages
    print("Core Packages:")
    print("-" * 50)
    all_ok &= check_package("torch")
    all_ok &= check_package("torchvision")
    all_ok &= check_package("opencv-python", "cv2")
    all_ok &= check_package("Pillow", "PIL")
    all_ok &= check_package("numpy")
    all_ok &= check_package("matplotlib")
    
    # Check CUDA
    all_ok &= check_cuda()
    
    # Check custom model imports
    all_ok &= check_model_imports()
    
    print(f"\n{'='*50}")
    if all_ok:
        print("✓ ALL CHECKS PASSED!")
        print("Your environment is ready for training and inference.")
    else:
        print("✗ SOME CHECKS FAILED")
        print("Please install missing packages using:")
        print("  pip install --user -r requirements.txt")
    print(f"{'='*50}\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
