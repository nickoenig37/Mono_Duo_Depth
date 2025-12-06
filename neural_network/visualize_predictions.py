#!/usr/bin/env python3
"""Visualize model predictions vs ground truth"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from dataset_loaders.custom_loader_rectified import CustomLoaderRectified
from depth_estimation_net import SiameseStereoNet

# Parameters
dataset_dir = "/u50/koenin1/ML_PROJECT/Mono_Duo_Depth/dataset/val/sneakythodelink_all640x480"
calib_file = "/u50/koenin1/ML_PROJECT/Mono_Duo_Depth/camera_scripts/manual_calibration_refined.npz"
model_path = "/u50/koenin1/ML_PROJECT/Mono_Duo_Depth/results/v3_finetune/best_finetuned_model.pth"
output_dir = "/u50/koenin1/ML_PROJECT/Mono_Duo_Depth/results/v3_finetune/visualizations"

os.makedirs(output_dir, exist_ok=True)

print("Loading dataset and model...")
val_dataset = CustomLoaderRectified(
    root_dir=dataset_dir,
    calib_file=calib_file,
    target_size=(480, 640),
    split='val'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseStereoNet()

state_dict = torch.load(model_path, map_location=device)
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Denormalize function for visualization
def denormalize(img_tensor):
    """Convert normalized tensor back to displayable image"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).cpu().numpy()

# Visualize first 5 samples
print(f"Generating visualizations in {output_dir}...")
with torch.no_grad():
    for i in range(min(5, len(val_dataset))):
        sample = val_dataset[i]
        left = sample['left'].unsqueeze(0).to(device)
        right = sample['right'].unsqueeze(0).to(device)
        gt_disp = sample['disparity'].squeeze().cpu().numpy()
        
        pred_disp = model(left, right).squeeze().cpu().numpy()
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Left image
        left_img = denormalize(sample['left'])
        axes[0, 0].imshow(left_img)
        axes[0, 0].set_title('Left Image')
        axes[0, 0].axis('off')
        
        # Right image
        right_img = denormalize(sample['right'])
        axes[0, 1].imshow(right_img)
        axes[0, 1].set_title('Right Image')
        axes[0, 1].axis('off')
        
        # Ground truth disparity
        valid_mask = gt_disp > 0
        gt_vis = gt_disp.copy()
        gt_vis[~valid_mask] = np.nan
        im1 = axes[0, 2].imshow(gt_vis, cmap='jet', vmin=0, vmax=150)
        axes[0, 2].set_title(f'GT Disparity (mean={gt_disp[valid_mask].mean():.1f}px)')
        axes[0, 2].axis('off')
        plt.colorbar(im1, ax=axes[0, 2], fraction=0.046)
        
        # Predicted disparity
        im2 = axes[1, 0].imshow(pred_disp, cmap='jet', vmin=0, vmax=150)
        axes[1, 0].set_title(f'Predicted Disparity (mean={pred_disp[valid_mask].mean():.1f}px)')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
        
        # Error map
        error = np.abs(pred_disp - gt_disp)
        error_vis = error.copy()
        error_vis[~valid_mask] = np.nan
        im3 = axes[1, 1].imshow(error_vis, cmap='hot', vmin=0, vmax=30)
        axes[1, 1].set_title(f'Abs Error (EPE={error[valid_mask].mean():.2f}px)')
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
        
        # Error histogram
        axes[1, 2].hist(error[valid_mask].flatten(), bins=50, range=(0, 50))
        axes[1, 2].set_xlabel('Absolute Error (px)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title(f'Error Distribution')
        axes[1, 2].axvline(error[valid_mask].mean(), color='r', linestyle='--', label=f'Mean={error[valid_mask].mean():.2f}')
        axes[1, 2].axvline(np.median(error[valid_mask]), color='g', linestyle='--', label=f'Median={np.median(error[valid_mask]):.2f}')
        axes[1, 2].legend()
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'sample_{i+1:03d}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_path} (EPE={error[valid_mask].mean():.2f}px)")

print(f"\n✓ Done! Check visualizations in: {output_dir}/")
print(f"  View with: display {output_dir}/sample_001.png")
print(f"  Or copy to local machine and view")
