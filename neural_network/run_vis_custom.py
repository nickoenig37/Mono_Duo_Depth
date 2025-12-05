import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset_loaders.custom_loader_rectified import CustomLoaderRectified
from depth_estimation_net import SiameseStereoNet

def visualize_prediction(model_path, dataset_dir, calib_file, device):
    # Load One Sample using the Rectified Loader
    dataset = CustomLoaderRectified(
        root_dir=dataset_dir,
        calib_file=calib_file,
        target_size=(480, 640),
        split='val'
    )
    
    if len(dataset) == 0:
        print("No samples found.")
        return

    # Pick a random sample
    idx = np.random.randint(0, len(dataset))
    sample = dataset[idx]
    
    left_tensor = sample['left'].unsqueeze(0).to(device)   # [1, 3, H, W]
    right_tensor = sample['right'].unsqueeze(0).to(device) # [1, 3, H, W]
    gt_disp = sample['disparity'].squeeze().numpy()        # [H, W]

    # Load Model
    model = SiameseStereoNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Run Inference
    with torch.no_grad():
        pred_disp = model(left_tensor, right_tensor).squeeze().cpu().numpy()

    # Prepare Image for Display (Denormalize)
    # Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    left_img = sample['left'].permute(1, 2, 0).numpy() * std + mean
    # Plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title(f"Rectified Left Input\n{os.path.basename(sample['left_path'])}") # More info
    plt.imshow(left_img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f"Predicted Disparity (px)\nMin: {pred_disp.min():.1f}, Max: {pred_disp.max():.1f}")
    plt.imshow(pred_disp, cmap='plasma')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Ground Truth Disparity")
    plt.imshow(gt_disp, cmap='plasma')
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    
    
    # Save with unique name in visualizations folder
    os.makedirs(os.path.join(os.path.dirname(__file__), 'visualizations'), exist_ok=True)
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'visualizations', f"vis_{os.path.basename(sample['left_path']).replace('.npy', '.png')}")
        
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {os.path.abspath(save_path)}")

def visualize_multiple(model_path, dataset_dir, calib_file, device, num_samples=5):
    # Load ALL samples in this specific directory (no split logic, use 100% of it)
    # We cheat the loader by saying split='train' and ratio=1.0 to get everything
    dataset = CustomLoaderRectified(
        root_dir=dataset_dir,
        calib_file=calib_file,
        target_size=(480, 640),
        split='train',
        split_ratio=1.0
    )
    
    if len(dataset) == 0:
        print(f"No samples found in {dataset_dir}")
        return

    print(f"Found {len(dataset)} samples in {dataset_dir}. Visualizing {num_samples} random ones...")
    
    # Random indices
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    # Load Model
    model = SiameseStereoNet().to(device)
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
        
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Normalize stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i, idx in enumerate(indices):
        sample = dataset[idx] # This returns tensors
        # Quick hack: CustomLoaderRectified __getitem__ doesn't return path. 
        # We need to peek at self.samples in the dataset object if we want filenames, 
        # but the loader stores paths in self.samples list.
        # Let's trust the loader index matches.
        sample_info = dataset.samples[idx]
        
        left_tensor = sample['left'].unsqueeze(0).to(device)   
        right_tensor = sample['right'].unsqueeze(0).to(device)
        gt_disp = sample['disparity'].squeeze().numpy()

        with torch.no_grad():
            pred_disp = model(left_tensor, right_tensor).squeeze().cpu().numpy()

        left_img = sample['left'].permute(1, 2, 0).numpy() * std + mean
        left_img = np.clip(left_img, 0, 1)
        
        # Plot
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title(f"Rectified Left: {sample_info['id']}")
        plt.imshow(left_img)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title(f"Predicted Disparity\nMean: {pred_disp.mean():.1f} px")
        plt.imshow(pred_disp, cmap='plasma')
        plt.colorbar()
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Ground Truth Disparity")
        plt.imshow(gt_disp, cmap='plasma')
        plt.colorbar()
        plt.axis('off')

        plt.tight_layout()
        
        plt.tight_layout()
        
        # Determine subdir from model name
        model_name = os.path.basename(model_path).replace('.pth', '')
        output_subdir = os.path.join(os.path.dirname(__file__), 'visualizations', model_name)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Save with unique name in the model-specific subdirectory
        save_name = os.path.join(output_subdir, f"vis_val_{i}_{sample_info['id']}.png")
        plt.savefig(save_name)
        plt.close()
        print(f"Saved {save_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--calib_file", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # visualize_prediction(args.model_path, args.dataset_dir, args.calib_file, device)
    visualize_multiple(args.model_path, args.dataset_dir, args.calib_file, device, num_samples=5)
