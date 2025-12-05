import argparse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from dataset_loaders.custom_loader_rectified import CustomLoaderRectified

def check_rectification(dataset_dir, calib_file):
    # Load dataset (train split, 100% ratio to see everything)
    dataset = CustomLoaderRectified(
        root_dir=dataset_dir,
        calib_file=calib_file,
        target_size=(480, 640),
        split='train',
        split_ratio=1.0
    )
    
    if len(dataset) == 0:
        print("No samples found.")
        return

    print(f"Checking rectification on {len(dataset)} samples. Press 'n' for next, 'q' to quit (in the window).")
    print("Or close the window to move to the next one.")

    # Loop through random samples
    indices = np.random.choice(len(dataset), size=10, replace=False)
    
    for idx in indices:
        sample = dataset[idx] # Returns tensors
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        left_img = sample['left'].permute(1, 2, 0).numpy() * std + mean
        right_img = sample['right'].permute(1, 2, 0).numpy() * std + mean
        
        left_img = np.clip(left_img, 0, 1)
        right_img = np.clip(right_img, 0, 1)

        # Create combined image
        # Concatenate horizontally
        combined = np.hstack((left_img, right_img))
        
        # Draw horizontal lines every 30 pixels
        plt.figure(figsize=(12, 6))
        plt.imshow(combined)
        
        for y in range(0, combined.shape[0], 30):
            plt.axhline(y, color='r', linestyle='-', alpha=0.5, linewidth=1)
            
        plt.title(f"Rectified Left vs Right (Sample {idx}) - Check Horizontal Alignment")
        plt.axis('off')
        
        os.makedirs(os.path.join(os.path.dirname(__file__), 'visualizations'), exist_ok=True)
        save_path = os.path.join(os.path.dirname(__file__), 'visualizations', f"rectification_check_{idx}.png")
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--calib_file", required=True)
    args = parser.parse_args()

    check_rectification(args.dataset_dir, args.calib_file)
