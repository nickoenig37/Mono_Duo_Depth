from pathlib import Path
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from depthEstimationNet import DepthEstimationNet
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image


def transform_depth_map(depth_arr, target_size=(480, 640)):
    # normalize to [0,1] to match current model/criterion expectations
    d = depth_arr.astype(np.float32)
    if d.dtype == np.float32 and d.max() > 1.0:
        d = d / 65535.0
    elif depth_arr.dtype == np.uint16:
        d = d / 65535.0
    depth_tensor = torch.from_numpy(d).unsqueeze(0)
    depth_tensor = torch.nn.functional.interpolate(
        depth_tensor.unsqueeze(0), size=target_size, mode='nearest'
    ).squeeze(0)
    return depth_tensor


def run_inference(model_path, sample_dir, device):
    """
    Run inference on a set of input images.
    """
    # Load trained model
    model = DepthEstimationNet().to(device)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.eval()

    input_dir = Path(sample_dir)
    if not input_dir.exists():
        print(f"ERROR: Sample directory does not exist: {input_dir}")
        return

    # Paths to npy files
    left_path = input_dir / "left.npy"
    right_path = input_dir / "right.npy"
    color_path = input_dir / "color.npy"
    depth_left = input_dir / "depth_left.npy"
    depth_path = depth_left if depth_left.exists() else (input_dir / "depth.npy")

    # Define transformations (same as in dataset class)
    target_size = (480, 640)
    rgb_transform = T.Compose([
        T.Resize(target_size, interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])

    # --- Load npy arrays ---
    left_arr = np.load(left_path)
    right_arr = np.load(right_path)
    color_arr = np.load(color_path)
    depth_arr = np.load(depth_path)

    # Convert to PIL for transforms (for RGB images only)
    left_img = Image.fromarray(left_arr.astype(np.uint8))
    right_img = Image.fromarray(right_arr.astype(np.uint8))
    color_img = Image.fromarray(color_arr.astype(np.uint8))

    # Apply transforms
    left_tensor = rgb_transform(left_img).unsqueeze(0).to(device)
    right_tensor = rgb_transform(right_img).unsqueeze(0).to(device)
    color_tensor = rgb_transform(color_img).unsqueeze(0).to(device)
    depth_tensor = transform_depth_map(depth_arr, target_size=target_size).unsqueeze(0).to(device)
    
    # --- Run inference ---
    with torch.no_grad():
        # Model expects (left, center, right)
        output = model(left_tensor, color_tensor, right_tensor)

    # --- Convert to numpy for visualization ---
    output_depth = output.squeeze().cpu().numpy()
    ground_truth_depth = depth_tensor.squeeze().cpu().numpy()

     # Convert transformed color tensor to numpy [H, W, 3]
    color_transformed = color_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    color_transformed = np.clip(color_transformed, 0, 1)  # just in case

    # --- Plot results ---
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Transformed Color (Input to Model)")
    plt.imshow(color_transformed)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Estimated Depth Map")
    plt.imshow(output_depth, cmap='plasma')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Transformed Ground Truth Depth Map")
    plt.imshow(ground_truth_depth, cmap='plasma')
    plt.axis('off')

    plt.tight_layout()
    # Save next to sample folder for convenience
    out_path = input_dir / 'inference.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved figure to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-sample inference")
    parser.add_argument('--model', type=str, default='../best_depth_model.pth')
    parser.add_argument('--sample', type=str, required=True, help='Path to sample folder')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_inference(args.model, args.sample, device)
