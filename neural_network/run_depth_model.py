from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from depthEstimationNet import DepthEstimationNet
import torchvision.transforms as Transforms
from PIL import Image
import torch.nn.functional as F

def transform_depth_map(depth_arr):
    depth_arr = depth_arr.astype(np.float32) / 65535.0  # normalize to [0,1]
    depth_tensor = torch.from_numpy(depth_arr).unsqueeze(0)  # (1, H, W)

    # Resize or crop depth if needed (keep NEAREST interp)
    depth_tensor = torch.nn.functional.interpolate(
        depth_tensor.unsqueeze(0),
        size=(480, 640),
        mode='nearest'
    ).squeeze(0)

    return depth_tensor

def run_inference(model_path, device):
    """
    Run inference on a set of input images.
    """
    # Load trained model
    model = DepthEstimationNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Hardcoded datapoint to test on
    input_dir = Path("../dataset/solar_car_lab_3/000008")

    # Paths to npy files
    left_path = input_dir / "left.npy"
    right_path = input_dir / "right.npy"
    color_path = input_dir / "color.npy"
    depth_path = input_dir / "depth.npy"

    # Define transformations (same as in dataset class)
    monocular_image_transform = Transforms.Compose([
        Transforms.Resize((480, 640)),
        Transforms.ToTensor()
    ])
    depth_sensor_RGB_image_transform = Transforms.Compose([
        Transforms.CenterCrop((500, 600)),
        Transforms.Resize((480, 640)),
        Transforms.ToTensor()
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
    left_tensor = monocular_image_transform(left_img).unsqueeze(0).to(device)
    right_tensor = monocular_image_transform(right_img).unsqueeze(0).to(device)
    color_tensor = depth_sensor_RGB_image_transform(color_img).unsqueeze(0).to(device)
    depth_tensor = transform_depth_map(depth_arr).unsqueeze(0).to(device)
    
    # --- Run inference ---
    with torch.no_grad():
        output = model(left_tensor, right_tensor, color_tensor)

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
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "../best_depth_model.pth"
    run_inference(model_path, device)
