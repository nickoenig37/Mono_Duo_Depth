from pathlib import Path

import torch
from PIL import Image
import matplotlib.pyplot as plt
from depthEstimationNet import DepthEstimationNet
import torchvision.transforms as Transforms

def run_inference(model_path, device):
    """
    Run inference on a set of input images.
    """
    # Load trained model
    model = DepthEstimationNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    input_dir = Path("../dataset/hallway_data/000118")
    # Find color.jpg, depth.png, left.jpg, right.jpg
    color_path = input_dir / "color.jpg"
    left_path = input_dir / "left.jpg"
    right_path = input_dir / "right.jpg"
    depth_path = input_dir / "depth.png"

    # Define transformations
    monocular_image_transform = Transforms.Compose([
        Transforms.Resize((480, 640)),
        Transforms.ToTensor()
    ])
    depth_sensor_transform = Transforms.Compose([
        Transforms.CenterCrop((500, 600)),
        Transforms.Resize((480, 640)),
        Transforms.ToTensor()
    ])

    # Load and transform images
    left_img = Image.open(left_path).convert("RGB")
    right_img = Image.open(right_path).convert("RGB")
    color_img = Image.open(color_path).convert("RGB")
    depth_img = Image.open(depth_path)
    if depth_img.mode != 'L':
        depth_img = depth_img.convert('L')

    left_tensor = monocular_image_transform(left_img).unsqueeze(0).to(device)
    right_tensor = monocular_image_transform(right_img).unsqueeze(0).to(device)
    color_tensor = depth_sensor_transform(color_img).unsqueeze(0).to(device)
    depth_tensor = depth_sensor_transform(depth_img).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(left_tensor, right_tensor, color_tensor)
    output_depth = output.squeeze().cpu().numpy()
    ground_truth_depth = depth_tensor.squeeze().cpu().numpy()

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Input Color Image")
    plt.imshow(color_img)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title("Estimated Depth Map")
    plt.imshow(output_depth, cmap='plasma')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title("Ground Truth Depth Map")
    plt.imshow(ground_truth_depth, cmap='plasma')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "best_depth_model.pth"
    run_inference(model_path, device)
