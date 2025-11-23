import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from depth_estimation_net import SiameseStereoNet
from dataset_loaders.stereo_preprocessor import StereoPreprocessor

def run_inference(model_path, dataset_dir, left_path, right_path, device, open_plot=False, save_plot=True, save_path='inference_results.png'):
    """
    Run inference on a set of input images.
    """
    if not os.path.exists(model_path):
        print("ERROR: Model path does not exist: ", model_path)
        return

    # Load trained model
    model = SiameseStereoNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Construct the paths for left and right images
    dataset_dir = Path(dataset_dir)
    left_path = dataset_dir / left_path
    right_path = dataset_dir / right_path

    if not left_path.exists() or not right_path.exists():
        print(f"ERROR: One or more input files do not exist for dataset_dir {dataset_dir} and file names {left_path}, {right_path}")
        return

    # Preprocess the data
    preprocessor = StereoPreprocessor(target_size=(480, 640))
    left_tensor, right_tensor = preprocessor.load_sample(left_path, right_path)
    
    # Run inference on the preprocessed data using the model
    with torch.no_grad():
        output = model(
            left_tensor.unsqueeze(0).to(device), 
            right_tensor.unsqueeze(0).to(device)
        )

    # Process the reuslts, retrieving the predicted disparity map and converting to numpy for visualization
    predicted_disparity_map = output.squeeze().cpu().numpy()

    # Retrieve the input left image tensor for visualization, and do permutations for visualization
    left_input_image = left_tensor.permute(1, 2, 0).numpy()
    left_input_image = np.clip(left_input_image, 0, 1)

    # Plot everything on one figure with 2 subplots
    plt.figure(figsize=(14, 5))

    # Plot 1: Left Input Image
    plt.subplot(1, 2, 1)
    plt.title("Left Input Image After Preprocessing")
    plt.imshow(left_input_image)
    plt.axis('off')

    # Plot 2: Predicted Disparity
    plt.subplot(1, 2, 2)
    plt.title("Predicted Disparity")
    plt.imshow(predicted_disparity_map, cmap='plasma')  #  Use 'plasma' colormap which is good for disparity (bright = close/high shift)
    plt.axis('off')

    if save_plot:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

    if open_plot:
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    """
    Ex Usage. python run_model.py --model_path ../results/v3/best_stereo_model.pth --dataset_dir ../dataset/FlyingThings3D/val --left_path image_clean/left/0000206.png --right_path image_clean/right/0000206.png --no_save_plot
    Ex Usage. python run_model.py --model_path ../results/v3/best_stereo_model.pth --dataset_dir ../dataset/RobotDataset --left_path 000208/left.jpg --right_path 000208/right.jpg --no_save_plot
    """
    parser = argparse.ArgumentParser(description="Run Stereo Depth Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--left_path", type=str, required=True, help="Path to the left image")
    parser.add_argument("--right_path", type=str, required=True, help="Path to the right image")
    parser.add_argument("--no_save_plot", action="store_false", dest="save_plot", help="Don't save the plot")
    parser.set_defaults(save_plot=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_inference(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        left_path=args.left_path,
        right_path=args.right_path,
        device=device,
        open_plot=True,
        save_plot=args.save_plot
    )