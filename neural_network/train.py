import torch
import os
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from depth_estimation_net import SiameseStereoNet
from dataset_loaders.flying_things_loader import FlyingThingsLoader
from run_model_with_depth import run_inference

def create_new_version_folder(base_dir):
    """
    Creates a new folder like v1, v2, ... inside base_dir.
    Returns the path to the created folder.
    """
    os.makedirs(base_dir, exist_ok=True)

    # Find the lowest unused version number
    version = 1
    while os.path.exists(os.path.join(base_dir, f"v{version}")):
        version += 1

    version_folder = os.path.join(base_dir, f"v{version}")
    os.makedirs(version_folder)
    print(f"Created results folder: {version_folder}\n")
    return version_folder

def compute_epe(pred, target):
    """
    Computes End Point Error (EPE).
    This is the average distance (in pixels) between the predicted disparity 
    and the ground truth disparity.
    """
    # We only evaluate on valid pixels (disparity > 0)
    mask = target > 0
    
    if mask.sum() == 0:
        return 0.0

    # Calculate absolute difference in pixels
    diff = torch.abs(pred[mask] - target[mask])
    epe = diff.mean().item()
    return epe

def plot_metrics(epoch, train_losses, train_epes, val_losses, val_epes, save_path):
    """
    Plots training and validation loss and EPE over epochs.
    """
    epochs = list(range(1, epoch + 1))

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()

    # EPE plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_epes, label='Train EPE')
    plt.plot(epochs, val_epes, label='Validation EPE')
    plt.xlabel('Epochs')
    plt.ylabel('End Point Error (EPE)')
    plt.title('Training and Validation EPE')
    plt.legend()
    plt.grid()

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_epoch(model, train_loader, val_loader, criterion, optimizer, device):
    """
    Runs one full epoch of training.
    """
    model.train()
    total_train_loss = 0
    total_train_epe = 0

    for batch_idx, batch in enumerate(train_loader):
        left = batch["left"].to(device)
        right = batch["right"].to(device)
        target_disp = batch["disparity"].to(device).squeeze(1)

        # Create a valid mask to ignore invalid disparity values
        mask = (target_disp > 0) & (target_disp < 192) # 192 is standard max disparity
        
        # If batch has no valid pixels, skip it to avoid NaN
        if mask.sum() == 0:
            print("WARNING: Batch with no valid pixels, skipping.")
            continue

        optimizer.zero_grad()
        
        # Forward Pass
        pred_disp = model(left, right).squeeze(1)
        
        # Calculate Loss ONLY on valid pixels
        loss = criterion(pred_disp[mask], target_disp[mask])
        
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        
        # Calculate Metric (EPE)
        epe = compute_epe(pred_disp, target_disp)
        total_train_epe += epe

    # Average metrics
    num_batches = len(train_loader)
    avg_train_loss = total_train_loss / num_batches
    avg_train_epe = total_train_epe / num_batches

    # ---- Validation ----
    model.eval()
    total_val_loss = 0
    total_val_epe = 0
    
    with torch.no_grad():
        for batch in val_loader:
            left = batch["left"].to(device)
            right = batch["right"].to(device)
            target_disp = batch["disparity"].to(device)
            
            mask = (target_disp > 0) & (target_disp < 192)
            if mask.sum() == 0: continue

            pred_disp = model(left, right)
            
            loss = criterion(pred_disp[mask], target_disp[mask])
            total_val_loss += loss.item()

            epe = compute_epe(pred_disp, target_disp)
            total_val_epe += epe

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_epe = total_val_epe / len(val_loader)

    return avg_train_loss, avg_train_epe, avg_val_loss, avg_val_epe

def activate_cuda():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("CUDA is available. Using GPU for training.")
        print(f"GPU Device       : {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version     : {torch.version.cuda}\n")

        return torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU for training.\n")

        return torch.device("cpu")

def main():
    # Activate CUDA if available
    device = activate_cuda()

    # Define the dataset directory relative to this script
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(base_dir, "dataset")
    results_dir = os.path.join(base_dir, "results")
    if not os.path.isdir(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return
    
    # Create a versioned results folder
    version_folder = create_new_version_folder(results_dir)
    model_save_path = os.path.join(version_folder, "best_stereo_model.pth")

    # Create inference results folder
    inference_save_path = os.path.join(version_folder, "inference_results")
    os.makedirs(inference_save_path, exist_ok=True)

    print("Preparing datasets:")

    # Define the train/val split ratios, and define a set number of training samples to use
    train_val_split = 0.2
    n_train = 100
    n_val = int(n_train * train_val_split / (1 - train_val_split))

    # Create dataset loader for the training/validation sets, choosing the loader based on the dataset we would like to use
    flying_things_dataset_dir = os.path.join(dataset_dir, "FlyingThings3D")
    train_dataset = FlyingThingsLoader( # Using FlyingThings dataset, resizing images for speed/memory efficiency
        root_dir=flying_things_dataset_dir, 
        split='train', 
        target_size=(480, 640),
        max_samples=n_train
    )
    val_dataset = FlyingThingsLoader(
        root_dir=flying_things_dataset_dir, 
        split='val', 
        target_size=(480, 640),
        max_samples=n_val
    )

    # Safety Check
    if len(train_dataset) == 0:
        print("Error: Train dataset is empty. Check your paths.")
        return
    elif len(val_dataset) == 0:
        print("Error: Validation dataset is empty. Check your paths.")
        return

    print(f"Training Samples: {len(train_dataset)} | Validation Samples: {len(val_dataset)}") 

    # Create DataLoaders for training and validation sets so we can iterate through them in batches instead of all at once
    # batch_size=8 is good for 640x480 on most GPUs, this can be reduced to batch_size=4 if memory errors occur
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    # Initialize the Neural Network, based on a Siamese Architecture
    model = SiameseStereoNet().to(device)

    # Define the loss function. We are using SmoothL1Loss instead of MSE as it is less sensitive to outliers (like edges)
    criterion = torch.nn.SmoothL1Loss()

    # Define the optimizer — Adam is good for training most CNNs
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("\nStarting training:\n")

    # Train the model for the set number of epochs. We are using a pre-trained model to train on top of (transfer learning), so 20-50 epochs is usually enough.
    train_losses, train_epes, val_losses, val_epes = [], [], [], []
    best_val_loss = float('inf')
    epochs = 50

    for epoch in range(epochs):
        train_loss, train_epe, val_loss, val_epe = train_epoch(
            model, train_loader, val_loader, criterion, optimizer, device
        )
        
        print(f"Epoch [{epoch+1}/{epochs}]:")
        print(f"Train Loss: {train_loss:.4f} | Train EPE: {train_epe:.4f} px")
        print(f"Val   Loss: {val_loss:.4f} | Val   EPE: {val_epe:.4f} px")

        # Save the model if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model (Val Loss: {val_loss:.4f})")

        print("\n" + "-" * 50 + "\n")

        # Record metrics for plotting, and save a plot of the current past epochs
        train_losses.append(train_loss)
        train_epes.append(train_epe)
        val_losses.append(val_loss)
        val_epes.append(val_epe)
        plot_metrics(
            epoch+1, train_losses, train_epes, val_losses, val_epes,
            os.path.join(version_folder, "training_validation_metrics.png")
        )

        # Run inference on a fixed sample from the validation set after each epoch, saving the results
        run_inference(
            model_path=model_save_path,
            dataset_dir=f"{flying_things_dataset_dir}/val",
            left_path="image_clean/left/0000206.png",
            right_path="image_clean/right/0000206.png",
            disparity_path="disparity/left/0000206.pfm",
            device=device,
            open_plot=False,
            save_plot=True,
            save_path=os.path.join(inference_save_path, f"inference_results_epoch_{epoch+1}.png")
        )

if __name__ == "__main__":
    main()