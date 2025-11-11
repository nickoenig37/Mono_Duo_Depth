import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from depthEstimationNet import DepthEstimationNet
from imageDepthDataset import ImageDepthDataset
import os

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

def compute_metrics(pred, target):
    """Computes AbsRel and threshold accuracies."""
    mask = target > 0  # avoid division by zero
    pred = pred[mask]
    target = target[mask]

    abs_rel = torch.mean(torch.abs(pred - target) / target).item()

    ratio = torch.max(pred / target, target / pred)
    delta = torch.mean((ratio < 1.25).float()).item()

    return abs_rel, delta

def train_epoch(model, train_loader, val_loader, criterion, optimizer, device):
    """
    Runs one full epoch of training over the dataset.

    Args:
        model: The depth estimation neural network.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        criterion: The loss function used to measure prediction error.
        optimizer: The optimization algorithm (e.g., Adam, SGD).
        device: 'cuda' or 'cpu' device to perform computations on.

    Returns:
        Average loss value over the epoch.
    """
    model.train()
    total_train_loss = 0
    total_train_absrel = 0
    total_train_delta = 0

    # ---- Training ----
    for batch in train_loader:
        left = batch["left"].to(device)
        right = batch["right"].to(device)
        color = batch["color"].to(device)
        depth = batch["depth"].to(device)

        optimizer.zero_grad()
        output = model(left, right, color)
        loss = criterion(output, depth)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        absrel, delta = compute_metrics(output, depth)
        total_train_absrel += absrel
        total_train_delta += delta

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_absrel = total_train_absrel / len(train_loader)
    avg_train_delta = total_train_delta / len(train_loader)

    # ---- Validation ----
    model.eval()
    total_val_loss = 0
    total_val_absrel = 0
    total_val_delta = 0
    with torch.no_grad():
        for batch in val_loader:
            left = batch["left"].to(device)
            right = batch["right"].to(device)
            color = batch["color"].to(device)
            depth = batch["depth"].to(device)

            output = model(left, right, color)
            total_val_loss += criterion(output, depth).item()

            absrel, delta = compute_metrics(output, depth)
            total_val_absrel += absrel
            total_val_delta += delta

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_absrel = total_val_absrel / len(val_loader)
    avg_val_delta = total_val_delta / len(val_loader)

    return avg_train_loss, avg_train_absrel, avg_train_delta, avg_val_loss, avg_val_absrel, avg_val_delta

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
    model_save_path = os.path.join(version_folder, "best_depth_model.pth")

    # Create a DataLoader for the dataset
    # Base dataset (no augmentation) just to count samples
    base_dataset = ImageDepthDataset(
        root_dir=dataset_dir,
        save_transformed=False,
        target_size=(480, 640),
        normalize_rgb=True,
        prefer_registered_depth=True,
        color_jitter=False,
        hflip_prob=0.0,
    )

    # Compute the split sizes of the validation and training sets
    train_val_split = 0.8
    n_total = len(base_dataset)
    n_train = int(n_total * train_val_split)
    n_val = n_total - n_train

    # Split the dataset based on the split sizes computed
    # Create train/val with different augmentation settings using same indices
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_full = ImageDepthDataset(
        root_dir=dataset_dir,
        save_transformed=False,
        target_size=(480, 640),
        normalize_rgb=True,
        prefer_registered_depth=True,
        color_jitter=True,
        hflip_prob=0.5,
    )
    val_full = ImageDepthDataset(
        root_dir=dataset_dir,
        save_transformed=False,
        target_size=(480, 640),
        normalize_rgb=True,
        prefer_registered_depth=True,
        color_jitter=False,
        hflip_prob=0.0,
    )

    # Subset selection
    train_dataset = torch.utils.data.Subset(train_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_full, val_indices)

    # Create DataLoaders for training and validation sets so we can iterate through them in batches instead of all at once
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    # Initialize the depth estimation model and move it to the device
    model = DepthEstimationNet().to(device)

    # Define the loss function — Mean Squared Error is common for regression tasks
    criterion = torch.nn.MSELoss()

    # Define the optimizer — Adam is good for training most CNNs
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("\nStarting training...\n")

    # Train the model for the set number of epochs
    best_val_loss = float('inf')
    epochs = 1000
    for epoch in range(epochs):
        train_loss, train_absrel, train_delta, val_loss, val_absrel, val_delta = train_epoch(model, train_loader, val_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}]:")
        print(f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        print(f"Train Absolute Relative Error = {train_absrel:.4f}, Train Threshold Accuracy (δ < 1.25) = {train_delta:.4f}")
        print(f"Val Absolute Relative Error = {val_absrel:.4f}, Val Threshold Accuracy (δ < 1.25) = {val_delta:.4f}")

        # Save the model if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model (Val Loss: {val_loss:.4f})")

        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()
