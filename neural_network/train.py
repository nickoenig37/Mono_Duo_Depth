import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from depthEstimationNet import DepthEstimationNet
from imageDepthDataset import ImageDepthDataset
import os

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

    avg_train_loss = total_train_loss / len(train_loader)

    # ---- Validation ----
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            left = batch["left"].to(device)
            right = batch["right"].to(device)
            color = batch["color"].to(device)
            depth = batch["depth"].to(device)

            output = model(left, right, color)
            total_val_loss += criterion(output, depth).item()

    avg_val_loss = total_val_loss / len(val_loader)

    return avg_train_loss, avg_val_loss

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
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    if not os.path.isdir(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return

    # Create a DataLoader for the dataset
    dataset = ImageDepthDataset(root_dir=dataset_dir, save_transformed=True)

    # Compute the split sizes of the validation and training sets
    train_val_split = 0.8
    n_total = len(dataset)
    n_train = int(n_total * train_val_split)
    n_val = n_total - n_train

    # Split the dataset based on the split sizes computed
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    # Create DataLoaders for training and validation sets so we can iterate through them in batches instead of all at once
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Initialize the depth estimation model and move it to the device
    model = DepthEstimationNet().to(device)

    # Define the loss function — Mean Squared Error is common for regression tasks
    criterion = torch.nn.MSELoss()

    # Define the optimizer — Adam is good for training most CNNs
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model for the set number of epochs
    best_val_loss = float('inf')
    epochs = 1000
    for epoch in range(epochs):
        train_loss, val_loss = train_epoch(model, train_loader, val_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}]: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Save the model if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_depth_model.pth")
            print(f"Saved new best model (Val Loss: {val_loss:.4f})")

if __name__ == "__main__":
    main()
