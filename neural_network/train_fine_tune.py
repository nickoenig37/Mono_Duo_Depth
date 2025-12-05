import argparse
import torch
import os
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from depth_estimation_net import SiameseStereoNet
from dataset_loaders.custom_loader_rectified import CustomLoaderRectified
from run_model_with_depth import run_inference 

def create_new_version_folder(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    version = 1
    while os.path.exists(os.path.join(base_dir, f"v{version}_finetune")):
        version += 1
    version_folder = os.path.join(base_dir, f"v{version}_finetune")
    os.makedirs(version_folder)
    print(f"Created results folder: {version_folder}\n")
    return version_folder

def compute_epe(pred, target):
    mask = target > 0
    if mask.sum() == 0:
        return 0.0
    diff = torch.abs(pred[mask] - target[mask])
    epe = diff.mean().item()
    return epe

def plot_metrics(epoch, train_losses, train_epes, val_losses, val_epes, save_path):
    epochs = list(range(1, epoch + 1))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_epes, label='Train EPE')
    plt.plot(epochs, val_epes, label='Validation EPE')
    plt.xlabel('Epochs')
    plt.ylabel('End Point Error (EPE)')
    plt.title('Training and Validation EPE')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_epoch(model, train_loader, val_loader, criterion, optimizer, device):
    model.train()
    total_train_loss = 0
    total_train_epe = 0

    for batch_idx, batch in enumerate(train_loader):
        left = batch["left"].to(device)
        right = batch["right"].to(device)
        target_disp = batch["disparity"].to(device).squeeze(1)

        mask = (target_disp > 0) & (target_disp < 192) # Max disparity check
        if mask.sum() == 0:
            continue

        optimizer.zero_grad()
        pred_disp = model(left, right).squeeze(1)
        loss = criterion(pred_disp[mask], target_disp[mask])
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        epe = compute_epe(pred_disp, target_disp)
        total_train_epe += epe

    avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
    avg_train_epe = total_train_epe / len(train_loader) if len(train_loader) > 0 else 0

    # Validation
    model.eval()
    total_val_loss = 0
    total_val_epe = 0
    
    with torch.no_grad():
        for batch in val_loader:
            left = batch["left"].to(device)
            right = batch["right"].to(device)
            target_disp = batch["disparity"].to(device).squeeze(1)
            
            mask = (target_disp > 0) & (target_disp < 192)
            if mask.sum() == 0: continue

            pred_disp = model(left, right).squeeze(1)
            loss = criterion(pred_disp[mask], target_disp[mask])
            total_val_loss += loss.item()
            total_val_epe += compute_epe(pred_disp, target_disp)

    avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
    avg_val_epe = total_val_epe / len(val_loader) if len(val_loader) > 0 else 0

    return avg_train_loss, avg_train_epe, avg_val_loss, avg_val_epe

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(base_dir, "results")
    
    # Create output folder
    version_folder = create_new_version_folder(results_dir)
    model_save_path = os.path.join(version_folder, "best_finetuned_model.pth")
    
    print("Initializing CustomLoaderRectified...")
    train_dataset = CustomLoaderRectified(
        root_dir=args.dataset_dir,
        calib_file=args.calib_file,
        target_size=(480, 640),
        split='train',
        split_ratio=0.8
    )
    val_dataset = CustomLoaderRectified(
        root_dir=args.dataset_dir,
        calib_file=args.calib_file,
        target_size=(480, 640),
        split='val',
        split_ratio=0.8
    )
    
    if len(train_dataset) == 0:
        print("Error: No training data found.")
        return

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Initialize Model
    model = SiameseStereoNet().to(device)
    
    # Load Pre-trained Weights
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"Loading pre-trained weights from {args.pretrained_model}")
        state_dict = torch.load(args.pretrained_model, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("WARNING: No pre-trained model found or provided. Training from scratch!")

    # Freeze Encoder Layers (Transfer Learning)
    # We freeze the base ResNet layers to preserve feature extraction capabilities
    if args.freeze_encoder:
        print("Freezing Encoder Layers...")
        for param in model.encoder_layer_1.parameters(): param.requires_grad = False
        for param in model.encoder_layer_2.parameters(): param.requires_grad = False
        for param in model.encoder_layer_3.parameters(): param.requires_grad = False

    # Optimizer & Scheduler
    # Use a lower learning rate for fine-tuning
    lr = 1e-4 if args.pretrained_model else 1e-3
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = torch.nn.SmoothL1Loss()

    print(f"Starting training with LR={lr}...")
    
    train_losses, train_epes, val_losses, val_epes = [], [], [], []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        t_loss, t_epe, v_loss, v_epe = train_epoch(model, train_loader, val_loader, criterion, optimizer, device)
        
        print(f"Epoch [{epoch+1}/{args.epochs}]: T_Loss: {t_loss:.4f} T_EPE: {t_epe:.4f} | V_Loss: {v_loss:.4f} V_EPE: {v_epe:.4f}")
        
        scheduler.step(v_loss)
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model.")

        train_losses.append(t_loss)
        train_epes.append(t_epe)
        val_losses.append(v_loss)
        val_epes.append(v_epe)
        plot_metrics(epoch+1, train_losses, train_epes, val_losses, val_epes, os.path.join(version_folder, "metrics.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Path to the custom dataset root (containing run folders)")
    parser.add_argument("--calib_file", required=True, help="Path to the .npz calibration file")
    parser.add_argument("--pretrained_model", type=str, help="Path to the .pth model trained on FlyingThings")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze_encoder", action='store_true', default=True, help="Freeze encoder layers")
    args = parser.parse_args()
    main(args)
