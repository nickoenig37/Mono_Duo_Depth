import argparse
import torch
import os
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from depth_estimation_net import SiameseStereoNet
from dataset_loaders.custom_loader_rectified import CustomLoaderRectified
from dataset_loaders.masking_utils import DepthMaskGenerator
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

def train_epoch(model, train_loader, val_loader, criterion, optimizer, device, mask_generator, epoch=0):
    model.train()
    total_train_loss = 0
    total_train_epe = 0
    total_train_valid_pixels = 0
    total_train_pixels = 0

    for batch_idx, batch in enumerate(train_loader):
        left = batch["left"].to(device)
        right = batch["right"].to(device)
        target_disp = batch["disparity"].to(device).squeeze(1)

        # Use advanced masking system
        valid_mask, mask_stats = mask_generator.create_combined_mask(
            target_disp,
            use_basic=True,      # Filter NaN/inf values
            use_range=True,      # Filter >10m and <0.3m distances
            use_edge=True,       # Filter edge artifacts  
            use_occlusion=True,  # Filter occlusion boundaries
            use_outlier=True,    # Filter window reflections, etc.
            verbose=(batch_idx == 0 and epoch == 0)  # Print stats on first batch of first epoch
        )
        
        # Track statistics
        total_train_valid_pixels += mask_stats['valid_pixels']
        total_train_pixels += mask_stats['total_pixels']
        
        # Skip batch if no valid pixels
        if valid_mask.sum() == 0:
            if batch_idx == 0:
                print(f"⚠️  Warning: No valid pixels in batch {batch_idx}!")
            continue

        optimizer.zero_grad()
        pred_disp = model(left, right).squeeze(1)
        loss = criterion(pred_disp[valid_mask], target_disp[valid_mask])
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        epe = compute_epe(pred_disp, target_disp)
        total_train_epe += epe

    avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
    avg_train_epe = total_train_epe / len(train_loader) if len(train_loader) > 0 else 0
    train_valid_pct = 100.0 * total_train_valid_pixels / total_train_pixels if total_train_pixels > 0 else 0

    # Validation
    model.eval()
    total_val_loss = 0
    total_val_epe = 0
    total_val_valid_pixels = 0
    total_val_pixels = 0
    
    with torch.no_grad():
        for batch in val_loader:
            left = batch["left"].to(device)
            right = batch["right"].to(device)
            target_disp = batch["disparity"].to(device).squeeze(1)
            
            # Apply same masking to validation
            valid_mask, mask_stats = mask_generator.create_combined_mask(
                target_disp,
                use_basic=True,
                use_range=True,
                use_edge=True,
                use_occlusion=True,
                use_outlier=True,
                verbose=False
            )
            
            total_val_valid_pixels += mask_stats['valid_pixels']
            total_val_pixels += mask_stats['total_pixels']
            
            if valid_mask.sum() == 0:
                continue

            pred_disp = model(left, right).squeeze(1)
            loss = criterion(pred_disp[valid_mask], target_disp[valid_mask])
            total_val_loss += loss.item()
            total_val_epe += compute_epe(pred_disp, target_disp)

    avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
    avg_val_epe = total_val_epe / len(val_loader) if len(val_loader) > 0 else 0
    val_valid_pct = 100.0 * total_val_valid_pixels / total_val_pixels if total_val_pixels > 0 else 0

    return avg_train_loss, avg_train_epe, avg_val_loss, avg_val_epe, train_valid_pct, val_valid_pct

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enable CUDA optimizations for better multi-GPU performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Auto-tune algorithms for your input size
        torch.backends.cudnn.enabled = True
    
    # Check for multiple GPUs and report
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Using device: {device}")
    print(f"Number of GPUs available: {num_gpus}")
    if num_gpus > 1:
        print(f"🚀 Multi-GPU training enabled! Using {num_gpus} GPUs:")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    elif num_gpus == 1:
        print(f"Single GPU training: {torch.cuda.get_device_name(0)}")
    print()

    # Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(base_dir, "results")
    
    # Create output folder
    version_folder = create_new_version_folder(results_dir)
    model_save_path = os.path.join(version_folder, "best_finetuned_model.pth")
    
    # Initialize Advanced Masking System for RealSense Data
    print("\n" + "="*60)
    print("Initializing Advanced Masking System for RealSense Data")
    print("="*60)
    mask_generator = DepthMaskGenerator(
        max_distance=10.0,           # RealSense D435 max reliable range
        min_distance=0.3,            # RealSense D435 min range  
        edge_margin=20,              # Exclude unreliable edge pixels
        occlusion_threshold=1.0,     # Detect occlusion boundaries
        outlier_percentile=99.5,     # Top 0.5% = outliers (windows, reflections)
        baseline=0.1371,             # Your stereo baseline from calibration (meters)
        focal_length=719.53          # Your focal length from calibration (pixels)
    )
    print("✓ Mask generator configured")
    print(f"  - Distance range: {mask_generator.min_distance}m - {mask_generator.max_distance}m")
    print(f"  - Edge margin: {mask_generator.edge_margin}px")
    print(f"  - Occlusion detection: enabled (threshold={mask_generator.occlusion_threshold})")
    print(f"  - Outlier filtering: enabled (top {100-mask_generator.outlier_percentile}%)")
    print("="*60 + "\n")
    
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

    # Adjust batch size for multi-GPU (multiply by number of GPUs)
    batch_size = 4 * max(1, num_gpus)
    # Scale num_workers with number of GPUs for better data loading
    num_workers = 4 * max(1, num_gpus)
    print(f"Batch size adjusted for {num_gpus} GPU(s): {batch_size}")
    print(f"Num workers set to: {num_workers}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Initialize Model
    model = SiameseStereoNet()
    
    # Load Pre-trained Weights BEFORE wrapping with DataParallel
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"Loading pre-trained weights from {args.pretrained_model}")
        state_dict = torch.load(args.pretrained_model, map_location='cpu')
        
        # Handle models saved with DataParallel (keys start with 'module.')
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
    else:
        print("WARNING: No pre-trained model found or provided. Training from scratch!")
    
    # Freeze Encoder Layers (Transfer Learning) BEFORE DataParallel
    # We freeze the base ResNet layers to preserve feature extraction capabilities
    if args.freeze_encoder:
        print("Freezing Encoder Layers...")
        for param in model.encoder_layer_1.parameters(): param.requires_grad = False
        for param in model.encoder_layer_2.parameters(): param.requires_grad = False
        for param in model.encoder_layer_3.parameters(): param.requires_grad = False
    
    # Enable multi-GPU training if available
    if num_gpus > 1:
        print(f"Wrapping model with DataParallel for {num_gpus} GPUs")
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)

    # Optimizer & Scheduler
    # Use a lower learning rate for fine-tuning (reduced from 1e-4 to 5e-5 for better convergence)
    lr = 5e-5 if args.pretrained_model else 1e-3
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = torch.nn.SmoothL1Loss()

    print(f"Starting training with LR={lr}...")
    
    train_losses, train_epes, val_losses, val_epes = [], [], [], []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        t_loss, t_epe, v_loss, v_epe, train_valid_pct, val_valid_pct = train_epoch(
            model, train_loader, val_loader, criterion, optimizer, device, mask_generator, epoch=epoch
        )
        
        print(f"Epoch [{epoch+1}/{args.epochs}]:")
        print(f"  Train - Loss: {t_loss:.4f} | EPE: {t_epe:.4f} | Valid Pixels: {train_valid_pct:.1f}%")
        print(f"  Val   - Loss: {v_loss:.4f} | EPE: {v_epe:.4f} | Valid Pixels: {val_valid_pct:.1f}%")
        
        scheduler.step(v_loss)
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            # Save model without DataParallel wrapper for compatibility
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), model_save_path)
            print(f"Saved new best model (Val Loss: {v_loss:.4f})")

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
