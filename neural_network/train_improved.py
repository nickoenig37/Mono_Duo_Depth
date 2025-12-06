#!/usr/bin/env python3
"""
Improved Fine-Tuning Script with Regularization
- Dropout in model architecture
- Weight decay for L2 regularization
- Learning rate scheduling
- Early stopping
- Gradient clipping
"""

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

def plot_metrics(epoch, train_losses, train_epes, val_losses, val_epes, learning_rates, save_path):
    epochs = list(range(1, epoch + 1))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss plot
    axes[0].plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
    axes[0].plot(epochs, val_losses, label='Validation Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # EPE plot
    axes[1].plot(epochs, train_epes, label='Train EPE', marker='o', markersize=3)
    axes[1].plot(epochs, val_epes, label='Validation EPE', marker='s', markersize=3)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('End Point Error (EPE)')
    axes[1].set_title('Training and Validation EPE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[2].plot(epochs, learning_rates, label='Learning Rate', marker='o', markersize=3, color='green')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def train_epoch(model, train_loader, val_loader, criterion, optimizer, device, mask_generator, epoch=0, max_grad_norm=1.0):
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
        
        # Gradient clipping to prevent exploding gradients
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()

        total_train_loss += loss.item()
        epe = compute_epe(pred_disp, target_disp)
        total_train_epe += epe

    avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
    avg_train_epe = total_train_epe / len(train_loader) if len(train_loader) > 0 else 0
    train_valid_pct = 100.0 * total_train_valid_pixels / total_train_pixels if total_train_pixels > 0 else 0

    # Validation phase
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
            epe = compute_epe(pred_disp, target_disp)
            total_val_epe += epe

    avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
    avg_val_epe = total_val_epe / len(val_loader) if len(val_loader) > 0 else 0
    val_valid_pct = 100.0 * total_val_valid_pixels / total_val_pixels if total_val_pixels > 0 else 0

    return avg_train_loss, avg_train_epe, avg_val_loss, avg_val_epe, train_valid_pct, val_valid_pct

def main(args):
    # Enable cuDNN benchmarking for performance
    torch.backends.cudnn.benchmark = True
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Multi-GPU setup
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    if num_gpus > 1:
        print("🚀 Multi-GPU training enabled! Using {} GPUs:".format(num_gpus))
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()
    
    # Create results folder
    version_folder = create_new_version_folder(args.output_dir)
    model_save_path = os.path.join(version_folder, "best_finetuned_model.pth")
    
    # Initialize Advanced Masking System
    print("=" * 60)
    print("Initializing Advanced Masking System for RealSense Data")
    print("=" * 60)
    
    # Use rectified calibration parameters from the dataset
    # These will be properly computed by CustomLoaderRectified
    # For now, use defaults that will be overridden
    mask_generator = DepthMaskGenerator(
        baseline=0.1371,      # Will match rectified baseline
        focal_length=719.53,  # Will match rectified focal
        max_distance=10.0,
        edge_margin=20,
        occlusion_threshold=1.0,
        outlier_percentile=99.5
    )
    
    print(f"  Min disparity (at {mask_generator.max_distance}m): {mask_generator.min_disparity:.2f} pixels")
    print(f"  Max disparity (at {mask_generator.min_distance}m): {mask_generator.max_disparity:.2f} pixels")
    print("✓ Mask generator configured")
    print(f"  - Distance range: {mask_generator.min_distance}m - {mask_generator.max_distance}m")
    print(f"  - Edge margin: {mask_generator.edge_margin}px")
    print(f"  - Occlusion detection: enabled (threshold={mask_generator.occlusion_threshold})")
    print(f"  - Outlier filtering: enabled (top {100-mask_generator.outlier_percentile:.1f}%)")
    print("=" * 60)
    print()

    # Load Dataset
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

    # Adjust batch size for multi-GPU
    batch_size = 4 * max(1, num_gpus)
    num_workers = 4 * max(1, num_gpus)
    print(f"Batch size adjusted for {num_gpus} GPU(s): {batch_size}")
    print(f"Num workers set to: {num_workers}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Initialize Model with Dropout
    print(f"\nInitializing model with dropout_p={args.dropout_p}...")
    model = SiameseStereoNet(dropout_p=args.dropout_p)
    
    # Load Pre-trained Weights
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"Loading pre-trained weights from {args.pretrained_model}")
        state_dict = torch.load(args.pretrained_model, map_location='cpu')
        
        # Handle models saved with DataParallel
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)  # strict=False to handle dropout layers
    else:
        print("WARNING: No pre-trained model found. Training from scratch!")
    
    # Freeze Encoder Layers (Transfer Learning)
    if args.freeze_encoder:
        print("Freezing Encoder Layers...")
        for param in model.encoder_layer_1.parameters(): param.requires_grad = False
        for param in model.encoder_layer_2.parameters(): param.requires_grad = False
        for param in model.encoder_layer_3.parameters(): param.requires_grad = False
    
    # Enable multi-GPU training
    if num_gpus > 1:
        print(f"Wrapping model with DataParallel for {num_gpus} GPUs")
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)

    # Optimizer with Weight Decay (L2 Regularization)
    print(f"\nOptimizer configuration:")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Gradient clipping: {args.max_grad_norm}")
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=args.lr_patience,
        min_lr=1e-7
    )
    print(f"  LR scheduler: ReduceLROnPlateau (factor=0.5, patience={args.lr_patience})")
    
    criterion = torch.nn.SmoothL1Loss()

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Early stopping patience: {args.early_stop_patience} epochs")
    print()
    
    train_losses, train_epes, val_losses, val_epes, learning_rates = [], [], [], [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(args.epochs):
        t_loss, t_epe, v_loss, v_epe, train_valid_pct, val_valid_pct = train_epoch(
            model, train_loader, val_loader, criterion, optimizer, device, 
            mask_generator, epoch=epoch, max_grad_norm=args.max_grad_norm
        )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] (LR={current_lr:.2e}):")
        print(f"  Train - Loss: {t_loss:.4f} | EPE: {t_epe:.4f} | Valid Pixels: {train_valid_pct:.1f}%")
        print(f"  Val   - Loss: {v_loss:.4f} | EPE: {v_epe:.4f} | Valid Pixels: {val_valid_pct:.1f}%")
        
        # Step scheduler and check if LR changed
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(v_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr < old_lr:
            print(f"  📉 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        # Early stopping logic
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            epochs_no_improve = 0
            # Save model without DataParallel wrapper
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), model_save_path)
            print(f"✓ Saved new best model (Val Loss: {v_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")
            
            if epochs_no_improve >= args.early_stop_patience:
                print(f"\n⚠️  Early stopping triggered! No improvement for {args.early_stop_patience} epochs.")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break

        train_losses.append(t_loss)
        train_epes.append(t_epe)
        val_losses.append(v_loss)
        val_epes.append(v_epe)
        
        plot_metrics(epoch+1, train_losses, train_epes, val_losses, val_epes, 
                    learning_rates, os.path.join(version_folder, "metrics.png"))
        print()
    
    print("=" * 60)
    print("Training Complete!")
    print(f"Best model saved to: {model_save_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Metrics plot saved to: {os.path.join(version_folder, 'metrics.png')}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved Fine-Tuning with Regularization")
    
    # Data arguments
    parser.add_argument("--dataset_dir", type=str, default="../dataset/train",
                       help="Path to training dataset")
    parser.add_argument("--calib_file", type=str, 
                       default="../camera_scripts/manual_calibration_refined.npz",
                       help="Path to calibration file")
    parser.add_argument("--output_dir", type=str, default="../results",
                       help="Output directory for results")
    
    # Model arguments
    parser.add_argument("--pretrained_model", type=str, 
                       default="trained_25000_flyingthings_stereo_model.pth",
                       help="Path to pre-trained model")
    parser.add_argument("--freeze_encoder", action='store_true', default=True,
                       help="Freeze encoder layers")
    parser.add_argument("--dropout_p", type=float, default=0.2,
                       help="Dropout probability (0.0-0.5, default: 0.2)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50,
                       help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay (L2 regularization)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm for clipping (0 to disable)")
    
    # Regularization arguments
    parser.add_argument("--lr_patience", type=int, default=5,
                       help="Patience for LR scheduler")
    parser.add_argument("--early_stop_patience", type=int, default=15,
                       help="Patience for early stopping")
    
    args = parser.parse_args()
    main(args)
