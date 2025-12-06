# Server Setup Complete! 🎉

## ✅ Installation Summary

Your ML environment on the server is now fully configured and ready to use!

### Installed Packages
- **PyTorch**: 2.8.0+cu128 (with CUDA 12.8 support)
- **TorchVision**: 0.23.0+cu128
- **OpenCV**: 4.12.0
- **NumPy**: 2.0.2
- **Matplotlib**: 3.9.4
- **Pillow**: 11.3.0

### GPU Resources Available 🚀
- **4x NVIDIA L40S GPUs**
- **46GB VRAM per GPU** (184GB total!)
- CUDA 12.8 enabled and working

### Model Information
- **Model**: SiameseStereoNet
- **Total Parameters**: 1,356,033
- **Status**: ✅ Successfully instantiated and verified

---

## 📋 Next Steps

### 1. Transfer Dataset from Your Local Machine

From your **local computer**, run:

```bash
# Transfer training data
rsync -avz --progress /path/to/local/dataset/train/ username@server:/u50/koenin1/ML_PROJECT/Mono_Duo_Depth/dataset/train/

# Transfer validation data
rsync -avz --progress /path/to/local/dataset/val/ username@server:/u50/koenin1/ML_PROJECT/Mono_Duo_Depth/dataset/val/
```

Replace:
- `/path/to/local/dataset/` with your actual dataset path
- `username@server` with your actual server credentials (e.g., `koenin1@ada.mcmaster.ca`)

**Pro Tip**: Use `screen` or `tmux` for large transfers that may take hours!

### 2. Verify Installation Anytime

```bash
cd /u50/koenin1/ML_PROJECT/Mono_Duo_Depth
python3 verify_installation.py
```

### 3. Run Training

Once you have the dataset transferred:

```bash
cd /u50/koenin1/ML_PROJECT/Mono_Duo_Depth/neural_network

# For training on FlyingThings dataset
python3 train.py --train_samples 10000 --val_samples 500 --epochs 50

# Monitor GPU usage in another terminal
watch -n 1 nvidia-smi
```

### 4. Run Inference

```bash
cd /u50/koenin1/ML_PROJECT/Mono_Duo_Depth/neural_network

python3 run_model.py \
  --model_path trained_25000_flyingthings_stereo_model.pth \
  --dataset_dir ../dataset/val/deniseslab_all640x480 \
  --left_path 000001/left.png \
  --right_path 000001/right.png
```

---

## 🔧 Useful Commands

### Check GPU Status
```bash
nvidia-smi
watch -n 1 nvidia-smi  # Auto-refresh every second
```

### Using Screen for Long-Running Jobs
```bash
# Start a new screen session
screen -S training

# Run your training
python3 train.py --epochs 100

# Detach: Press Ctrl+A then D
# Reattach later: screen -r training
# List sessions: screen -ls
```

### Git Workflow (Keeping Code in Sync)
```bash
# On server: Pull latest changes
cd /u50/koenin1/ML_PROJECT/Mono_Duo_Depth
git pull origin main

# Make changes, test, then commit
git add .
git commit -m "Updated training parameters"
git push origin main

# On local machine: Pull changes
git pull origin main
```

### Transfer Results Back to Local Machine
```bash
# From your local machine
rsync -avz --progress username@server:/u50/koenin1/ML_PROJECT/Mono_Duo_Depth/results/ ./results/
```

---

## 🐛 Troubleshooting

### If pip is not found
```bash
export PATH=$HOME/.local/bin:$PATH
```

### To reinstall packages
```bash
pip install --user --upgrade -r requirements.txt
```

### If CUDA out of memory during training
Edit `train.py` and reduce batch size or use fewer samples:
```bash
python3 train.py --train_samples 1000  # Start small
```

### Check which GPU to use
```python
# In your Python code
import torch
device = torch.device('cuda:0')  # Use GPU 0
# Or let PyTorch choose
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## 📂 Dataset Structure Expected

Your dataset should follow this structure:

```
dataset/
├── train/
│   ├── 93_Stroud_1100_ALL640x480/
│   ├── depth_of_thode_all640x480/
│   └── ...
└── val/
    ├── deniseslab_all640x480/
    └── sneakythodelink_all640x480/
```

For FlyingThings3D (if you're using it):
```
dataset/
├── FlyingThings3D/
│   ├── train/
│   │   ├── image_clean/
│   │   └── disparity/
│   └── val/
│       ├── image_clean/
│       └── disparity/
```

---

## 💡 Performance Tips

1. **Use all 4 GPUs** - Consider data parallelism for faster training:
   ```python
   model = torch.nn.DataParallel(model)
   ```

2. **Monitor training** - Use TensorBoard or the built-in plotting in train.py

3. **Save checkpoints** - The training script saves models in results/v{N}/ folders

4. **Background processes** - Always use `screen` or `tmux` for training that takes hours/days

---

## 📞 Quick Reference

- **Project Root**: `/u50/koenin1/ML_PROJECT/Mono_Duo_Depth`
- **Neural Network Code**: `/u50/koenin1/ML_PROJECT/Mono_Duo_Depth/neural_network`
- **Pip Location**: `~/.local/bin/pip`
- **Python**: `python3` (version 3.9.21)
- **GPUs**: 4x NVIDIA L40S (46GB each)

---

**Ready to train! 🚀**

Run `python3 verify_installation.py` anytime to check your setup.
