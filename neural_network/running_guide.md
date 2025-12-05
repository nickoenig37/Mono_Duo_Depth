# Running Guide: Custom Depth Estimation Pipeline

This guide documents the exact commands to replicate the work done on the Mono-Duo Depth project.

## 1. Calibration & Rectification Check

First, ensure your new calibration (`manual_calibration_refined.npz`) is valid and aligning images correctly.

```bash
cd ~/Documents/Capstone_Bot/capstone_bot/ros2_ws/src/Mono_Duo_Depth/neural_network

# Run the visual check
python3 check_rectification.py \
  --dataset_dir ../dataset/val \
  --calib_file ../camera_scripts/manual_calibration_refined.npz
```

- **Output**: Check `neural_network/visualizations/rectification_check_*.png`.
- **Success**: Horizontal red lines should cross the same features in both Left and Right images.

## 2. Training (Fine-Tuning)

To train the model on your custom dataset using the rectified loader.

```bash
cd ~/Documents/Capstone_Bot/capstone_bot/ros2_ws/src/Mono_Duo_Depth/neural_network

# Train (adjust epochs as needed)
python3 train_fine_tune.py \
  --dataset_dir ../dataset \
  --calib_file ../camera_scripts/manual_calibration_refined.npz \
  --pretrained_model trained_25000_flyingthings_stereo_model.pth \
  --epochs 20
```

- **Output**: Cleaned weights saved in `results/vX_finetune/`.

## 3. Visualization & Validation

To visualize how the model performs on unseen validation data.

```bash
cd ~/Documents/Capstone_Bot/capstone_bot/ros2_ws/src/Mono_Duo_Depth/neural_network

# Visualize (Replace v2_finetune with your actual latest version)
python3 run_vis_custom.py \
  --model_path results/v2_finetune/best_finetuned_model.pth \
  --dataset_dir ../dataset/val \
  --calib_file ../camera_scripts/manual_calibration_refined.npz
```

- **Output**: Images saved in `neural_network/visualizations/v2_finetune/`.
- **What to look for**: The "Predicted Disparity" should resemble the "Ground Truth" (but smoother).

## Notes

- **Visualizations**: All check images are now saved to `neural_network/visualizations/` and are ignored by git.
- **Pipeline Details**: See [preprocessing_pipeline.md](./preprocessing_pipeline.md) for the math/theory.

## 4. Model History

### v2_finetune (20 Epochs)

- **Base**: trained_25000_flyingthings_stereo_model.pth
- **Best Validation EPE**: ~10.08 px (Epoch 12)
- **Final Validation EPE**: ~12.44 px (Overfitting started after Epoch 12)
- **Visualizations**: `neural_network/visualizations/best_finetuned_model/` (Generated from `best_finetuned_model.pth`)
