import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomLoaderRectified(Dataset):
    def __init__(self, root_dir, calib_file, target_size=(480, 640), split='train', split_ratio=0.8):
        """
        Args:
            root_dir: Directory containing the numbered frame folders (e.g. .../dataset/run_timestamp)
            calib_file: Path to .npz calibration file
            target_size: (H, W) for resizing (Note: Rectification works best at native resolution, resizing after might be better)
            split: 'train' or 'val'
            split_ratio: Ratio of data to use for training
        """
        self.root_dir = root_dir
        self.target_size = target_size
        self.split = split
        
        # Load Calibration
        if not os.path.exists(calib_file):
            raise FileNotFoundError(f"Calibration file not found: {calib_file}")
            
        calib = np.load(calib_file)
        self.K1 = calib['K1']
        self.D1 = calib['D1']
        self.K2 = calib['K2']
        self.D2 = calib['D2']
        self.R = calib['R']
        self.T = calib['T']
        
        # We assume the input images are consistent in size. 
        # We look at the first valid sample to determine size for rectification init
        self.samples = self._find_samples()
        if len(self.samples) == 0:
            print(f"No samples found in {root_dir}")
            self.image_size = (480, 640) # Fallback
        else:
            # Load the first left image to get size
            sample = self.samples[0]
            try:
                # Load color.npy as left image (as per user instruction)
                # It is RGB
                img = np.load(sample['left'])
                h, w, _ = img.shape
                self.image_size = (w, h)
            except Exception as e:
                print(f"Failed to load sample for size determination: {e}")
                self.image_size = (640, 480)

        # Initialize Rectification Maps
        self._init_rectification()

        # Split samples
        # Sort to ensure potential temporal sequences are kept largely together or deterministic
        self.samples.sort(key=lambda x: x['id']) 
        split_idx = int(len(self.samples) * split_ratio)
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
            
        print(f"CustomLoaderRectified ({split}): {len(self.samples)} samples. Image Size: {self.image_size}")

        # Transforms for normalization (Imagenet mean/std is standard, but user mentioned partner used FlyingThings)
        # We will output Tensors in [0,1] range for now, roughly matching what ToTensor does.
        # If the partner logic includes specific normalization, we should match it. 
        # The FlyingThingsLoader uses StereoPreprocessor which does:
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _find_samples(self):
        samples = []
        # Support recursive search or flat list of folders
        # The user's structure: dataset/timestamp_folder/000001/left.npy
        # Or dataset/000001/left.npy?
        # dataset_recorder_node creates "run_dir" then "000001".
        # So root_dir should probably vary. We'll walk.
        
        for subdir, dirs, files in os.walk(self.root_dir):
            # Check for required files
            # User wants: Left = color.npy, Right = right.npy, Depth = depth.npy
            if 'color.npy' in files and 'right.npy' in files and 'depth.npy' in files:
                 samples.append({
                     'left': os.path.join(subdir, 'color.npy'),
                     'right': os.path.join(subdir, 'right.npy'),
                     'depth': os.path.join(subdir, 'depth.npy'),
                     'id': os.path.basename(subdir)
                 })
        return samples

    def _init_rectification(self):
        # Stereo Rectification
        # P1, P2 are the new projection matrices
        # Q is the disparity-to-depth mapping matrix
        # alpha=0 means we crop invalid black borders (zoom in). alpha=1 means we keep everything (black borders).
        # We'll use alpha=0 to avoid training on black artifacts, or maybe 0.5? Let's stick to 0 for valid pixels.
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2, self.image_size, self.R, self.T, 
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        
        self.P1 = P1
        self.P2 = P2
        self.Q = Q
        
        # Effective Focal Length and Baseline from P2
        # P2 = [f, 0, cx, f*Tx]
        # P2[0,3] = f * Tx. Baseline B = |Tx|. 
        # So B = |P2[0,3] / P2[0,0]|
        self.f_rect = P1[0,0]
        self.B_rect = abs(P2[0,3] / P2[0,0])
        
        print(f"Rectified Focal Length: {self.f_rect:.2f}")
        print(f"Rectified Baseline: {self.B_rect:.4f} meters")

        # Compute Undistort/Rectify Maps
        self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(
            self.K1, self.D1, R1, P1, self.image_size, cv2.CV_32FC1
        )
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(
            self.K2, self.D2, R2, P2, self.image_size, cv2.CV_32FC1
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load .npy files
        # Left is color.npy (RGB)
        left_img = np.load(sample['left']) # [H, W, 3]
        right_img = np.load(sample['right']) # [H, W, 3]
        depth_map = np.load(sample['depth']) # [H, W] or [H, W, 1]
        
        # Ensure correct types and shapes
        if left_img.dtype != np.uint8: left_img = left_img.astype(np.uint8)
        if right_img.dtype != np.uint8: right_img = right_img.astype(np.uint8)
        
        # Depth might be float (meters) or uint16 (mm)
        # dataset_recorder says: if depth_save_mm: uint16 (mm), else float32 (m)
        if depth_map.dtype == np.uint16:
            depth_map = depth_map.astype(np.float32) / 1000.0 # Convert to meters
        
        # --- DEPTH CLEANING ---
        # 1. Min/Max Thresholding
        # Realsense often gives noise near 0 or very far.
        min_depth = 0.1 # 10 cm
        max_depth = 20.0 # 20 meters
        mask_valid = (depth_map > min_depth) & (depth_map < max_depth)
        depth_map[~mask_valid] = 0.0 # Set invalid to 0
        
        # 2. Despeckling (Optional but recommended for "terrible" noise)
        # Simple median filter to remove salt-and-pepper noise
        # Only valid if we still have the map in dense form. 
        # But wait, medianBlur doesn't like float32 containing 0s as borders sometimes.
        # Let's simple apply it.
        # depth_map = cv2.medianBlur(depth_map, 3) # Need to be careful with 0s
        
        # Remap (Rectification)
        # INTER_LINEAR for images, INTER_NEAREST for depth
        left_rect = cv2.remap(left_img, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
        depth_rect = cv2.remap(depth_map, self.map1_l, self.map2_l, cv2.INTER_NEAREST)
        
        # Calculate Disparity Ground Truth
        # d = f * B / Z
        valid_mask = (depth_rect > 0)
        disparity = np.zeros_like(depth_rect)
        
        # Avoid division by zero
        # And ensure we don't get massive disparities from tiny depth values that slipped through
        # We cap max disparity to something reasonable (e.g. 192 or 256 depending on image width)
        # But for GT, we just want the real value.
        disparity[valid_mask] = (self.f_rect * self.B_rect) / depth_rect[valid_mask]
        
        # Filter out insane disparities (e.g. > image width)
        max_disp = self.image_size[1] # width
        disparity[disparity > max_disp] = 0

        
        # Setup Tensors
        transform = transforms.Compose([
            transforms.ToTensor(), # HWC [0,255] -> CHW [0.0, 1.0]
            self.normalize
        ])
        
        left_tensor = transform(left_rect)
        right_tensor = transform(right_rect)
        disp_tensor = torch.from_numpy(disparity).float().unsqueeze(0) # [1, H, W]
        
        # Resize if target_size != image_size
        # But wait, resizing *after* disparity calculation scales the image but *changes* the disparity magnitude?
        # If we resize image by factor S, disparity also scales by S.
        # (f becomes f*S, so d becomes d*S).
        # It's safer to train at full res or resize carefully.
        # The FlyingThingsLoader resizes using StereoPreprocessor.
        # For simplicity, if target_size is different, we can resize the tensors.
        if self.target_size != self.image_size:
             # Resize images
             left_tensor = torch.nn.functional.interpolate(left_tensor.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
             right_tensor = torch.nn.functional.interpolate(right_tensor.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
             
             # Resize disparity (and scale values!)
             target_h, target_w = self.target_size
             orig_h, orig_w = self.image_size
             scale_x = target_w / orig_w
             
             disp_tensor = torch.nn.functional.interpolate(disp_tensor.unsqueeze(0), size=self.target_size, mode='nearest').squeeze(0)
             disp_tensor = disp_tensor * scale_x
        
        return {
            "left": left_tensor,
            "right": right_tensor,
            "disparity": disp_tensor
        }
