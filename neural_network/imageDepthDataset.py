import os
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from PIL import Image

class ImageDepthDataset(Dataset):
    def __init__(
        self,
        root_dir,
        save_transformed: bool = False,
        target_size=(480, 640),
        normalize_rgb: bool = True,
        imagenet_mean=(0.485, 0.456, 0.406),
        imagenet_std=(0.229, 0.224, 0.225),
        prefer_registered_depth: bool = True,
        color_jitter: bool = False,
        hflip_prob: float = 0.0,
    ):
        self.root_dir = root_dir
        self.save_transformed = save_transformed
        self.samples = []
        self.target_size = target_size
        self.normalize_rgb = normalize_rgb
        self.imagenet_mean = imagenet_mean
        self.imagenet_std = imagenet_std
        self.prefer_registered_depth = prefer_registered_depth
        self.color_jitter = color_jitter
        self.hflip_prob = float(hflip_prob)

        # Use same transform for all RGB images to keep geometry consistent
        rgb_ops = [
            T.Resize(self.target_size, interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
        ]
        if self.normalize_rgb:
            rgb_ops.append(T.Normalize(self.imagenet_mean, self.imagenet_std))
        self._rgb_transform = T.Compose(rgb_ops)

        self._load_dataset()

    def _transform_depth_map(self, depth_arr):
        """Return depth in [0,1] normalized space (to match current model),
        resized with nearest, plus a valid mask (depth>0 after scaling).
        """
        depth_norm = depth_arr.astype(np.float32)
        # If uint16 mm, scale to [0,1] by 65535 to match existing training scale
        if depth_norm.dtype == np.float32 and depth_norm.max() > 1.0:
            # already float but likely mm; keep legacy behavior by dividing by 65535
            depth_norm = depth_norm / 65535.0
        elif depth_arr.dtype == np.uint16:
            depth_norm = depth_norm / 65535.0

        depth_tensor = torch.from_numpy(depth_norm).unsqueeze(0)  # (1,H,W)
        depth_tensor = torch.nn.functional.interpolate(
            depth_tensor.unsqueeze(0), size=self.target_size, mode='nearest'
        ).squeeze(0)

        valid_mask = (depth_tensor > 0.0).float()
        return depth_tensor, valid_mask

    def _load_dataset(self):
        print(f"Loading dataset from: {self.root_dir}")

        self.samples = []
        for scene_dir in os.listdir(self.root_dir):
            scene_path = os.path.join(self.root_dir, scene_dir)
            if not os.path.isdir(scene_path):
                continue

            for sample_dir in os.listdir(scene_path):
                sample_path = os.path.join(scene_path, sample_dir)
                if not os.path.isdir(sample_path):
                    continue

                left_path = os.path.join(sample_path, "left.npy")
                right_path = os.path.join(sample_path, "right.npy")
                color_path = os.path.join(sample_path, "color.npy")
                # prefer registered depth if available
                depth_left_path = os.path.join(sample_path, "depth_left.npy")
                depth_path = depth_left_path if (self.prefer_registered_depth and os.path.exists(depth_left_path)) else os.path.join(sample_path, "depth.npy")

                if all(os.path.exists(p) for p in [left_path, right_path, color_path, depth_path]):
                    # keep sample record
                    self.samples.append({
                        "left": left_path,
                        "right": right_path,
                        "color": color_path,
                        "depth": depth_path
                    })

                    # Apply transforms and save transformed images into the same folder
                    if self.save_transformed:
                        try:
                            # Load and convert npy to PIL for transform
                            color_arr = np.load(color_path)
                            depth_arr = np.load(depth_path)

                            # Convert color to PIL and transform it
                            color_img = Image.fromarray(color_arr.astype(np.uint8))
                            color_t = self._rgb_transform(color_img)

                            # Transform depth properly
                            depth_t, _ = self._transform_depth_map(depth_arr)

                            # Save back as npy tensors
                            np.save(os.path.join(sample_path, "color_transformed.npy"), color_t.permute(1, 2, 0).numpy())
                            np.save(os.path.join(sample_path, "depth_transformed.npy"), depth_t.numpy())       
                        except Exception as e:
                            print(f"Warning: failed to save transformed images for {sample_path}: {e}")
                    else:
                        # Delete transformed npy if they exist
                        for fname in ["color_transformed.npy", "depth_transformed.npy"]:
                            fpath = os.path.join(sample_path, fname)
                            if os.path.exists(fpath):
                                os.remove(fpath)
                else:
                    print(f"Skipping {sample_path}, missing required files.")

        if len(self.samples) == 0:
            print("No valid datapoints found in dataset directory.")
        else:
            print(f"Found {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load npy arrays
        left_arr = np.load(sample["left"])
        right_arr = np.load(sample["right"])
        color_arr = np.load(sample["color"])
        depth_arr = np.load(sample["depth"])

        # Convert to PIL images for torchvision transforms
        left_img = Image.fromarray(left_arr.astype(np.uint8))
        right_img = Image.fromarray(right_arr.astype(np.uint8))
        color_img = Image.fromarray(color_arr.astype(np.uint8))

        # Apply consistent transforms
        left_t = self._rgb_transform(left_img)
        right_t = self._rgb_transform(right_img)
        color_t = self._rgb_transform(color_img)
        depth_tensor, valid_mask = self._transform_depth_map(depth_arr)

        # Optional simple augmentation: color jitter + horizontal flip
        if self.color_jitter:
            jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)
            left_t = jitter(left_t)
            right_t = jitter(right_t)
            color_t = jitter(color_t)
        if self.hflip_prob > 0.0 and torch.rand(1).item() < self.hflip_prob:
            left_t = torch.flip(left_t, dims=[2])
            right_t = torch.flip(right_t, dims=[2])
            color_t = torch.flip(color_t, dims=[2])
            depth_tensor = torch.flip(depth_tensor, dims=[2])
            valid_mask = torch.flip(valid_mask, dims=[2])

        return {
            "left": left_t,
            "right": right_t,
            "color": color_t,
            "depth": depth_tensor,
            "valid_mask": valid_mask,
        }