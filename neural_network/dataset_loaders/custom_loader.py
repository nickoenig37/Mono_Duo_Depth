import os
import torch
import numpy as np
import torchvision.transforms as Transforms
from torch.utils.data import Dataset
from PIL import Image

# This is for our own custom datasets that we have collected
class CustomLoader(Dataset):
    def __init__(self, root_dir, split, save_transformed=False):
        self.root_dir = root_dir
        self.split = split
        self.save_transformed = save_transformed
        self.samples = []
        self._monocular_RGB_image_transform = Transforms.Compose([
            Transforms.Resize((480, 640)),
            Transforms.ToTensor()
        ])

        self._load_dataset()

    def _transform_depth_map(self, depth_arr):
        depth_arr = depth_arr.astype(np.float32) / 65535.0  # normalize to [0,1]
        depth_tensor = torch.from_numpy(depth_arr).unsqueeze(0)  # (1, H, W)

        # Resize or crop depth if needed (keep NEAREST interp)
        depth_tensor = torch.nn.functional.interpolate(
            depth_tensor.unsqueeze(0),
            size=(480, 640),
            mode='nearest'
        ).squeeze(0)

        return depth_tensor

    def _load_dataset(self):
        dataset_split_path = os.path.join(self.root_dir, self.split)
        if not os.path.exists(dataset_split_path):
            print(f"ERROR: Error while initially loading the {self.split} dataset. Could not find {dataset_split_path}")
            return
        else:
            print(f"Loading FlyingThings3D {self.split} data in: {dataset_split_path}")

        self.samples = []
        for scene_dir in os.listdir(dataset_split_path):
            scene_path = os.path.join(dataset_split_path, scene_dir)
            if not os.path.isdir(scene_path):
                continue

            for sample_dir in os.listdir(scene_path):
                sample_path = os.path.join(scene_path, sample_dir)
                if not os.path.isdir(sample_path):
                    continue

                left_path = os.path.join(sample_path, "color.npy")
                right_path = os.path.join(sample_path, "right.npy")
                depth_path = os.path.join(sample_path, "depth.npy")

                if all(os.path.exists(p) for p in [left_path, right_path, depth_path]):
                    # keep sample record
                    self.samples.append({
                        "left": left_path,
                        "right": right_path,
                        "depth": depth_path
                    })

                    # Apply transforms and save transformed images into the same folder
                    if self.save_transformed:
                        try:
                            # Load and convert npy to PIL for transform
                            depth_arr = np.load(depth_path)

                            # Transform depth properly
                            depth_t = self._transform_depth_map(depth_arr)

                            # Save back as npy tensor
                            np.save(os.path.join(sample_path, "depth_transformed.npy"), depth_t.numpy())       
                        except Exception as e:
                            print(f"Warning: failed to save transformed image for {sample_path}: {e}")
                    else:
                        # Delete transformed npy if they exist
                        for fname in ["depth_transformed.npy"]:
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
        depth_arr = np.load(sample["depth"])

        # Convert to PIL images for torchvision transforms
        left_img = Image.fromarray(left_arr.astype(np.uint8))
        right_img = Image.fromarray(right_arr.astype(np.uint8))

        # Apply transforms
        left_img = self._monocular_RGB_image_transform(left_img)
        right_img = self._monocular_RGB_image_transform(right_img)
        depth_tensor = self._transform_depth_map(depth_arr)

        return {
            "left": left_img,
            "right": right_img,
            "depth": depth_tensor
        }