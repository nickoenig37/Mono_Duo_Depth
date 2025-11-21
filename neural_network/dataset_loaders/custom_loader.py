import os
import torch
import numpy as np
import torchvision.transforms as Transforms
from torch.utils.data import Dataset
from PIL import Image

# This is for our own custom datasets that we have collected
class CustomLoader(Dataset):
    def __init__(self, root_dir, save_transformed=False):
        self.root_dir = root_dir
        self.save_transformed = save_transformed
        self.samples = []
        self._monocular_RGB_image_transform = Transforms.Compose([
            Transforms.Resize((480, 640)),
            Transforms.ToTensor()
        ])
        self._depth_sensor_RGB_image_transform = Transforms.Compose([
            Transforms.CenterCrop((500, 600)),
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
                depth_path = os.path.join(sample_path, "depth.npy")

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
                            color_img = Image.fromarray(color_arr.astype(np.uint8), mode='RGB')
                            color_t = self._depth_sensor_RGB_image_transform(color_img)

                            # Transform depth properly
                            depth_t = self._transform_depth_map(depth_arr)

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

        # Apply transforms
        left_img = self._monocular_RGB_image_transform(left_img)
        right_img = self._monocular_RGB_image_transform(right_img)
        color_img = self._depth_sensor_RGB_image_transform(color_img)
        depth_tensor = self._transform_depth_map(depth_arr)

        return {
            "left": left_img,
            "right": right_img,
            "color": color_img,
            "depth": depth_tensor
        }