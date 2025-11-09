import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as Transforms

class ImageDepthDataset(Dataset):
    def __init__(self, root_dir, save_transformed=False):
        self.root_dir = root_dir
        self.save_transformed = save_transformed
        self.samples = []
        self._monocular_image_transform = Transforms.Compose([
            Transforms.Resize((480, 640)),
            Transforms.ToTensor()
        ])
        self._depth_sensor_transform = Transforms.Compose([
            Transforms.CenterCrop((500, 600)),
            Transforms.Resize((480, 640)),
            Transforms.ToTensor()
        ])

        self.load_dataset()

    def load_dataset(self):
        print(f"Loading dataset from: {self.root_dir}")

        self.samples = []
        to_pil = Transforms.ToPILImage()

        for scene_dir in os.listdir(self.root_dir):
            scene_path = os.path.join(self.root_dir, scene_dir)
            if not os.path.isdir(scene_path):
                continue

            for sample_dir in os.listdir(scene_path):
                sample_path = os.path.join(scene_path, sample_dir)
                if not os.path.isdir(sample_path):
                    continue

                left_path = os.path.join(sample_path, "left.jpg")
                right_path = os.path.join(sample_path, "right.jpg")
                color_path = os.path.join(sample_path, "color.jpg")
                depth_path = os.path.join(sample_path, "depth.png")

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
                            # color (uses depth_sensor_transform)
                            color_img = Image.open(color_path).convert("RGB")
                            color_t = self._depth_sensor_transform(color_img)
                            color_pil = to_pil(color_t)
                            color_pil.save(os.path.join(sample_path, "color_transformed.jpg"))

                            # depth (ensure single-channel)
                            depth_img = Image.open(depth_path)
                            if depth_img.mode != 'L':
                                depth_img = depth_img.convert('L')
                            depth_t = self._depth_sensor_transform(depth_img)
                            depth_pil = to_pil(depth_t)
                            depth_pil.save(os.path.join(sample_path, "depth_transformed.png"))        
                        except Exception as e:
                            print(f"Warning: failed to save transformed images for {sample_path}: {e}")
                    else:
                        # Delete transformed images if they exist
                        transformed_color_path = os.path.join(sample_path, "color_transformed.jpg")
                        transformed_depth_path = os.path.join(sample_path, "depth_transformed.png")
                        if os.path.exists(transformed_color_path):
                            os.remove(transformed_color_path)
                        if os.path.exists(transformed_depth_path):
                            os.remove(transformed_depth_path)
                else:
                    print(f"Skipping {sample_path}, missing required files.")

        if len(self.samples) == 0:
            print("No valid datapoints found in dataset directory.")
            return None, None, None, None
        else:
            print(f"Found {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        left_img = Image.open(sample["left"]).convert("RGB")
        right_img = Image.open(sample["right"]).convert("RGB")
        color_img = Image.open(sample["color"]).convert("RGB")
        depth_img = Image.open(sample["depth"])

        # Apply transformations
        left_img = self._monocular_image_transform(left_img)
        right_img = self._monocular_image_transform(right_img)
        color_img = self._depth_sensor_transform(color_img)
        if depth_img.mode != 'L':
            depth_img = depth_img.convert('L')
        depth_tensor = self._depth_sensor_transform(depth_img)

        return {
            "left": left_img,
            "right": right_img,
            "color": color_img,
            "depth": depth_tensor
        }