import torch
import numpy as np
import cv2
import torchvision.transforms as Transforms
from PIL import Image

class StereoPreprocessor:
    def __init__(self, target_size=(480, 640)):
        """
        Handles the loading and transformation of stereo image pairs and their disparity maps.
        target_size: (height, width) tuple for resizing of the images and disparity maps.
        """
        self.target_h, self.target_w = target_size

        # Standard PyTorch normalization for RGB images
        self.rgb_transform = Transforms.Compose([
            Transforms.Resize((self.target_h, self.target_w)),
            Transforms.ToTensor(),
        ])
    
    def _read_pfm(self, file_path):
        """
        Reads a PFM file using OpenCV and sanitizes 'inf' values.
        file_path: Path to the PFM disparity file.
        Returns a numpy array of the disparity map.
        """

        # Read the PFM file using OpenCV
        disparity = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if disparity is None:
            raise Exception(f"Failed to load disparity file: {file_path}")

        # Sanitize 'inf' values by replacing them with 0.0, which represent invalid disparity pixels. 
        disparity = np.nan_to_num(disparity, posinf=0.0, neginf=0.0, nan=0.0)

        return disparity

    def load_sample(self, left_path, right_path, disparity_path):
        """
        Loads a single sample (Left, Right, Disparity).
        Returns: left_tensor, right_tensor, disp_tensor
        """
        # Load the Left/Right images as an RGB PIL Image object
        left_img = Image.open(left_path).convert('RGB')
        right_img = Image.open(right_path).convert('RGB')
        
        # Load the disparity map, which is a numpy matrix of floats
        disp_np = self._read_pfm(disparity_path)

        # Ensure disparity is non-negative, as FlyingThings can have negative values depending on camera setup
        disp_np = np.abs(disp_np)
        
        # Get the original dimensions of the left image, which are needed for scaling the disparity values later
        # This is commonly (540, 960) for FlyingThings3D
        orig_w, orig_h = left_img.size
        
        # Apply RGB transforms (Resizes and converts to Tensor)
        left_tensor = self.rgb_transform(left_img)
        right_tensor = self.rgb_transform(right_img)
        
        # Convert the numpy disparity to a torch tensor
        disp_tensor = torch.from_numpy(disp_np.copy()).unsqueeze(0).float()
        
        # Resize Disparity (Nearest Neighbor to preserve sharp edges)
        disp_tensor = torch.nn.functional.interpolate(
            disp_tensor.unsqueeze(0), 
            size=(self.target_h, self.target_w), 
            mode='nearest'
        ).squeeze(0)
        
        # Scale the disparity values according to the resizing
        # If image shrank by 50%, the pixel shift (disparity) also shrank by 50%. We must multiply by the width ratio to fix this.
        scale_factor = self.target_w / float(orig_w)
        disp_tensor = disp_tensor * scale_factor

        return left_tensor, right_tensor, disp_tensor