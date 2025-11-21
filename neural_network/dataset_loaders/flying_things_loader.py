import os
from torch.utils.data import Dataset
from dataset_loaders.stereo_preprocessor import StereoPreprocessor

class FlyingThingsLoader(Dataset):
    def __init__(self, root_dir, split, target_size=(480, 640), max_samples=None):
        """
        Initializes the FlyingThings3D Dataset Loader.

        root_dir: Path to the datasets folder that includes the train and val folders.
        split: 'train' or 'val' to specify which subset to load.
        target_size: (Height, Width) to resize images to.
        max_samples: Optional limit on number of samples to load for debugging. If None, loads all samples.
        """
        self.root_dir = root_dir
        self.split = split
        self.max_samples = max_samples
        self.samples = []
        self.preprocessor = StereoPreprocessor(target_size=target_size)

        self._load_dataset()

    def _load_dataset(self):
        """
        Loads the dataset file paths into memory for training/validation. The expected path structure is:
        datasets/
            FlyingThings3D/
                train/
                    image_clean/
                        left/
                        right/
                    disparity/
                        left/
                val/
                    image_clean/
                        left/
                        right/
                    disparity/
                        left/
        """

        # Load dataset samples from the directory structure
        dataset_split_path = os.path.join(self.root_dir, self.split)
        if not os.path.exists(dataset_split_path):
            print(f"ERROR: Error while initially loading the {self.split} dataset. Could not find {dataset_split_path}")
            return
        else:
            print(f"Loading FlyingThings3D {self.split} data in: {self.root_dir}")
        
        # Define directories to search through, and check for their existence
        image_dir_left = os.path.join(dataset_split_path, 'image_clean', 'left')
        if not os.path.exists(image_dir_left):
            print(f"ERROR: Error while initially loading the {self.split} dataset. Could not find the left folder: {image_dir_left}")
            return
        
        image_dir_right = os.path.join(dataset_split_path, 'image_clean', 'right')
        if not os.path.exists(image_dir_right):
            print(f"ERROR: Error while initially loading the {self.split} dataset. Could not find the right folder: {image_dir_right}")
            return

        disp_dir = os.path.join(dataset_split_path, 'disparity', 'left')
        if not os.path.exists(disp_dir):
            print(f"ERROR: Error while initially loading the {self.split} dataset. Could not find the disparity folder: {disp_dir}")
            return

        # Iterate through each left RGB image and find corresponding right and disparity files to build each sample
        count = 0
        for subdir, _, files in os.walk(image_dir_left):
            for file in files:
                # If a left image has been found, construct the paths for the right image and disparity map, and create a sample entry
                if file.endswith('.png'):
                    # Get relative path from the left image directory to create the corresponding left, right and disparity paths
                    rel_dir = os.path.relpath(subdir, image_dir_left)
                    
                    # Construct full paths for each image
                    l_path = os.path.join(subdir, file)
                    r_path = os.path.join(image_dir_right, rel_dir, file)
                    d_path = os.path.join(disp_dir, rel_dir, file.replace('.png', '.pfm'))
                    
                    # Only create the sample if all three files exist, otherwise print a warning and skip
                    if os.path.exists(r_path) and os.path.exists(d_path):
                        self.samples.append({
                            "left": l_path,
                            "right": r_path,
                            "disparity": d_path
                        })
                        count += 1
                    else:
                        print(f"WARNING: Missing files for sample, skipping: {l_path}, {r_path}, {d_path}")

                    # If a max sample limit is set, stop scanning when reached
                    if self.max_samples is not None and count >= self.max_samples:
                        return

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset at the given index.
        Returns a dictionary with keys: 'left', 'right', 'disparity'.
        """
        # Get the sample entry from all samples
        sample = self.samples[idx]

        # Load and preprocess the sample using the StereoPreprocessor
        left_tensor, right_tensor, disp_tensor = self.preprocessor.load_sample(
            sample["left"], sample["right"], sample["disparity"]
        )

        return {
            "left": left_tensor,
            "right": right_tensor,
            "disparity": disp_tensor
        }