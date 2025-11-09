import cv2
import os
import time
from datetime import datetime
import sys

def get_next_index(dataset_dir):
    # Get list of existing numbered folders
    subfolders = [f for f in os.listdir(dataset_dir) if f.isdigit()]
    
    if not subfolders:
        return 1  # start fresh
    
    # Find the largest number and increment
    last_idx = max(int(f) for f in subfolders)
    return last_idx + 1

def main():
    # Generate a unique dataset name based on current date and time
    dataset_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Retrieve the base dataset, which is two parents up
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dataset_dir = os.path.join(parent_dir, "dataset")

    # Retrieve the path for the specific dataset being created based on user input
    dataset_dir = os.path.join(base_dataset_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Get the next index for saving images
    img_index = get_next_index(dataset_dir)
    print(f"Saving data to: {dataset_dir}")
    print(f"Starting from index {img_index:06d}")

    # Open both cameras (0 and 1 are common indices; adjust if needed)
    cam_left = cv2.VideoCapture(2)
    cam_right = cv2.VideoCapture(0)

    # Check if cameras opened successfully
    if not cam_left.isOpened() or not cam_right.isOpened():
        print("Error: One or both cameras could not be opened.")
        return
    
    last_capture_time = time.time()
    while True:
        # Read frames from both cameras
        ret_left, frame_left = cam_left.read()
        ret_right, frame_right = cam_right.read()

        if not ret_left or not ret_right:
            print("Error: Failed to read from one or both cameras.")
            break

        # Combine the two frames side by side and show the video
        combined = cv2.hconcat([frame_left, frame_right])
        cv2.imshow("Dual Camera Feed", combined)

        # Take a picture every second
        current_time = time.time()
        if current_time - last_capture_time >= 1.0:
            folder_name = f"{img_index:06d}"  # zero-padded (000001, 000002, etc.)
            folder_path = os.path.join(dataset_dir, folder_name)

            # Safety check: if folder already exists, terminate to prevent overwrite
            if os.path.exists(folder_path):
                print(f"Error: Attempting to overwrite existing datapoint folder '{folder_name}'.")
                print("Terminating to prevent data loss.")
                sys.exit(1)

            os.makedirs(folder_path)

            # File paths
            left_path = os.path.join(folder_path, "left.jpg")
            right_path = os.path.join(folder_path, "right.jpg")
            meta_path = os.path.join(folder_path, "meta.txt")

            # Save images
            cv2.imwrite(left_path, frame_left)
            cv2.imwrite(right_path, frame_right)

            # Save metadata
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(meta_path, "w") as f:
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Datapoint: {folder_name}\n")

            print(f"Saved datapoint {folder_name} → {left_path}, {right_path}")
            
            img_index += 1
            last_capture_time = current_time

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release cameras and close windows
    cam_left.release()
    cam_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
