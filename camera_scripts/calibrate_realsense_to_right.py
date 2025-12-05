import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# === CONFIG ===
BOARD_DIMS = (6, 9) 
SQUARE_SIZE = 0.025
ROOT_DIR = "calibration_images"  # The top folder to search

# === HARDCODED REALSENSE (LEFT) ===
K_rs = np.array([[606.83831787, 0, 324.79705811], [0, 606.81646729, 242.94799805], [0, 0, 1]])
D_rs = np.zeros(5)

# Prepare Object Points
objp = np.zeros((BOARD_DIMS[0] * BOARD_DIMS[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_DIMS[0], 0:BOARD_DIMS[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

objpoints = []
imgpoints_l = []
imgpoints_r = []

# Visualization Function
def visualize_coverage(image_shape, imgpoints, title="Coverage"):
    plt.figure(figsize=(8,6))
    for corners in imgpoints:
        corners = corners.reshape(-1, 2)
        plt.scatter(corners[:,0], corners[:,1], s=2, alpha=0.5)
    plt.xlim(0, image_shape[1])
    plt.ylim(image_shape[0], 0) 
    plt.title(title)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.grid(True)
    plt.show()

print(f"--- Searching recursively in '{ROOT_DIR}' ---")

valid_pairs_count = 0
gray_r_shape = None

# --- RECURSIVE FILE LOADER ---
for root, dirs, files in os.walk(ROOT_DIR):
    # Find all 'left' images in this specific folder (supports png and jpg)
    left_files = [f for f in files if f.startswith('left_') and (f.endswith('.png') or f.endswith('.jpg'))]
    
    for l_filename in left_files:
        # Construct the expected right filename
        # Assuming format is strictly left_XXX -> right_XXX
        r_filename = l_filename.replace('left_', 'right_')
        
        l_path = os.path.join(root, l_filename)
        r_path = os.path.join(root, r_filename)
        
        # Only process if the pair exists
        if os.path.exists(r_path):
            img_l = cv2.imread(l_path)
            img_r = cv2.imread(r_path)
            
            if img_l is None or img_r is None:
                continue

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            
            # Save shape for calibration later
            if gray_r_shape is None:
                gray_r_shape = gray_r.shape[::-1]

            # Find Corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, BOARD_DIMS, None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, BOARD_DIMS, None)

            # Only add if BOTH cameras see it (Your constraint)
            if ret_l and ret_r:
                corners_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                corners_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                
                objpoints.append(objp)
                imgpoints_l.append(corners_l)
                imgpoints_r.append(corners_r)
                valid_pairs_count += 1
                
                # Optional: Print progress every 10 images
                if valid_pairs_count % 10 == 0:
                    print(f"Found {valid_pairs_count} valid pairs so far...")

print(f"\nProcessing {valid_pairs_count} valid pairs total.")

if valid_pairs_count == 0:
    print("No valid checkerboards found in any pair. Exiting.")
    exit()

cv2.destroyAllWindows()

# Plot coverage BEFORE calibration to see if 'Big Data' helped
visualize_coverage(gray_r_shape, imgpoints_r, f"Right Camera Coverage ({valid_pairs_count} pairs)")

# --- Step 2: Calibrate Right Camera INDEPENDENTLY ---
print("Calibrating Right Camera alone...")
ret_r, K2_init, D2_init, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, gray_r_shape, None, None)
print(f"Right Camera Initial RMS: {ret_r}")
print(f"Right Camera Matrix Guess:\n{K2_init}")

# --- Step 3: Stereo Calibrate ---
print("Running Stereo Calibration...")
flags = cv2.CALIB_FIX_INTRINSIC 

ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    K_rs, D_rs,        
    K2_init, D2_init, 
    gray_r_shape,
    flags=flags
)

print(f"\nFINAL RMS Error: {ret}")
print(f"Translation:\n{T}")
print(f"Rotation (R):\n{R}")

if ret < 1.0:
    np.savez("hybrid_calibration_mass.npz", K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T)
    print("Saved successfully.")
else:
    print("Error too high. Do not use.")