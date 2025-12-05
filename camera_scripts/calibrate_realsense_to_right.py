import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# === CONFIG ===
BOARD_DIMS = (6, 9) # Your square board
SQUARE_SIZE = 0.025

# === HARDCODED REALSENSE (LEFT) ===
K_rs = np.array([[606.83831787, 0, 324.79705811], [0, 606.81646729, 242.94799805], [0, 0, 1]])
D_rs = np.zeros(5)

# Load Images
left_images = sorted(glob.glob('calibration_images/left_*.png'))
right_images = sorted(glob.glob('calibration_images/right_*.png'))

if len(left_images) == 0:
    print("No images found. Check path.")
    exit()

# Prepare Object Points
objp = np.zeros((BOARD_DIMS[0] * BOARD_DIMS[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_DIMS[0], 0:BOARD_DIMS[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

objpoints = []
imgpoints_l = []
imgpoints_r = []

import matplotlib.pyplot as plt

def visualize_coverage(image_shape, imgpoints, title="Coverage"):
    plt.figure(figsize=(8,6))
    for corners in imgpoints:
        corners = corners.reshape(-1, 2)
        plt.scatter(corners[:,0], corners[:,1], s=2, alpha=0.5)
    plt.xlim(0, image_shape[1])
    plt.ylim(image_shape[0], 0) # Invert Y to match image coords
    plt.title(title)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.grid(True)
    plt.show()

print("--- Step 1: verifying Corners (Check for Symmetry Issues) ---")

for l_path, r_path in zip(left_images, right_images):
    img_l = cv2.imread(l_path)
    img_r = cv2.imread(r_path)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, BOARD_DIMS, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, BOARD_DIMS, None)

    if ret_l and ret_r:
        # Refine
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        # VISUALIZATION: Draw corners to verify order
        # Red is the first corner (origin). If Red is top-left in one and bottom-right in other, THAT is your error.
        cv2.drawChessboardCorners(img_l, BOARD_DIMS, corners_l, ret_l)
        cv2.drawChessboardCorners(img_r, BOARD_DIMS, corners_r, ret_r)
        
        # Stack images to show comparison
        vis = np.hstack((img_l, img_r))
        cv2.namedWindow('Check Corners - Press Space', cv2.WINDOW_NORMAL)
        cv2.imshow('Check Corners - Press Space', vis)
        key = cv2.waitKey(100) # Wait 100ms
        
        objpoints.append(objp)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)

print(f"\nProcessing {len(objpoints)} valid pairs...")
cv2.destroyAllWindows()

visualize_coverage(gray_r.shape, imgpoints_r, "Right Camera Coverage")

# --- Step 2: Calibrate Right Camera INDEPENDENTLY ---
# We need a good guess for K2 before doing stereo, or it collapses to Identity.
print("Calibrating Right Camera alone...")
ret_r, K2_init, D2_init, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, gray_r.shape[::-1], None, None)
print(f"Right Camera Initial RMS: {ret_r}")
print(f"Right Camera Matrix Guess:\n{K2_init}")

# --- Step 3: Stereo Calibrate with Guesses ---
print("Running Stereo Calibration...")
# We use CALIB_USE_INTRINSIC_GUESS for Right, and FIX_INTRINSIC for Left
flags = cv2.CALIB_FIX_INTRINSIC 

ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    K_rs, D_rs,        # Fixed Left
    K2_init, D2_init,  # Guessed Right (from Step 2)
    gray_r.shape[::-1],
    flags=flags
)

print(f"\nFINAL RMS Error: {ret}")
print(f"Translation:\n{T}")
print(f"Rotation (R):\n{R}")

if ret < 1.0:
    np.savez("hybrid_calibration.npz", K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T)
    print("Saved successfully.")
else:
    print("Error too high. Do not use.")