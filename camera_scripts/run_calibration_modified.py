import cv2
import numpy as np
import glob
import os

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
    print("No images found in calibration_images/. Check path.")
    exit()

print(f"Found {len(left_images)} left images and {len(right_images)} right images.")

# Prepare Object Points
objp = np.zeros((BOARD_DIMS[0] * BOARD_DIMS[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_DIMS[0], 0:BOARD_DIMS[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

objpoints = []
imgpoints_l = []
imgpoints_r = []

print("--- Step 1: Extracting Corners ---")

valid_pairs = 0
for l_path, r_path in zip(left_images, right_images):
    img_l = cv2.imread(l_path)
    img_r = cv2.imread(r_path)
    # Check if images loaded successfully
    if img_l is None or img_r is None:
        print(f"Skipping {l_path} or {r_path} (failed to load)")
        continue
        
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, BOARD_DIMS, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, BOARD_DIMS, None)

    if ret_l and ret_r:
        # Refine
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        objpoints.append(objp)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)
        valid_pairs += 1
    else:
        # Optional: Print which ones failed
        # print(f"Failed corners: {os.path.basename(l_path)}")
        pass

print(f"\nProcessing {valid_pairs} valid pairs...")

if valid_pairs < 10:
    print("Warning: Very few valid pairs (<10). Calibration might be poor.")

image_size = gray_r.shape[::-1]

# --- Step 2: Calibrate Right Camera INDEPENDENTLY ---
print("Calibrating Right Camera alone...")
ret_r, K2_init, D2_init, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, image_size, None, None)
print(f"Right Camera Initial RMS: {ret_r}")
print("D2 Init:", D2_init)

# --- Step 3: Stereo Calibrate with Guesses ---
print("Running Stereo Calibration...")
flags = cv2.CALIB_FIX_INTRINSIC 

ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    K_rs, D_rs,        # Fixed Left
    K2_init, D2_init,  # Guessed Right
    image_size,
    flags=flags
)

print(f"\nFINAL RMS Error: {ret}")
print(f"Translation (T):\n{T}")
print(f"Rotation (R):\n{R}")
print(f"Distortion Left (D1):\n{D1}")
print(f"Distortion Right (D2):\n{D2}")

# Always save, regardless of RMS error, because this is the best we can get.
output_file = "manual_calibration_refined.npz"
np.savez(output_file, K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T)
print(f"Saved refined calibration to: {os.path.abspath(output_file)}")
