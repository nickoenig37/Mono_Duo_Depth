import numpy as np
import os

def save_manual_calibration():
    # === Left Camera (RealSense) - Hardcoded from calibrate_realsense_to_right.py ===
    # The user implied they used the script which has these hardcoded.
    K1 = np.array([
        [606.83831787, 0, 324.79705811], 
        [0, 606.81646729, 242.94799805], 
        [0, 0, 1]
    ])
    D1 = np.zeros(5) # Hardcoded as zeros in the script

    # === Right Camera (OV2710) - User Provided ===
    # From "Right Camera Matrix Guess" / Final Stereo Calibration
    K2 = np.array([
        [412.00110291, 0., 321.64105194],
        [0., 413.79554348, 267.29828363],
        [0., 0., 1.]
    ])
    
    # Distortion D2 was NOT provided in the snippet. 
    # Using zeros is the safest assumption without data, though effectively skips distortion correction for the right cam.
    D2 = np.zeros(5) 

    # === Stereo Extrinsics - User Provided ===
    T = np.array([
        [-0.13127811],
        [ 0.00112071],
        [-0.03945792]
    ])
    
    R = np.array([
        [ 9.99736235e-01, -5.30760928e-04, -2.29603639e-02],
        [-1.74921687e-03,  9.95069315e-01, -9.91665206e-02],
        [ 2.28997873e-02,  9.91805266e-02,  9.94805922e-01]
    ])

    save_path = "manual_calibration_user.npz"
    np.savez(save_path, K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T)
    print(f"Saved manual calibration to: {os.path.abspath(save_path)}")
    print("Contents:")
    print(f"K1:\n{K1}")
    print(f"K2:\n{K2}")
    print(f"T:\n{T}")
    print(f"R:\n{R}")

if __name__ == "__main__":
    save_manual_calibration()
