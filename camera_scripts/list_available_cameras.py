import cv2

def list_available_cameras(max_devices=10):
    available = []
    for index in range(max_devices):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available.append(index)
            cap.release()
    return available

if __name__ == "__main__":
    cameras = list_available_cameras()
    if cameras:
        print("Available camera indices:", cameras)
    else:
        print("No cameras detected.")