import cv2

def main():
    # Open both cameras (0 and 1 are common indices; adjust if needed)
    cam_left = cv2.VideoCapture(2)
    cam_right = cv2.VideoCapture(0)

    # Check if cameras opened successfully
    if not cam_left.isOpened() or not cam_right.isOpened():
        print("Error: One or both cameras could not be opened.")
        return
    
    while True:
        # Read frames from both cameras
        ret_left, frame_left = cam_left.read()
        ret_right, frame_right = cam_right.read()

        if not ret_left or not ret_right:
            print("Error: Failed to read from one or both cameras.")
            break

        # Combine the two frames side by side
        combined = cv2.hconcat([frame_left, frame_right])

        # Show combined video
        cv2.imshow("Dual Camera Feed", combined)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release cameras and close windows
    cam_left.release()
    cam_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
