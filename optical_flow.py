import cv2
import numpy as np

# Open video
video = cv2.VideoCapture("crowd.mp4")

# Read first frame
success, frame = video.read()

if not success:
    print("Could not open video.")
    exit()

# Convert first frame to grayscale
previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

frame_number = 1

while True:

    success, frame = video.read()

    if not success:
        break

    frame_number += 1

    # Convert current frame to grayscale
    current_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    # Convert flow to magnitude and direction
    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    # Average movement in this frame
    average_movement = np.mean(magnitude)

    print(
        f"Frame {frame_number}: "
        f"Average optical flow = "
        f"{average_movement:.2f}"
    )

    # Update previous frame
    previous_gray = current_gray

video.release()

print("\nOptical flow analysis completed!")