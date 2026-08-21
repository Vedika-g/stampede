from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8m.pt")

# Open video
video = cv2.VideoCapture("crowd.mp4")

while True:

    success, frame = video.read()

    if not success:
        break

    # Detect people
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    # Count detected people
    if result.boxes is not None:
        people_count = len(result.boxes)
    else:
        people_count = 0

    print(f"People detected: {people_count}")

video.release()

print("YOLO processing completed!")