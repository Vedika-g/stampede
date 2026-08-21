from ultralytics import YOLO
import cv2

# Load stronger YOLO model
model = YOLO("yolov8m.pt")

# Open your video
video = cv2.VideoCapture("crowd.mp4")

if not video.isOpened():
    print("ERROR: Could not open crowd.mp4")
    exit()

frame_number = 0

while True:
    success, frame = video.read()

    if not success:
        break

    frame_number += 1

    # Detect ONLY people
    results = model(
        frame,
        classes=[0],       # COCO class 0 = person
        conf=0.20,
        verbose=False
    )

    result = results[0]

    # Count all detected people in this frame
    if result.boxes is not None:
        people_count = len(result.boxes)
    else:
        people_count = 0

    # Draw bounding boxes
    annotated_frame = result.plot()

    # Display total people count
    cv2.putText(
        annotated_frame,
        f"People Present: {people_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    # Display frame number
    cv2.putText(
        annotated_frame,
        f"Frame: {frame_number}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Show video
    cv2.imshow("YOLOv8m - People Detection", annotated_frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

print("\nDetection completed!")