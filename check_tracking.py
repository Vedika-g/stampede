from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

video = cv2.VideoCapture("crowd.mp4")

success, frame = video.read()

if success:
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        tracker="bytetrack.yaml",
        verbose=True
    )

    result = results[0]

    print("\n--- TRACKING DIAGNOSTIC ---")
    print("Number of detections:", len(result.boxes))
    print("Boxes:", result.boxes)
    print("IDs:", result.boxes.id)
    print("Classes:", result.boxes.cls)
    print("---------------------------")

video.release()