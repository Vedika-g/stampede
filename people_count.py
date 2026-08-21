from ultralytics import YOLO
import cv2

model = YOLO("yolov8m.pt")

video = cv2.VideoCapture("crowd.mp4")

while True:

    success, frame = video.read()

    if not success:
        break

    # YOLO detection
    results = model.predict(
        frame,
        conf=0.20,
        classes=[0],
        verbose=False
    )

    result = results[0]

    # Count detections
    if result.boxes is not None:
        people_count = len(result.boxes)
    else:
        people_count = 0

    # Draw YOLO boxes
    annotated = result.plot()

    # Draw count
    cv2.rectangle(
        annotated,
        (10, 10),
        (400, 90),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        annotated,
        f"PEOPLE: {people_count}",
        (25, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        4
    )

    cv2.imshow(
        "YOLOv8m - Crowd Count",
        annotated
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()