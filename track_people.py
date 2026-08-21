from ultralytics import YOLO
import cv2

model = YOLO("yolov8m.pt")

video = cv2.VideoCapture("crowd.mp4")

while True:

    success, frame = video.read()

    if not success:
        break

    results = model.predict(
        frame,
        conf=0.20,
        classes=[0],
        verbose=False
    )

    result = results[0]

    # Count detected people
    if result.boxes is not None:
        people_count = len(result.boxes)
    else:
        people_count = 0

    # Draw YOLO bounding boxes
    annotated = result.plot()

    # Draw BIG count
    cv2.rectangle(
        annotated,
        (10, 10),
        (330, 70),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        annotated,
        f"PEOPLE COUNT: {people_count}",
        (20, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        3
    )

    cv2.imshow(
        "YOLOv8m Crowd Detection",
        annotated
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()