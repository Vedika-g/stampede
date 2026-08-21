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

    if result.boxes is not None:
        count = len(result.boxes)
    else:
        count = 0

    print("People:", count)

    annotated = result.plot()

    cv2.imshow("YOLO Test", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()