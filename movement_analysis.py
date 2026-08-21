from ultralytics import YOLO
import cv2
import math

model = YOLO("yolov8n.pt")

video = cv2.VideoCapture("crowd.mp4")

# Store previous position of each person
previous_positions = {}

frame_number = 0

while True:

    success, frame = video.read()

    if not success:
        break

    frame_number += 1

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    if result.boxes.id is not None:

        ids = result.boxes.id.int().cpu().tolist()
        boxes = result.boxes.xywh.cpu().tolist()

        print(f"\nFrame {frame_number}")

        for person_id, box in zip(ids, boxes):

            x, y, width, height = box

            # Check whether this person appeared in the previous frame
            if person_id in previous_positions:

                previous_x, previous_y = previous_positions[person_id]

                # Calculate movement distance
                distance = math.sqrt(
                    (x - previous_x) ** 2 +
                    (y - previous_y) ** 2
                )

                print(
                    f"Person {person_id}: "
                    f"Position=({x:.1f}, {y:.1f}) "
                    f"Movement={distance:.2f}"
                )

            else:

                print(
                    f"Person {person_id}: "
                    f"First appearance"
                )

            # Save current position
            previous_positions[person_id] = (x, y)

video.release()

print("\nMovement analysis completed!")