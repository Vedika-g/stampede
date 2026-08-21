import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
video = cv2.VideoCapture("crowd.mp4")

if not video.isOpened():
    print("Could not open video.")
    exit()

# Video information
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

print(f"Video size: {width} x {height}")
print(f"FPS: {fps}")

# Output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

output = cv2.VideoWriter(
    "dynamic_heatmap.mp4",
    fourcc,
    fps,
    (width, height)
)

# Keep track of unique people
unique_ids = set()

# Highest simultaneous crowd count
peak_count = 0

frame_number = 0

# Process video using tracking
results = model.track(
    source="crowd.mp4",
    classes=[0],
    tracker="bytetrack.yaml",
    conf=0.25,
    stream=True,
    verbose=False
)

for result in results:

    frame_number += 1

    # Original frame
    frame = result.orig_img.copy()

    # Create density map
    density_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

    current_ids = []

    # Check detections
    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes

        # Get tracking IDs
        if boxes.id is not None:

            ids = boxes.id.cpu().numpy().astype(int)

            current_ids = ids.tolist()

            # Add IDs to unique people set
            for person_id in ids:
                unique_ids.add(person_id)

        # Get bounding boxes
        coordinates = boxes.xyxy.cpu().numpy()

        # Create heatmap points
        for box in coordinates:

            x1, y1, x2, y2 = box.astype(int)

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Add density around person
            cv2.circle(
                density_map,
                (center_x, center_y),
                60,
                1,
                -1
            )

    # Current crowd count
    current_count = len(current_ids)

    # Update peak count
    if current_count > peak_count:
        peak_count = current_count

    # Blur density
    density_map = cv2.GaussianBlur(
        density_map,
        (0, 0),
        30
    )

    # Normalize heatmap
    if density_map.max() > 0:

        density_map = cv2.normalize(
            density_map,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

    density_map = density_map.astype(np.uint8)

    # Convert density to heatmap
    heatmap = cv2.applyColorMap(
        density_map,
        cv2.COLORMAP_JET
    )

    # Overlay heatmap
    frame = cv2.addWeighted(
        frame,
        0.65,
        heatmap,
        0.35,
        0
    )

    # Draw tracking boxes
    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes

        coordinates = boxes.xyxy.cpu().numpy()

        for i, box in enumerate(coordinates):

            x1, y1, x2, y2 = box.astype(int)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

            # Draw ID
            if boxes.id is not None:

                person_id = int(
                    boxes.id[i].item()
                )

                cv2.putText(
                    frame,
                    f"ID {person_id}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

    # Information panel
    cv2.putText(
        frame,
        f"Current People: {current_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Unique People: {len(unique_ids)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Peak Crowd: {peak_count}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "LIVE CROWD HEATMAP",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    # Save frame
    output.write(frame)

    # Display
    cv2.imshow(
        "Dynamic Crowd Heatmap",
        frame
    )

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Cleanup
video.release()
output.release()
cv2.destroyAllWindows()

print("\n==============================")
print("DYNAMIC HEATMAP COMPLETED")
print("==============================")
print(f"Total unique people: {len(unique_ids)}")
print(f"Peak crowd: {peak_count}")
print("Saved as: dynamic_heatmap.mp4")