from ultralytics import YOLO
import cv2
import numpy as np

# ============================================================
# 1. LOAD YOLO MODEL
# ============================================================

model = YOLO("yolov8n.pt")

# ============================================================
# 2. OPEN VIDEO
# ============================================================

video = cv2.VideoCapture("crowd.mp4")

if not video.isOpened():
    print("ERROR: Could not open crowd.mp4")
    exit()

# ============================================================
# 3. VIDEO INFORMATION
# ============================================================

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video resolution: {width} x {height}")

# ============================================================
# 4. HEATMAP
# ============================================================

heatmap = np.zeros(
    (height, width),
    dtype=np.float32
)

# Heatmap settings
DECAY = 0.92
HEAT_RADIUS = 50
BLUR_SIGMA = 25

# ============================================================
# 5. PROCESS VIDEO
# ============================================================

frame_number = 0

while True:

    success, frame = video.read()

    if not success:
        break

    frame_number += 1

    # --------------------------------------------------------
    # YOLO DETECTION + TRACKING
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        tracker="bytetrack.yaml",
        conf=0.25,
        verbose=False
    )

    result = results[0]

    people_count = 0

    # --------------------------------------------------------
    # FADE OLD HEAT
    # --------------------------------------------------------

    heatmap *= DECAY

    # --------------------------------------------------------
    # DETECT PEOPLE
    # --------------------------------------------------------

    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes.xywh.cpu().numpy()

        people_count = len(boxes)

        for box in boxes:

            x, y, w, h = box

            x = int(x)
            y = int(y)

            # ------------------------------------------------
            # ADD HEAT AROUND PERSON
            # ------------------------------------------------

            cv2.circle(
                heatmap,
                (x, y),
                HEAT_RADIUS,
                1.0,
                -1
            )

            # ------------------------------------------------
            # DRAW PERSON BOX
            # ------------------------------------------------

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)

            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    # ========================================================
    # HEATMAP PROCESSING
    # ========================================================

    blurred = cv2.GaussianBlur(
        heatmap,
        (0, 0),
        sigmaX=BLUR_SIGMA,
        sigmaY=BLUR_SIGMA
    )

    max_value = blurred.max()

    if max_value > 0:

        normalized = (
            blurred / max_value * 255
        ).astype(np.uint8)

    else:

        normalized = np.zeros(
            (height, width),
            dtype=np.uint8
        )

    heatmap_color = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET
    )

    # ========================================================
    # COMBINE VIDEO + HEATMAP
    # ========================================================

    output = cv2.addWeighted(
        frame,
        0.65,
        heatmap_color,
        0.35,
        0
    )

    # ========================================================
    # BASIC CROWD DENSITY
    # ========================================================

    if people_count >= 15:

        density = "HIGH"

    elif people_count >= 8:

        density = "MEDIUM"

    else:

        density = "LOW"

    # ========================================================
    # DISPLAY INFORMATION
    # ========================================================

    cv2.putText(
        output,
        f"People: {people_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3
    )

    cv2.putText(
        output,
        f"Density: {density}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Frame: {frame_number}",
        (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Stampede Detection System",
        output
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# CLEANUP
# ============================================================

video.release()
cv2.destroyAllWindows()

print()
print("Detection completed!")