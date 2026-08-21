from ultralytics import YOLO
import cv2
import numpy as np

# ============================================================
# 1. LOAD YOLO MODEL
# ============================================================

model = YOLO("yolov8n.pt")

# ============================================================
# 2. OPEN LIVE CAMERA
# ============================================================

# 0 = default laptop/webcam camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()

# ============================================================
# 3. GET CAMERA DIMENSIONS
# ============================================================

width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera resolution: {width} x {height}")

# ============================================================
# 4. CREATE HEATMAP
# ============================================================

heatmap = np.zeros(
    (height, width),
    dtype=np.float32
)

# ============================================================
# 5. SETTINGS
# ============================================================

DECAY = 0.92
HEAT_RADIUS = 50
BLUR_SIGMA = 25

# ============================================================
# 6. LIVE PROCESSING
# ============================================================

frame_number = 0

while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
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

            # Add heat around person
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

    # --------------------------------------------------------
    # BLUR HEATMAP
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        heatmap,
        (0, 0),
        sigmaX=BLUR_SIGMA,
        sigmaY=BLUR_SIGMA
    )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # APPLY HEATMAP
    # --------------------------------------------------------

    heatmap_color = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET
    )

    # --------------------------------------------------------
    # BLEND CAMERA + HEATMAP
    # --------------------------------------------------------

    output = cv2.addWeighted(
        frame,
        0.65,
        heatmap_color,
        0.35,
        0
    )

    # ========================================================
    # STAMPEDE RISK INDICATOR
    # ========================================================

    # Basic placeholder threshold.
    # We will replace this with a real risk model later.

    if people_count >= 15:

        risk = "HIGH RISK"

        risk_color = (0, 0, 255)

    elif people_count >= 8:

        risk = "MEDIUM RISK"

        risk_color = (0, 165, 255)

    else:

        risk = "NORMAL"

        risk_color = (0, 255, 0)

    # --------------------------------------------------------
    # DISPLAY PEOPLE COUNT
    # --------------------------------------------------------

    cv2.putText(
        output,
        f"People: {people_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        3
    )

    # --------------------------------------------------------
    # DISPLAY RISK
    # --------------------------------------------------------

    cv2.putText(
        output,
        f"Status: {risk}",
        (30, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        risk_color,
        3
    )

    # --------------------------------------------------------
    # DISPLAY FRAME
    # --------------------------------------------------------

    cv2.putText(
        output,
        f"Frame: {frame_number}",
        (30, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # SHOW LIVE VIDEO
    # --------------------------------------------------------

    cv2.imshow(
        "Stampede Early Warning System",
        output
    )

    # --------------------------------------------------------
    # PRESS Q TO EXIT
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# 7. RELEASE
# ============================================================

camera.release()

cv2.destroyAllWindows()

print("Live detection stopped.")