from ultralytics import YOLO
import cv2
import numpy as np

# ============================================================
# LOAD YOLOv8m
# ============================================================

model = YOLO("yolov8m.pt")

# ============================================================
# OPEN VIDEO
# ============================================================

video = cv2.VideoCapture("crowd.mp4")

if not video.isOpened():
    print("ERROR: Could not open crowd.mp4")
    exit()

# ============================================================
# HEATMAP SETTINGS
# ============================================================

heatmap = None

HEAT_RADIUS = 45

# How quickly old heat fades
HEAT_DECAY = 0.985

# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    success, frame = video.read()

    if not success:
        break

    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model.predict(
        frame,
        conf=0.20,
        classes=[0],
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------------
    # CREATE HEATMAP
    # --------------------------------------------------------

    height, width = frame.shape[:2]

    if heatmap is None:

        heatmap = np.zeros(
            (height, width),
            dtype=np.float32
        )

    # --------------------------------------------------------
    # COUNT PEOPLE
    # --------------------------------------------------------

    if result.boxes is not None:

        boxes = result.boxes.xyxy.cpu().numpy()

        people_count = len(boxes)

        for box in boxes:

            x1, y1, x2, y2 = box

            # -----------------------------------------------
            # PERSON CENTER
            # -----------------------------------------------

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )

            # -----------------------------------------------
            # ADD PERSON TO HEATMAP
            # -----------------------------------------------

            cv2.circle(
                heatmap,
                (center_x, center_y),
                HEAT_RADIUS,
                1.0,
                -1
            )

    else:

        people_count = 0

    # ========================================================
    # DECAY PREVIOUS HEAT
    # ========================================================

    heatmap *= HEAT_DECAY

    # ========================================================
    # SMOOTH HEATMAP
    # ========================================================

    smooth_heatmap = cv2.GaussianBlur(
        heatmap,
        (0, 0),
        sigmaX=25,
        sigmaY=25
    )

    # ========================================================
    # NORMALIZE
    # ========================================================

    if smooth_heatmap.max() > 0:

        normalized = cv2.normalize(
            smooth_heatmap,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

    else:

        normalized = np.zeros(
            (height, width),
            dtype=np.uint8
        )

    normalized = normalized.astype(np.uint8)

    # ========================================================
    # APPLY HEATMAP COLORS
    # ========================================================

    heatmap_color = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET
    )

    # ========================================================
    # YOLO BOUNDING BOXES
    # ========================================================

    detection_frame = result.plot()

    # ========================================================
    # COMBINE VIDEO + HEATMAP
    # ========================================================

    combined = cv2.addWeighted(
        detection_frame,
        0.60,
        heatmap_color,
        0.40,
        0
    )

    # ========================================================
    # DISPLAY PEOPLE COUNT
    # ========================================================

    cv2.rectangle(
        combined,
        (10, 10),
        (400, 85),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        combined,
        f"PEOPLE: {people_count}",
        (25, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 255, 0),
        3
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Stampede Detection - Continuous Heatmap",
        combined
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

video.release()
cv2.destroyAllWindows()

print("Heatmap processing completed.")