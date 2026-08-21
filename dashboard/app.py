import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Stampede Early Warning System",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Stampede Early Warning System")
st.caption("AI-Based Crowd Monitoring and Early Warning System")

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")


model = load_model()

# ============================================================
# VIDEO DISPLAY
# ============================================================

st.subheader("🎥 Live Crowd Monitoring")

video_display = st.empty()

# ============================================================
# METRICS
# ============================================================

st.divider()

col1, col2, col3, col4 = st.columns(4)

people_display = col1.empty()
density_display = col2.empty()
movement_display = col3.empty()
risk_display = col4.empty()

status_display = st.empty()

# ============================================================
# VIDEO
# ============================================================

video = cv2.VideoCapture("crowd.mp4")

if not video.isOpened():
    st.error("❌ Could not open crowd.mp4")
    st.stop()

# ============================================================
# HEATMAP
# ============================================================

heatmap = None

HEAT_DECAY = 0.98
HEAT_RADIUS = 45

frame_number = 0
start_time = time.time()

# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    success, frame = video.read()

    if not success:
        break

    frame_number += 1

    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    frame = cv2.resize(
        frame,
        (960, 540)
    )

    height, width = frame.shape[:2]

    # Create heatmap
    if heatmap is None:

        heatmap = np.zeros(
            (height, width),
            dtype=np.float32
        )

    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model.predict(
        frame,
        imgsz=640,
        conf=0.20,
        classes=[0],
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------------
    # PEOPLE
    # --------------------------------------------------------

    people_count = 0
    centers = []

    if result.boxes is not None:

        boxes = result.boxes.xyxy.cpu().numpy()

        people_count = len(boxes)

        for box in boxes:

            x1, y1, x2, y2 = box

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            centers.append((cx, cy))

            # Add person's location to heatmap
            cv2.circle(
                heatmap,
                (cx, cy),
                HEAT_RADIUS,
                1.0,
                -1
            )

    # --------------------------------------------------------
    # DECAY OLD HEAT
    # --------------------------------------------------------

    heatmap *= HEAT_DECAY

    # --------------------------------------------------------
    # BLUR
    # --------------------------------------------------------

    smooth_heatmap = cv2.GaussianBlur(
        heatmap,
        (0, 0),
        sigmaX=20,
        sigmaY=20
    )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CREATE HEATMAP COLOR
    # --------------------------------------------------------

    heatmap_color = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET
    )

    # --------------------------------------------------------
    # CREATE YOLO FRAME
    # --------------------------------------------------------

    detection_frame = result.plot()

    # --------------------------------------------------------
    # COMBINE YOLO + HEATMAP
    # --------------------------------------------------------

    combined_frame = cv2.addWeighted(
        detection_frame,
        0.60,
        heatmap_color,
        0.40,
        0
    )

    # --------------------------------------------------------
    # PEOPLE COUNT
    # --------------------------------------------------------

    cv2.putText(
        combined_frame,
        f"People Present: {people_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # FRAME NUMBER
    # --------------------------------------------------------

    cv2.putText(
        combined_frame,
        f"Frame: {frame_number}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # SPATIAL DENSITY
    # --------------------------------------------------------

    grid_rows = 6
    grid_cols = 8

    occupied_cells = set()

    for cx, cy in centers:

        grid_x = min(
            int(cx / width * grid_cols),
            grid_cols - 1
        )

        grid_y = min(
            int(cy / height * grid_rows),
            grid_rows - 1
        )

        occupied_cells.add(
            (grid_x, grid_y)
        )

    total_cells = grid_rows * grid_cols

    density = (
        len(occupied_cells) /
        total_cells
    ) * 100

    # --------------------------------------------------------
    # TEMPORARY RISK SCORE
    # --------------------------------------------------------

    risk_score = min(density, 100)

    if risk_score >= 70:

        status = "🔴 CRITICAL"

    elif risk_score >= 50:

        status = "🟠 WARNING"

    elif risk_score >= 30:

        status = "🟡 CAUTION"

    else:

        status = "🟢 NORMAL"

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    if elapsed > 0:
        fps = frame_number / elapsed
    else:
        fps = 0

    # --------------------------------------------------------
    # DISPLAY ONE VIDEO
    # --------------------------------------------------------

    combined_rgb = cv2.cvtColor(
        combined_frame,
        cv2.COLOR_BGR2RGB
    )

    video_display.image(
        combined_rgb,
        channels="RGB",
        use_container_width=True
    )

    # --------------------------------------------------------
    # DASHBOARD METRICS
    # --------------------------------------------------------

    people_display.metric(
        "👥 People Present",
        people_count
    )

    density_display.metric(
        "🔥 Crowd Density",
        f"{density:.1f}%"
    )

    movement_display.metric(
        "🏃 Movement",
        "Pending"
    )

    risk_display.metric(
        "⚠️ Risk Score",
        f"{risk_score:.1f}/100"
    )

    status_display.markdown(
        f"## {status}"
    )

# ============================================================
# CLEANUP
# ============================================================

video.release()

st.success("✅ Video processing completed.")