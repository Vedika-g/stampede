import sys
import os
import time
import requests
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, ROOT_DIR)

# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import torch

from huggingface_hub import hf_hub_download

from csrnet.csrnet_model import CSRNet
from risk_engine import StampedeRiskEngine

from dashboard_ui import (
    apply_dashboard_style,
    show_header,
    show_sidebar,
    show_section_title,
    create_video_placeholder,
    show_risk_status,
    show_risk_progress,
    create_history_container,
    update_history_chart,
    show_alert_panel,
    show_system_info,
    show_footer,
    show_monitoring_status
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Stampede Early Warning System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DASHBOARD STARTUP
# ============================================================

apply_dashboard_style()

show_header()

input_mode, phone_url = show_sidebar()

# ============================================================
# PERFORMANCE CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# YOLOv8m
# ------------------------------------------------------------
# Increased image size and lower confidence improve detection
# of smaller / partially occluded people.
# ------------------------------------------------------------

YOLO_INTERVAL = 2

YOLO_IMGSZ = 768

YOLO_CONF = 0.20

# ------------------------------------------------------------
# CSRNet
# ------------------------------------------------------------

CSRNET_INTERVAL = 20

# ------------------------------------------------------------
# Optical Flow
# ------------------------------------------------------------

FLOW_INTERVAL = 6

# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

DISPLAY_WIDTH = 576

DISPLAY_HEIGHT = 324

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------

UI_UPDATE_INTERVAL = 8

# ------------------------------------------------------------
# Backend
# ------------------------------------------------------------

API_URL = "http://127.0.0.1:8000/analysis"

API_SEND_INTERVAL = 5

backend_executor = ThreadPoolExecutor(
    max_workers=1
)

# ============================================================
# CUDA CONFIGURATION
# ============================================================

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:

    torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.backends.cudnn.allow_tf32 = True

    torch.set_float32_matmul_precision("high")

DEVICE = 0 if USE_CUDA else "cpu"

HALF = True if USE_CUDA else False

# ============================================================
# MODEL LOADERS
# ============================================================

@st.cache_resource
def load_yolo():

    model_path = os.path.join(
        ROOT_DIR,
        "yolov8m.pt"
    )

    model = YOLO(model_path)

    return model


@st.cache_resource
def load_csrnet():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    weights_path = hf_hub_download(
        repo_id="AbdurRahman011/csrnet-indian-metro-crowd-density",
        filename="csrnet_v3_best.pth"
    )

    model = CSRNet()

    checkpoint = torch.load(
        weights_path,
        map_location=device
    )

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            checkpoint = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:

            checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(
        checkpoint,
        strict=False
    )

    model.to(device)

    model.eval()

    return model, device


# ============================================================
# LOAD MODELS
# ============================================================

with st.spinner("Loading AI models..."):

    yolo = load_yolo()

    csrnet, csrnet_device = load_csrnet()

# ============================================================
# CSRNET NORMALIZATION
# ============================================================

CSR_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32
).view(3, 1, 1)

CSR_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32
).view(3, 1, 1)

# ============================================================
# BACKEND FUNCTION
# ============================================================

def save_analysis_to_backend(
    camera_id,
    people_count,
    density,
    movement,
    risk_score,
    risk_level
):

    data = {

        "camera_id": str(
            camera_id
        ),

        "people_count": int(
            people_count
        ),

        "density": float(
            density
        ),

        "movement": float(
            movement
        ),

        "risk_score": float(
            risk_score
        ),

        "risk_level": str(
            risk_level
        )
    }

    try:

        requests.post(
            API_URL,
            json=data,
            timeout=2
        )

    except Exception:

        pass


# ============================================================
# UI LAYOUT
# ============================================================

show_section_title(
    "🎥 Live Crowd Monitoring",
    "Real-time AI-powered crowd detection and stampede risk analysis"
)

video_display = create_video_placeholder()

st.divider()

# ============================================================
# RISK STATUS
# ============================================================

risk_status_display = st.empty()

# ============================================================
# ANALYTICS
# ============================================================

st.divider()

show_section_title(
    "📈 Crowd Analytics",
    "Continuous monitoring of crowd size, density, movement and risk"
)

history_placeholder = create_history_container()

# ============================================================
# ALERT CENTER
# ============================================================

st.divider()

show_section_title(
    "🚨 Alert Center",
    "Current crowd safety status"
)

alert_placeholder = st.empty()

# ============================================================
# RISK PROGRESS
# ============================================================

st.divider()

risk_progress_placeholder = st.empty()

# ============================================================
# SYSTEM STATUS
# ============================================================

st.divider()

status_display = st.empty()

# ============================================================
# START BUTTON
# ============================================================

st.divider()

start_monitoring = st.button(
    "▶️ START MONITORING",
    use_container_width=True
)

if not start_monitoring:

    st.info(
        "Click **START MONITORING** to begin AI crowd analysis."
    )

    show_footer()

    st.stop()

# ============================================================
# VIDEO SOURCE
# ============================================================

if input_mode == "Recorded Video":

    video_source = os.path.join(
        ROOT_DIR,
        "people_walking.mp4"
    )

    camera_id = "RECORDED_VIDEO"

else:

    if not phone_url.strip():

        st.error(
            "❌ Enter your phone camera URL."
        )

        st.stop()

    video_source = phone_url.strip()

    camera_id = "PHONE_CAMERA"

# ============================================================
# OPEN VIDEO
# ============================================================

video = cv2.VideoCapture(
    video_source
)

video.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

if not video.isOpened():

    st.error(
        "❌ Could not open video source."
    )

    st.stop()

# ============================================================
# STATE VARIABLES
# ============================================================

frame_number = 0

processed_yolo_frames = 0

processed_csrnet_frames = 0

processed_flow_frames = 0

peak_people = 0

last_people_count = 0

last_boxes = []

last_density = 0.0

last_heatmap = None

previous_gray = None

previous_movement = 0.0

last_movement = 0.0

risk_engine = StampedeRiskEngine()

start_time = time.time()

last_backend_send = 0

backend_task = None

# ------------------------------------------------------------
# History buffer
# ------------------------------------------------------------

history = deque(
    maxlen=45
)

# ------------------------------------------------------------
# Cached display
# ------------------------------------------------------------

last_display = None

# ============================================================
# MONITORING STATUS
# ============================================================

show_monitoring_status()

# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

try:

    while True:

        # ====================================================
        # READ FRAME
        # ====================================================

        success, original_frame = video.read()

        if not success:

            break

        frame_number += 1

        # ====================================================
        # YOLO PERSON DETECTION
        # Every 2 frames
        # ====================================================

        if frame_number % YOLO_INTERVAL == 0:

            results = yolo.predict(

                original_frame,

                classes=[0],

                conf=YOLO_CONF,

                imgsz=YOLO_IMGSZ,

                device=DEVICE,

                half=HALF,

                verbose=False
            )

            result = results[0]

            current_boxes = []

            if (
                result.boxes is not None
                and len(result.boxes) > 0
            ):

                coordinates = (
                    result
                    .boxes
                    .xyxy
                    .detach()
                    .cpu()
                    .numpy()
                )

                for box in coordinates:

                    x1, y1, x2, y2 = (
                        box.astype(int)
                    )

                    box_width = x2 - x1

                    box_height = y2 - y1

                    # ------------------------------------------------
                    # Smaller boxes are now accepted.
                    # This helps detect people farther away.
                    # ------------------------------------------------

                    if box_width < 6:

                        continue

                    if box_height < 12:

                        continue

                    current_boxes.append(
                        (
                            x1,
                            y1,
                            x2,
                            y2
                        )
                    )

            # ------------------------------------------------
            # Update latest YOLO results
            # ------------------------------------------------

            last_boxes = current_boxes

            last_people_count = len(
                current_boxes
            )

            processed_yolo_frames += 1

        # ====================================================
        # CURRENT PEOPLE COUNT
        # ====================================================

        people_count = last_people_count

        if people_count > peak_people:

            peak_people = people_count

        # ====================================================
        # CSRNET DENSITY
        # Every 20 frames
        # ====================================================

        if frame_number % CSRNET_INTERVAL == 0:

            csr_frame = cv2.resize(

                original_frame,

                (384, 216),

                interpolation=cv2.INTER_AREA
            )

            rgb = cv2.cvtColor(

                csr_frame,

                cv2.COLOR_BGR2RGB
            )

            # ------------------------------------------------
            # NumPy → Torch
            # ------------------------------------------------

            tensor = torch.from_numpy(
                rgb
            )

            tensor = (
                tensor
                .permute(
                    2,
                    0,
                    1
                )
                .contiguous()
                .float()
            )

            tensor.div_(255.0)

            tensor.sub_(
                CSR_MEAN
            ).div_(
                CSR_STD
            )

            tensor = tensor.unsqueeze(0)

            tensor = tensor.to(
                csrnet_device,
                non_blocking=True
            )

            with torch.inference_mode():

                density_output = csrnet(
                    tensor
                )

            density_map = (
                density_output
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Remove negative values
            # ------------------------------------------------

            np.maximum(
                density_map,
                0,
                out=density_map
            )

            last_density = float(
                density_map.sum()
            )

            processed_csrnet_frames += 1

            # =================================================
            # CSRNET HEATMAP
            # =================================================

            density_min = float(
                density_map.min()
            )

            density_max = float(
                density_map.max()
            )

            if density_max > density_min:

                normalized = cv2.normalize(

                    density_map,

                    None,

                    0,

                    255,

                    cv2.NORM_MINMAX
                ).astype(
                    np.uint8
                )

            else:

                normalized = np.zeros(
                    density_map.shape,
                    dtype=np.uint8
                )

            heatmap = cv2.applyColorMap(

                normalized,

                cv2.COLORMAP_JET
            )

            last_heatmap = cv2.resize(

                heatmap,

                (
                    DISPLAY_WIDTH,
                    DISPLAY_HEIGHT
                ),

                interpolation=cv2.INTER_LINEAR
            )

        # ====================================================
        # OPTICAL FLOW
        # Every 6 frames
        # ====================================================

        sudden_movement = False

        if (
            previous_gray is None
            or frame_number % FLOW_INTERVAL == 0
        ):

            flow_frame = cv2.resize(

                original_frame,

                (320, 180),

                interpolation=cv2.INTER_AREA
            )

            gray = cv2.cvtColor(

                flow_frame,

                cv2.COLOR_BGR2GRAY
            )

            if previous_gray is None:

                previous_gray = gray

            else:

                flow = cv2.calcOpticalFlowFarneback(

                    previous_gray,

                    gray,

                    None,

                    0.5,

                    2,

                    10,

                    2,

                    3,

                    1.1,

                    0
                )

                # ------------------------------------------------
                # Magnitude
                # ------------------------------------------------

                magnitude = cv2.magnitude(

                    flow[..., 0],

                    flow[..., 1]
                )

                current_movement = float(
                    np.mean(magnitude)
                )

                # ------------------------------------------------
                # Sudden movement detection
                # ------------------------------------------------

                if (
                    previous_movement > 0
                    and
                    current_movement
                    >
                    previous_movement * 1.5
                ):

                    sudden_movement = True

                last_movement = (
                    current_movement
                )

                previous_movement = (
                    current_movement
                )

                previous_gray = gray

                processed_flow_frames += 1

        # ====================================================
        # RISK ENGINE
        # ====================================================

        risk = risk_engine.calculate_risk(

            people_count,

            last_density,

            last_movement,

            sudden_movement
        )

        risk_score = risk[
            "risk_score"
        ]

        risk_level = risk[
            "risk_level"
        ]

        warning = risk[
            "warning"
        ]

        # ====================================================
        # HISTORY
        # ====================================================

        history.append({

            "people": people_count,

            "density": last_density,

            "movement": last_movement,

            "risk": risk_score

        })

        # ====================================================
        # BACKEND UPDATE
        # Non-blocking
        # ====================================================

        current_time = time.time()

        if (
            current_time
            - last_backend_send
            >= API_SEND_INTERVAL
        ):

            if (
                backend_task is None
                or backend_task.done()
            ):

                backend_task = (
                    backend_executor.submit(

                        save_analysis_to_backend,

                        camera_id,

                        people_count,

                        last_density,

                        last_movement,

                        risk_score,

                        risk_level
                    )
                )

                last_backend_send = (
                    current_time
                )

        # ====================================================
        # STREAMLIT UI UPDATE
        # ====================================================

        if (
            frame_number % UI_UPDATE_INTERVAL == 0
            or frame_number == 1
        ):

            # =================================================
            # CREATE DISPLAY FRAME
            # =================================================

            frame = cv2.resize(

                original_frame,

                (
                    DISPLAY_WIDTH,
                    DISPLAY_HEIGHT
                ),

                interpolation=cv2.INTER_AREA
            )

            display = frame.copy()

            # =================================================
            # HEATMAP OVERLAY
            # =================================================

            if last_heatmap is not None:

                display = cv2.addWeighted(

                    display,

                    0.70,

                    last_heatmap,

                    0.30,

                    0
                )

            # =================================================
            # DRAW YOLO BOXES
            # =================================================

            original_height, original_width = (
                original_frame.shape[:2]
            )

            scale_x = (
                DISPLAY_WIDTH
                / original_width
            )

            scale_y = (
                DISPLAY_HEIGHT
                / original_height
            )

            for box in last_boxes:

                x1, y1, x2, y2 = box

                x1 = int(
                    x1 * scale_x
                )

                y1 = int(
                    y1 * scale_y
                )

                x2 = int(
                    x2 * scale_x
                )

                y2 = int(
                    y2 * scale_y
                )

                cv2.rectangle(

                    display,

                    (x1, y1),

                    (x2, y2),

                    (0, 255, 0),

                    2
                )

            # =================================================
            # TEXT OVERLAY
            # =================================================

            cv2.putText(

                display,

                f"People: {people_count}",

                (15, 30),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 255),

                2
            )

            cv2.putText(

                display,

                f"Density: {last_density:.1f}",

                (15, 60),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 255),

                2
            )

            cv2.putText(

                display,

                f"Movement: {last_movement:.2f}",

                (15, 90),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 255),

                2
            )

            cv2.putText(

                display,

                f"Risk: {risk_score:.1f}",

                (15, 120),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 255),

                2
            )

            # =================================================
            # STAMPEDE WARNING OVERLAY
            # =================================================

            if warning:

                cv2.putText(

                    display,

                    "!!! STAMPEDE WARNING !!!",

                    (15, 155),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    (0, 0, 255),

                    3
                )

            # =================================================
            # CACHE DISPLAY
            # =================================================

            last_display = cv2.cvtColor(

                display,

                cv2.COLOR_BGR2RGB
            )

            # =================================================
            # RISK STATUS
            # =================================================

            show_risk_status(

                risk_status_display,

                risk_level,

                risk_score,

                warning
            )

            # =================================================
            # ALERT CENTER
            # =================================================

            show_alert_panel(

                alert_placeholder,

                risk_level,

                risk_score,

                warning
            )

            # =================================================
            # RISK PROGRESS
            # =================================================

            with risk_progress_placeholder.container():

                show_risk_progress(
                    risk_score
                )

            # =================================================
            # HISTORY
            # =================================================

            update_history_chart(

                history_placeholder,

                list(history)
            )

            # =================================================
            # MONITORING STATUS
            # =================================================

            if warning:

                status_display.error(

                    "🚨 STAMPEDE WARNING — "
                    "Potential stampede conditions detected."
                )

            elif risk_level == "HIGH":

                status_display.error(

                    "🔴 HIGH RISK — "
                    "Immediate attention recommended."
                )

            elif risk_level == "MEDIUM":

                status_display.warning(

                    "🟠 MEDIUM RISK — "
                    "Continue monitoring crowd conditions."
                )

            else:

                status_display.success(

                    "🟢 MONITORING ACTIVE — "
                    "Crowd conditions currently stable."
                )

            # =================================================
            # VIDEO DISPLAY
            # =================================================

            if last_display is not None:

                video_display.image(

                    last_display,

                    channels="RGB",

                    use_container_width=True
                )

# ============================================================
# CLEANUP
# ============================================================

finally:

    video.release()

    backend_executor.shutdown(
        wait=False
    )

# ============================================================
# FINAL STATISTICS
# ============================================================

elapsed = (
    time.time()
    - start_time
)

final_fps = (

    frame_number / elapsed

    if elapsed > 0

    else 0
)

# ============================================================
# SYSTEM INFORMATION
# ============================================================

show_system_info(

    camera_id,

    final_fps,

    processed_yolo_frames,

    processed_csrnet_frames,

    backend_status=True
)

st.success(
    "✅ Processing completed successfully."
)

st.info(

    f"Frames processed: {frame_number} | "
    f"YOLO: {processed_yolo_frames} | "
    f"CSRNet: {processed_csrnet_frames} | "
    f"Optical Flow: {processed_flow_frames} | "
    f"Peak crowd: {peak_people} people"
)

show_footer()