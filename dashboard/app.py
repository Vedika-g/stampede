import sys
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    ROOT_DIR
)


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st

from ultralytics import YOLO

import cv2
import numpy as np
import torch

from PIL import Image

from torchvision import transforms

from huggingface_hub import hf_hub_download

from csrnet.csrnet_model import CSRNet

from risk_engine import StampedeRiskEngine


# ============================================================
# DASHBOARD UI
# ============================================================

from dashboard_ui import (
    apply_dashboard_style,
    show_header,
    show_sidebar,
    show_section_title,
    create_video_placeholder,
    create_metric_cards,
    update_metrics,
    show_risk_status,
    update_score_cards,
    show_risk_progress,
    create_history_container,
    update_history_chart,
    show_alert_panel,
    show_system_info,
    show_footer
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Stampede Early Warning System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# APPLY UI
# ============================================================

apply_dashboard_style()

show_header()

input_mode, phone_url = show_sidebar()


# ============================================================
# BACKEND CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/analysis"

API_SEND_INTERVAL = 5

backend_executor = ThreadPoolExecutor(
    max_workers=1
)

backend_task = None


def save_analysis_to_backend(
    camera_id,
    people_count,
    density,
    movement,
    risk_score,
    risk_level
):

    data = {

        "camera_id": str(camera_id),

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

        # Backend failure must never
        # stop AI processing.

        pass


# ============================================================
# PERFORMANCE CONFIGURATION
# ============================================================

# DO NOT CHANGE

YOLO_INTERVAL = 3

YOLO_IMGSZ = 960

YOLO_CONF = 0.30

CSRNET_INTERVAL = 10

FLOW_INTERVAL = 3

DISPLAY_WIDTH = 640

DISPLAY_HEIGHT = 360


# ============================================================
# CUDA
# ============================================================

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:

    torch.backends.cudnn.benchmark = True


DEVICE = (
    0
    if USE_CUDA
    else "cpu"
)

HALF = (
    True
    if USE_CUDA
    else False
)


# ============================================================
# LOAD YOLO
# ============================================================

@st.cache_resource
def load_yolo():

    model_path = os.path.join(
        ROOT_DIR,
        "yolov8m.pt"
    )

    model = YOLO(
        model_path
    )

    return model


# ============================================================
# LOAD CSRNET
# ============================================================

@st.cache_resource
def load_csrnet():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    weights_path = hf_hub_download(
        repo_id=(
            "AbdurRahman011/"
            "csrnet-indian-metro-crowd-density"
        ),
        filename="csrnet_v3_best.pth"
    )

    model = CSRNet()

    checkpoint = torch.load(
        weights_path,
        map_location=device
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "state_dict" in checkpoint:

            checkpoint = (
                checkpoint["state_dict"]
            )

        elif "model_state_dict" in checkpoint:

            checkpoint = (
                checkpoint["model_state_dict"]
            )

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

with st.spinner(
    "Loading AI models..."
):

    yolo = load_yolo()

    csrnet, csrnet_device = load_csrnet()


# ============================================================
# CSRNET TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# MAIN MONITORING AREA
# ============================================================

show_section_title(
    "🎥 Live Crowd Monitoring",
    "AI-annotated video feed with person detection, "
    "density heatmap and movement analysis"
)


video_display = create_video_placeholder()


# ============================================================
# MAIN METRICS
# ============================================================

st.markdown(
    "### 📊 Real-Time Intelligence"
)

metric_columns = create_metric_cards()


# ============================================================
# RISK STATUS
# ============================================================

st.markdown(
    "### ⚠️ Crowd Safety Status"
)

risk_status_display = st.empty()


# ============================================================
# ANALYTICS AREA
# ============================================================

left_column, right_column = st.columns(
    [2.2, 1],
    gap="large"
)


with left_column:

    show_section_title(
        "📈 Live Risk Analytics",
        "Recent crowd behaviour and AI risk trends"
    )

    history_placeholder = (
        create_history_container()
    )


with right_column:

    show_section_title(
        "🚨 Alert Center",
        "Current crowd safety condition"
    )

    alert_placeholder = st.empty()


# ============================================================
# RISK COMPONENTS
# ============================================================

show_section_title(
    "🧠 AI Risk Components",
    "Individual factors contributing to the overall risk score"
)

score_columns = st.columns(3)


# ============================================================
# RISK PROGRESS
# ============================================================

risk_progress_placeholder = st.empty()


# ============================================================
# STATUS
# ============================================================

status_display = st.empty()


# ============================================================
# START MONITORING
# ============================================================

start_monitoring = st.sidebar.button(
    "▶ Start Monitoring",
    use_container_width=True
)


if start_monitoring:

    # ========================================================
    # VIDEO SOURCE
    # ========================================================

    if input_mode == "Recorded Video":

        video_source = os.path.join(
            ROOT_DIR,
            "crowd.mp4"
        )

        camera_id = "RECORDED_VIDEO"

    else:

        if not phone_url.strip():

            st.error(
                "❌ Enter your phone camera URL."
            )

            st.stop()

        video_source = (
            phone_url.strip()
        )

        camera_id = "PHONE_CAMERA"


    # ========================================================
    # OPEN VIDEO
    # ========================================================

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


    status_display.success(
        "✅ Video source connected. "
        "AI monitoring started."
    )


    # ========================================================
    # VARIABLES
    # ========================================================

    frame_number = 0

    processed_yolo_frames = 0

    processed_csrnet_frames = 0

    peak_people = 0

    last_people_count = 0

    last_boxes = []

    last_density = 0.0

    last_heatmap = None

    previous_gray = None

    previous_movement = 0.0

    last_movement = 0.0

    risk_engine = (
        StampedeRiskEngine()
    )

    start_time = time.time()

    last_backend_send = 0

    backend_task = None

    history = []


    # ========================================================
    # PROCESSING LOOP
    # ========================================================

    while True:

        success, original_frame = (
            video.read()
        )

        if not success:

            break


        frame_number += 1


        # ====================================================
        # YOLOv8m PERSON DETECTION
        # ====================================================

        if (
            frame_number
            % YOLO_INTERVAL
            == 0
        ):

            results = yolo.track(

                original_frame,

                classes=[0],

                conf=YOLO_CONF,

                imgsz=YOLO_IMGSZ,

                device=DEVICE,

                half=HALF,

                persist=True,

                tracker="bytetrack.yaml",

                verbose=False
            )


            result = results[0]

            current_boxes = []


            if (

                result.boxes is not None

                and

                len(result.boxes) > 0

            ):

                boxes = result.boxes

                coordinates = (

                    boxes.xyxy
                    .detach()
                    .cpu()
                    .numpy()
                )


                # =========================================
                # FILTER SMALL DETECTIONS
                # =========================================

                for box in coordinates:

                    x1, y1, x2, y2 = (
                        box.astype(int)
                    )

                    box_width = (
                        x2 - x1
                    )

                    box_height = (
                        y2 - y1
                    )


                    if box_width < 10:

                        continue


                    if box_height < 20:

                        continue


                    current_boxes.append(
                        box
                    )


            last_boxes = (
                current_boxes
            )

            last_people_count = (
                len(current_boxes)
            )

            processed_yolo_frames += 1


        # ====================================================
        # PEOPLE COUNT
        # ====================================================

        people_count = (
            last_people_count
        )

        peak_people = max(
            peak_people,
            people_count
        )


        # ====================================================
        # DISPLAY FRAME
        # ====================================================

        frame = cv2.resize(

            original_frame,

            (
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT
            ),

            interpolation=cv2.INTER_AREA
        )


        height, width = (
            frame.shape[:2]
        )


        # ====================================================
        # CSRNET
        # ====================================================

        if (
            frame_number
            % CSRNET_INTERVAL
            == 0
        ):

            csr_frame = cv2.resize(

                original_frame,

                (
                    512,
                    288
                ),

                interpolation=cv2.INTER_AREA
            )


            rgb = cv2.cvtColor(

                csr_frame,

                cv2.COLOR_BGR2RGB
            )


            image = Image.fromarray(
                rgb
            )


            tensor = transform(
                image
            ).unsqueeze(0)


            tensor = tensor.to(

                csrnet_device,

                non_blocking=True
            )


            with torch.inference_mode():

                density_output = (
                    csrnet(tensor)
                )


            density_map = (

                density_output
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )


            density_map = np.maximum(
                density_map,
                0
            )


            last_density = float(
                density_map.sum()
            )


            processed_csrnet_frames += 1


            # =============================================
            # HEATMAP
            # =============================================

            min_value = (
                density_map.min()
            )

            max_value = (
                density_map.max()
            )


            if max_value > min_value:

                normalized = cv2.normalize(

                    density_map,

                    None,

                    0,

                    255,

                    cv2.NORM_MINMAX
                )

            else:

                normalized = (
                    np.zeros_like(
                        density_map,
                        dtype=np.uint8
                    )
                )


            normalized = (
                normalized.astype(
                    np.uint8
                )
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
        # ====================================================

        gray = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2GRAY
        )


        sudden_movement = False


        if previous_gray is None:

            previous_gray = (
                gray.copy()
            )


        elif (
            frame_number
            % FLOW_INTERVAL
            == 0
        ):

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


            magnitude, angle = (
                cv2.cartToPolar(

                    flow[..., 0],

                    flow[..., 1]
                )
            )


            current_movement = float(
                np.mean(magnitude)
            )


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

            previous_gray = (
                gray.copy()
            )


        # ====================================================
        # RISK ENGINE
        # ====================================================

        risk = (
            risk_engine.calculate_risk(

                people_count,

                last_density,

                last_movement,

                sudden_movement
            )
        )


        risk_score = (
            risk["risk_score"]
        )

        risk_level = (
            risk["risk_level"]
        )

        warning = (
            risk["warning"]
        )


        # ====================================================
        # SAVE HISTORY
        # ====================================================

        history.append({

            "people": people_count,

            "density": last_density,

            "movement": last_movement,

            "risk": risk_score
        })


        if len(history) > 60:

            history.pop(0)


        # ====================================================
        # BACKEND
        # ====================================================

        current_time = time.time()


        if (

            current_time
            - last_backend_send
            >= API_SEND_INTERVAL

        ):

            if (

                backend_task is None

                or

                backend_task.done()

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
        # HEATMAP OVERLAY
        # ====================================================

        if last_heatmap is not None:

            display = cv2.addWeighted(

                frame,

                0.70,

                last_heatmap,

                0.30,

                0
            )

        else:

            display = frame.copy()


        # ====================================================
        # YOLO BOXES
        # ====================================================

        original_height, original_width = (
            original_frame.shape[:2]
        )


        scale_x = (
            width /
            original_width
        )

        scale_y = (
            height /
            original_height
        )


        for box in last_boxes:

            x1, y1, x2, y2 = (
                box.astype(int)
            )


            x1 = int(
                x1 * scale_x
            )

            x2 = int(
                x2 * scale_x
            )

            y1 = int(
                y1 * scale_y
            )

            y2 = int(
                y2 * scale_y
            )


            cv2.rectangle(

                display,

                (x1, y1),

                (x2, y2),

                (255, 255, 255),

                2
            )


        # ====================================================
        # VIDEO OVERLAY
        # ====================================================

        cv2.putText(

            display,

            f"People: {people_count}",

            (15, 30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (0, 255, 0),

            2
        )


        cv2.putText(

            display,

            f"Density: {last_density:.1f}",

            (15, 60),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (0, 255, 255),

            2
        )


        cv2.putText(

            display,

            f"Movement: {last_movement:.2f}",

            (15, 90),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            2
        )


        cv2.putText(

            display,

            f"Risk: {risk_score:.1f}/100",

            (15, 120),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (0, 165, 255),

            2
        )


        if warning:

            cv2.putText(

                display,

                "!!! STAMPEDE WARNING !!!",

                (
                    max(
                        width - 350,
                        200
                    ),
                    35
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (0, 0, 255),

                2
            )


        # ====================================================
        # UPDATE MAIN METRICS
        # ====================================================

        update_metrics(

            metric_columns,

            people_count,

            peak_people,

            last_density,

            last_movement,

            risk_score
        )


        # ====================================================
        # UPDATE RISK STATUS
        # ====================================================

        show_risk_status(

            risk_status_display,

            risk_level,

            risk_score,

            warning
        )


        # ====================================================
        # UPDATE ALERT CENTER
        # ====================================================

        show_alert_panel(

            alert_placeholder,

            risk_level,

            risk_score,

            warning
        )


        # ====================================================
        # UPDATE RISK COMPONENTS
        # ====================================================

        update_score_cards(

            score_columns,

            risk
        )


        # ====================================================
        # RISK PROGRESS
        # ====================================================

        with risk_progress_placeholder.container():

            show_risk_progress(
                risk_score
            )


        # ====================================================
        # LIVE CHART
        # ====================================================

        update_history_chart(

            history_placeholder,

            history
        )


        # ====================================================
        # STATUS
        # ====================================================

        if warning:

            status_display.error(

                "🚨 STAMPEDE WARNING — "
                "Sustained high-risk conditions detected!"
            )

        elif risk_level == "HIGH":

            status_display.warning(

                "🔴 HIGH RISK — "
                "Abnormal crowd conditions detected."
            )

        elif risk_level == "MEDIUM":

            status_display.warning(

                "🟠 MEDIUM RISK — "
                "Increased crowd activity detected."
            )

        else:

            status_display.success(

                "🟢 LOW RISK — "
                "Crowd conditions are currently stable."
            )


        # ====================================================
        # FPS
        # ====================================================

        elapsed = (
            time.time()
            - start_time
        )


        fps = (

            frame_number / elapsed

            if elapsed > 0

            else 0
        )


        # ====================================================
        # SIDEBAR FPS
        # ====================================================

        fps_placeholder = (
            st.sidebar.empty()
        )

        fps_placeholder.metric(
            "Processing FPS",
            f"{fps:.1f}"
        )


        # ====================================================
        # DISPLAY VIDEO
        # ====================================================

        rgb_display = cv2.cvtColor(

            display,

            cv2.COLOR_BGR2RGB
        )


        video_display.image(

            rgb_display,

            channels="RGB",

            use_container_width=True
        )


    # ========================================================
    # CLEANUP
    # ========================================================

    video.release()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )


    final_fps = (

        frame_number / elapsed

        if elapsed > 0

        else 0
    )


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

        f"Frames processed: {frame_number}  |  "
        f"YOLO: {processed_yolo_frames}  |  "
        f"CSRNet: {processed_csrnet_frames}  |  "
        f"Peak crowd: {peak_people} people"
    )


# ============================================================
# FOOTER
# ============================================================

show_footer()