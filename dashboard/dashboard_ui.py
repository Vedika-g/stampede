import streamlit as st


# ============================================================
# DASHBOARD STYLE
# ============================================================

def apply_dashboard_style():

    st.markdown(

        """
        <style>

        /* Main application */

        .stApp {
            background-color: #080b10;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }


        /* Sidebar */

        section[data-testid="stSidebar"] {

            background-color: #0c1016;

            border-right: 1px solid #252c36;
        }


        /* Headings */

        h1 {

            font-size: 38px !important;

            font-weight: 800 !important;

            letter-spacing: -1px;
        }

        h2 {

            font-size: 27px !important;

            font-weight: 750 !important;
        }

        h3 {

            font-size: 20px !important;

            font-weight: 700 !important;
        }


        /* Metric cards */

        div[data-testid="stMetric"] {

            background-color: #11161e;

            border: 1px solid #29313c;

            border-radius: 14px;

            padding: 18px 20px;

            min-height: 125px;
        }


        div[data-testid="stMetricLabel"] {

            color: #8d99a8;

            font-size: 12px;

            font-weight: 700;
        }


        div[data-testid="stMetricValue"] {

            color: #f5f7fa;

            font-size: 29px;

            font-weight: 800;
        }


        /* Buttons */

        .stButton > button {

            width: 100%;

            min-height: 44px;

            border-radius: 10px;

            border: 1px solid #303946;

            background-color: #151b23;

            color: #f5f7fa;

            font-weight: 700;
        }


        .stButton > button:hover {

            background-color: #1d2630;

            border-color: #647181;
        }


        /* Text inputs */

        .stTextInput > div > div > input {

            background-color: #11161e;

            color: #f5f7fa;

            border: 1px solid #303946;

            border-radius: 9px;
        }


        /* Dividers */

        hr {

            border-color: #252c36;
        }


        /* Captions */

        .stCaption {

            color: #8995a4;
        }


        /* Progress bar */

        div[data-testid="stProgressBar"] {

            height: 10px;
        }

        </style>
        """,

        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

def show_header():

    st.title(
        "🚨 AI Stampede Early Warning System"
    )

    st.caption(

        "Real-time crowd intelligence • "
        "Density analysis • "
        "Movement detection • "
        "Predictive risk monitoring"
    )

    st.divider()


# ============================================================
# SIDEBAR
# ============================================================

def show_sidebar():

    with st.sidebar:

        st.markdown(
            "## 🛡️ CROWDGUARD"
        )

        st.caption(
            "AI-Powered Crowd Safety Platform"
        )

        st.divider()


        # ====================================================
        # MONITORING
        # ====================================================

        st.markdown(
            "### 🎥 Monitoring"
        )

        input_mode = st.radio(

            "Select Input Source",

            [
                "Recorded Video",
                "Phone Camera"
            ]
        )

        phone_url = ""


        if input_mode == "Phone Camera":

            phone_url = st.text_input(

                "Phone Camera Stream URL",

                placeholder=(
                    "http://192.168.x.x:8080/video"
                )
            )


        st.divider()


        # ====================================================
        # AI PIPELINE
        # ====================================================

        st.markdown(
            "### 🤖 AI Pipeline"
        )

        st.write(
            "🎯 **YOLOv8m**"
        )

        st.caption(
            "Person detection & tracking"
        )


        st.write(
            "🧠 **CSRNet**"
        )

        st.caption(
            "Crowd density estimation"
        )


        st.write(
            "🏃 **Optical Flow**"
        )

        st.caption(
            "Movement analysis"
        )


        st.write(
            "⚠️ **Risk Engine**"
        )

        st.caption(
            "Stampede risk prediction"
        )


        st.divider()


        # ====================================================
        # HARDWARE
        # ====================================================

        st.markdown(
            "### 💻 Hardware"
        )

        try:

            import torch

            if torch.cuda.is_available():

                st.success(
                    "🟢 GPU ACCELERATION ACTIVE"
                )

                try:

                    gpu_name = (
                        torch.cuda
                        .get_device_name(0)
                    )

                    st.caption(
                        gpu_name
                    )

                except Exception:

                    st.caption(
                        "NVIDIA GPU detected"
                    )

            else:

                st.warning(
                    "🟠 CPU MODE"
                )

        except Exception:

            st.info(
                "Hardware information unavailable"
            )


        st.divider()


        # ====================================================
        # PROCESSING
        # ====================================================

        st.markdown(
            "### ⚙️ Processing"
        )

        # Keep this synchronized with app.py

        st.caption(
            "YOLO interval: 2 frames"
        )

        st.caption(
            "CSRNet interval: 15 frames"
        )

        st.caption(
            "Optical Flow interval: 5 frames"
        )

        st.caption(
            "Dashboard update: 5 frames"
        )


        st.divider()


        # ====================================================
        # VERSION
        # ====================================================

        st.caption(
            "CrowdGuard v1.0"
        )

        st.caption(
            "AI-Based Stampede Early Warning System"
        )


    return input_mode, phone_url


# ============================================================
# SECTION TITLE
# ============================================================

def show_section_title(
    title,
    description=None
):

    st.subheader(title)

    if description:

        st.caption(
            description
        )


# ============================================================
# VIDEO
# ============================================================

def create_video_placeholder():

    return st.empty()


# ============================================================
# METRIC CARDS
# ============================================================

def create_metric_cards():

    return st.columns(
        4,
        gap="medium"
    )


def update_metrics(

    columns,

    people_count,

    peak_people,

    density,

    movement,

    risk_score

):

    columns[0].metric(

        "👥 PEOPLE DETECTED",

        f"{people_count}",

        f"Peak: {peak_people}"
    )


    columns[1].metric(

        "🔥 CROWD DENSITY",

        f"{density:.1f}",

        "CSRNet"
    )


    columns[2].metric(

        "🏃 MOVEMENT",

        f"{movement:.2f}",

        "Optical Flow"
    )


    columns[3].metric(

        "⚠️ RISK SCORE",

        f"{risk_score:.1f}/100",

        "AI Risk Engine"
    )


# ============================================================
# RISK STATUS
# ============================================================

def show_risk_status(

    container,

    risk_level,

    risk_score,

    warning=False

):

    if warning:

        container.error(

            f"""
🚨 STAMPEDE WARNING

Sustained high-risk crowd conditions detected.

Risk Score: {risk_score:.1f}/100
"""
        )


    elif risk_level == "HIGH":

        container.error(

            f"""
🔴 HIGH RISK

Immediate attention recommended.

Risk Score: {risk_score:.1f}/100
"""
        )


    elif risk_level == "MEDIUM":

        container.warning(

            f"""
🟠 MEDIUM RISK

Continue monitoring crowd conditions.

Risk Score: {risk_score:.1f}/100
"""
        )


    else:

        container.success(

            f"""
🟢 LOW RISK

Crowd conditions are currently stable.

Risk Score: {risk_score:.1f}/100
"""
        )


# ============================================================
# RISK SCORE COMPONENTS
# ============================================================

def update_score_cards(
    columns,
    risk
):

    people_score = float(
        risk.get(
            "people_score",
            0
        )
    )

    density_score = float(
        risk.get(
            "density_score",
            0
        )
    )

    movement_score = float(
        risk.get(
            "movement_score",
            0
        )
    )


    columns[0].metric(

        "👥 PEOPLE SCORE",

        f"{people_score:.1f}/100"
    )


    columns[1].metric(

        "🔥 DENSITY SCORE",

        f"{density_score:.1f}/100"
    )


    columns[2].metric(

        "🏃 MOVEMENT SCORE",

        f"{movement_score:.1f}/100"
    )


# ============================================================
# OVERALL RISK PROGRESS
# ============================================================

def show_risk_progress(
    risk_score
):

    st.markdown(
        "#### Overall Risk"
    )


    try:

        score = float(
            risk_score
        )

    except (
        TypeError,
        ValueError
    ):

        score = 0.0


    score = max(
        0.0,
        min(
            score,
            100.0
        )
    )


    st.progress(
        score / 100.0
    )


    if score < 40:

        st.caption(
            f"🟢 LOW — {score:.1f}/100"
        )


    elif score < 70:

        st.caption(
            f"🟠 MEDIUM — {score:.1f}/100"
        )


    else:

        st.caption(
            f"🔴 HIGH — {score:.1f}/100"
        )


# ============================================================
# HISTORY
# ============================================================

def create_history_container():

    return st.empty()


def update_history_chart(

    container,

    history

):

    if not history:

        return


    try:

        import pandas as pd

        df = pd.DataFrame(
            history
        )


        if df.empty:

            return


        numeric_columns = []


        for column in [

            "people",
            "density",
            "movement",
            "risk"

        ]:

            if column in df.columns:

                numeric_columns.append(
                    column
                )


        if not numeric_columns:

            return


        chart_data = df[
            numeric_columns
        ].copy()


        container.line_chart(

            chart_data,

            height=280
        )


    except Exception:

        return


# ============================================================
# ALERT PANEL
# ============================================================

def show_alert_panel(

    container,

    risk_level,

    risk_score,

    warning=False

):

    if warning:

        container.error(

            """
🚨 STAMPEDE WARNING

Immediate crowd-control action recommended.
"""
        )


    elif risk_level == "HIGH":

        container.error(

            """
🔴 HIGH RISK

Dangerous crowd conditions detected.
"""
        )


    elif risk_level == "MEDIUM":

        container.warning(

            """
🟠 MEDIUM RISK

Crowd conditions require continued monitoring.
"""
        )


    else:

        container.success(

            """
🟢 ALL CLEAR

No immediate crowd safety threat detected.
"""
        )


# ============================================================
# MONITORING STATUS
# ============================================================

def show_monitoring_status():

    st.success(

        "🟢 MONITORING ACTIVE — "
        "AI detection and risk analysis are running."
    )


# ============================================================
# SYSTEM INFORMATION
# ============================================================

def show_system_info(

    camera_id,

    fps,

    yolo_frames,

    csrnet_frames,

    backend_status

):

    st.divider()


    show_section_title(

        "📊 System Information",

        "Runtime statistics and processing information"
    )


    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )


    col1.metric(

        "📹 CAMERA",

        str(camera_id)
    )


    col2.metric(

        "⚡ PROCESSING FPS",

        f"{fps:.1f}"
    )


    col3.metric(

        "🎯 YOLO FRAMES",

        str(yolo_frames)
    )


    col4.metric(

        "🧠 CSRNet FRAMES",

        str(csrnet_frames)
    )


    if backend_status:

        col5.success(
            "🟢 BACKEND CONNECTED"
        )

    else:

        col5.error(
            "🔴 BACKEND OFFLINE"
        )


# ============================================================
# FOOTER
# ============================================================

def show_footer():

    st.divider()


    st.caption(

        "CrowdGuard • "
        "AI-Based Stampede Early Warning System"
    )


    st.caption(

        "YOLOv8m • CSRNet • Optical Flow • "
        "Risk Engine • FastAPI • MySQL"
    )