# 🚨 AI-Based Stampede Early Warning System

An AI-powered computer vision system designed to **detect abnormal crowd behavior and provide early warnings for potential stampede situations**.

The system analyzes live or recorded video using **YOLO-based person detection, CSRNet-based crowd density estimation, and Optical Flow-based movement analysis**. These signals are combined by a risk engine to estimate crowd risk and help authorities take preventive action before a critical situation develops.

---

## 🎯 Problem Statement

Stampedes can occur when large crowds experience sudden increases in density, uncontrolled movement, panic, or bottlenecks.

Traditional surveillance systems primarily depend on human monitoring, which can make it difficult to identify dangerous crowd conditions early enough.

This project aims to provide an **automated AI-assisted early warning mechanism** that continuously monitors:

* 👥 Number of people
* 📊 Crowd density
* 🏃 Crowd movement
* ⚠️ Abnormal movement patterns
* 🔥 High-risk crowd zones
* 🚨 Overall stampede risk

The goal is to provide an **early indication of potentially dangerous crowd conditions**, allowing authorities to respond before the situation escalates.

---

## ✨ Key Features

### 👥 People Detection

Uses **YOLO** to detect and count people in the monitored area.

* Real-time person detection
* Person-only detection using the COCO `person` class
* Crowd counting
* Tracking of detected individuals
* Peak crowd count monitoring

### 📊 Crowd Density Estimation

Uses **CSRNet** to estimate crowd density, particularly in situations where simple object detection may become less reliable because of heavy crowding or occlusion.

### 🏃 Movement Analysis

Uses **Optical Flow** to analyze crowd movement.

The system can identify:

* Movement intensity
* Average movement speed
* Sudden changes in movement
* Unusual movement patterns

### 🔥 Dynamic Crowd Heatmap

The system generates a dynamic heatmap showing areas experiencing significant crowd activity.

This helps identify:

* Crowd concentration
* High-activity regions
* Potential bottlenecks
* Areas requiring attention

### ⚠️ Risk Assessment

Multiple signals are combined to estimate the current crowd risk.

The risk engine considers factors such as:

* Crowd size
* Crowd density
* Movement intensity
* Abnormal movement
* Crowd concentration

The dashboard presents the resulting risk level in an easy-to-understand format.

### 📹 Multiple Input Modes

The system supports:

* 🎥 Pre-recorded video
* 📱 Phone/IP camera feed
* 📷 Live camera monitoring

### 📊 Emergency Operations Dashboard

A Streamlit-based dashboard provides a centralized view of the emergency situation.

The dashboard includes:

* Current people count
* Crowd density
* Movement level
* Risk level
* Alerts
* Heatmaps
* Emergency zones
* Response teams
* Shelters
* Hospitals
* Infrastructure
* Evacuation routes
* Field reports
* Sensors
* Resources
* Analytics
* AI intelligence
* Activity feed

---

## 🧠 System Architecture

```text
                    VIDEO INPUT
                         │
              ┌──────────┴──────────┐
              │                     │
        Recorded Video         Live Camera
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Preprocessing  │
                └────────┬────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       YOLO           CSRNet       Optical Flow
          │              │              │
          ▼              ▼              ▼
    People Count    Crowd Density   Movement Data
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                ┌─────────────────┐
                │   Risk Engine   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Risk Assessment │
                └────────┬────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Alert Generation       Dashboard
```

---

## 🛠️ Technologies Used

| Technology       | Purpose                              |
| ---------------- | ------------------------------------ |
| **Python**       | Core development                     |
| **YOLOv8**       | Person detection and tracking        |
| **CSRNet**       | Crowd density estimation             |
| **OpenCV**       | Video processing and computer vision |
| **Optical Flow** | Crowd movement analysis              |
| **PyTorch**      | Deep learning framework              |
| **ByteTrack**    | Person tracking                      |
| **NumPy**        | Numerical processing                 |
| **Streamlit**    | Emergency operations dashboard       |
| **CUDA**         | GPU acceleration                     |
| **Git/GitHub**   | Version control                      |

---

## 📁 Project Structure

```text
AI-Stampede-Early-Warning-System/
│
├── dashboard/
│   ├── app.py
│   └── dashboard_ui.py
│
├── csrnet/
│   ├── csrnet_model.py
│   └── test_csrnet.py
│
├── crowd.mp4
│
├── people_count.py
├── movement_analysis.py
├── optical_flow.py
├── heatmap.py
├── live_heatmap.py
├── dynamic_heatmap.py
├── track_people.py
├── stampede_detection.py
│
├── risk_engine.py
│
├── yolo_live.py
├── test_yolo.py
├── test_yolo8m.py
│
├── yolo_models/
│   ├── yolov8n.pt
│   └── yolov8m.pt
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the System

Start the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser.

---

## 🎥 Video Input

The system can process a recorded video such as:

```text
crowd.mp4
```

A live camera/IP camera can also be configured as the video source.

For an IP camera, the source can be configured using the camera's streaming URL.

Example:

```python
camera_url = "http://YOUR_PHONE_IP:PORT/video"
```

---

## 🚀 GPU Acceleration

The system can use an NVIDIA GPU through CUDA to accelerate deep-learning inference.

GPU acceleration is particularly useful for:

* YOLO inference
* CSRNet inference
* Real-time video processing

The project was tested with an **NVIDIA RTX 3050 Laptop GPU** and CUDA-enabled PyTorch.

You can verify CUDA availability with:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## 🔬 AI Components

### 1. YOLO

YOLO is responsible for detecting people within each video frame.

```text
Frame
  ↓
YOLO
  ↓
Person Detection
  ↓
People Count
```

The system uses the COCO `person` class for crowd monitoring.

---

### 2. CSRNet

CSRNet is used for estimating crowd density.

Unlike simple person detection, density estimation can provide useful information when people are heavily packed together and individual detection becomes difficult.

```text
Crowded Frame
      ↓
    CSRNet
      ↓
Density Map
      ↓
Crowd Density
```

---

### 3. Optical Flow

Optical Flow estimates how pixels move between consecutive frames.

This allows the system to analyze crowd movement.

```text
Frame N
   +
Frame N+1
   ↓
Optical Flow
   ↓
Movement Vectors
   ↓
Movement Intensity
```

Sudden or highly irregular movement can contribute to an increased risk assessment.

---

### 4. Person Tracking

Detected people can be tracked across frames using **ByteTrack**.

Tracking helps calculate:

* Unique people
* Crowd movement
* Persistent detections
* Peak crowd count

---

## ⚠️ Risk Engine

The Risk Engine combines different crowd parameters into an overall risk assessment.

Conceptually:

```text
People Count
      +
Crowd Density
      +
Movement Intensity
      +
Abnormal Movement
      ↓
   Risk Engine
      ↓
Risk Score / Risk Level
```

The system can categorize situations into levels such as:

```text
🟢 LOW
🟡 MEDIUM
🟠 HIGH
🔴 CRITICAL
```

The exact thresholds can be configured according to the deployment environment.

---

## 📈 Dashboard

The emergency dashboard provides a centralized operational view.

### Monitoring

```text
People Count
Crowd Density
Movement Level
Risk Level
```

### Emergency Intelligence

```text
Alerts
Field Reports
Sensors
AI Analysis
Activity Feed
```

### Emergency Resources

```text
Response Teams
Hospitals
Shelters
Infrastructure
Resources
Evacuation Routes
```

This allows the system to act as an **AI-assisted command and monitoring interface** rather than only a computer vision demo.

---

## 🗺️ Emergency Scenario

The dashboard includes a centralized simulated emergency scenario based on a **Belagavi flood emergency**.

The simulation includes:

* Emergency zones
* Alerts
* Response teams
* Shelters
* Hospitals
* Infrastructure
* Field reports
* Evacuation routes
* Sensors
* Emergency resources
* Analytics
* AI intelligence
* Activity monitoring

The scenario demonstrates how AI-generated crowd intelligence can be incorporated into a broader emergency-management platform.

---

## 🎯 Expected Outcome

The system is designed to provide an **early indication of dangerous crowd conditions** rather than waiting for a stampede to occur.

Potential applications include:

* 🛕 Religious gatherings
* 🏟️ Stadiums
* 🎤 Concerts
* 🚉 Railway stations
* 🚌 Transportation hubs
* 🏛️ Public events
* 🚨 Emergency evacuation
* 🎪 Large-scale festivals

---

## 🔮 Future Improvements

Possible future improvements include:

* Real-time cloud deployment
* Multi-camera monitoring
* Edge-device deployment
* Automated SMS/email alerts
* Integration with CCTV networks
* Improved crowd behavior classification
* Transformer-based crowd analysis
* More advanced anomaly detection
* Geographic risk mapping
* Historical incident analysis
* Mobile application for emergency responders
* Integration with IoT sensors
* Automatic evacuation recommendations

---

## ⚠️ Limitations

The system is intended as an **AI-assisted early warning and decision-support system**.

It should not be considered a replacement for trained emergency personnel.

Accuracy can be affected by:

* Camera angle
* Lighting conditions
* Occlusion
* Extremely dense crowds
* Video quality
* Camera movement
* Weather conditions

Real-world deployment would require extensive validation using diverse crowd datasets and controlled testing.

---

## 👩‍💻 Project Developed By

**Vedika Gornal**

B.E. Computer Science & Engineering

Visvesvaraya Technological University (VTU)

---

## ⭐ Project Highlights

> **Detect → Analyze → Assess → Alert**

The core objective of this project is to move from **reactive crowd management to proactive crowd safety** by using Artificial Intelligence and Computer Vision to identify potentially dangerous conditions before they become critical.
