# AI-Powered Road Safety and Pedestrian Detection System
Live : https://venugopal-road-safety.streamlit.app/

[![OpenCV AI Competition 2026](https://img.shields.io/badge/OpenCV%20AI%20Competition-2026%20AWS-orange.svg)](https://opencv.org/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Tests: 18 Passing](https://img.shields.io/badge/Tests-18%20Passing-brightgreen.svg)](tests/)
[![Model: YOLOv8 Nano](https://img.shields.io/badge/Model-YOLOv8%20Nano-blueviolet.svg)](https://github.com/ultralytics/ultralytics)
[![Dataset: BDD100K](https://img.shields.io/badge/Dataset-BDD100K%208927%20Images-teal.svg)](https://bdd-data.berkeley.edu/)

> **Submission for the OpenCV AI Competition 2026 powered by AWS**
> An explainable, real-time computer vision system combining deep learning object detection with
> classical geometric lane analysis to provide situational awareness and VRU hazard warnings.
> Demonstrated on **real-world BDD100K road footage**.

---

> [!WARNING]
> **SAFETY & PROTOTYPE DISCLAIMER**
> *This system is a research prototype developed for competition and educational purposes.
> It generates visual heuristic estimates derived from 2D camera geometry and does not provide
> calibrated physical 3D distances or certified collision prevention. It must not be deployed
> as an autonomous safety-critical vehicle control system.*

---

## 🌟 Key Features

| Capability | Status | Implementation |
|:-----------|:------:|:--------------|
| **Pedestrian & Cyclist Detection** | ✅ | YOLOv8n (COCO class 0 & 1) |
| **Vehicle Detection** | ✅ | Cars, Motorcycles, Buses, Trucks (COCO 2, 3, 5, 7) |
| **Multi-Object Tracking** | ✅ | Centroid distance tracker with IDs & bbox growth rate |
| **Classical Lane Detection** | ✅ | Grayscale → Gaussian → Canny → ROI → HoughLinesP → Polyfit |
| **EMA Temporal Smoothing** | ✅ | α=0.70 for jitter-free lane visualization |
| **Lane Departure Warning (LDW)** | ✅ | Lateral offset threshold (default 50px) |
| **Traffic Sign Detection** | ✅ | COCO Stop Signs / Traffic Lights + HSV Heuristics |
| **3-Tier Hazard Alerts** | ✅ | SAFE / CAUTION / CRITICAL_WARNING |
| **Automotive HUD Overlay** | ✅ | Alpha-blended telemetry banner |
| **Streamlit Dashboard** | ✅ | 6-tab dark-mode interface |
| **BDD100K Real Dataset** | ✅ | 8,927 real-world road scene images |
| **Session Analytics** | ✅ | Detection stats, FPS, latency charts |
| **Batch Video Processing** | ✅ | MP4 input → annotated MP4 download |
| **Live Webcam Mode** | ✅ | Real-time stream analysis |

---

## 🏗️ System Architecture

```
INPUT (Image / Video / Webcam)
     │
     ▼
Validation & Normalization Layer
     │
     ├─────────────────────────────────────┐
     │                                     │
     ▼                                     ▼
YOLOv8n Object Detector           Classical Lane Detector
(COCO: Person, Bike, Car,         (Grayscale → Gaussian →
 Moto, Bus, Truck, Signs)          Canny → ROI Mask →
     │                              HoughLinesP → Polyfit x(y))
     ▼                                     │
Centroid & Trajectory Tracker             ▼
(Track IDs + Bbox Growth Rate)      Lane Center + EMA Smoothing
     │                                     │
     └────────────────┬────────────────────┘
                      │
                      ▼
              Traffic Sign Engine
            (COCO + HSV Heuristics)
                      │
                      ▼
          Heuristic Risk Analyzer
         (Ego-Corridor, Depth Proxy)
                      │
                      ▼
        HUD Visualizer + Streamlit
```

---

## ⚡ Quickstart

### Prerequisites
- Python 3.11 (Windows 11 / Linux / macOS)
- Git

### 1. Clone & Create Virtual Environment

```powershell
# Windows PowerShell
git clone https://github.com/your-org/AI-Road-Safety.git
cd AI-Road-Safety
python -m venv D:\PROJECTS\OPENCV\venv
D:\PROJECTS\OPENCV\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
# Install PyTorch CPU (required first on Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

### 3. Set Up Real-World Dataset

```powershell
# BDD100K images are already in assets/sample_images/archive/bdd100k/
# Copy 50 curated demo images to assets/real_images/bdd100k/
python scripts/setup_real_data.py
```

### 4. Launch the Streamlit Dashboard

```powershell
# From the project root with venv activated:
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📂 Project Structure

```
AI-Road-Safety/
├── app.py                              # Streamlit dashboard (6 tabs)
├── configs/
│   └── config.yaml                     # Centralized configuration
├── assets/
│   ├── real_images/
│   │   └── bdd100k/                    # 50 curated BDD100K real images
│   ├── real_videos/
│   │   └── urban_intersection.mp4      # Real-world traffic video
│   └── sample_videos/
│       └── sample_road_driving.mp4     # Synthetic fallback video
├── src/road_safety/
│   ├── config.py                       # AppConfig & dataclasses
│   ├── logging_config.py               # Structured logger setup
│   ├── detection/
│   │   ├── detector.py                 # YOLOv8 wrapper + DetectionResult
│   │   └── tracker.py                  # SimpleCentroidTracker
│   ├── lanes/
│   │   └── lane_detector.py            # Canny/Hough lane pipeline
│   ├── safety/
│   │   └── risk_analyzer.py            # 3-tier hazard assessor
│   ├── signs/
│   │   └── sign_detector.py            # Traffic sign engine
│   ├── processing/
│   │   ├── pipeline.py                 # RoadSafetyPipeline
│   │   └── video_io.py                 # VideoReader/Writer/Processor
│   └── utils/
│       ├── validation.py               # Frame/file validators
│       └── visualization.py            # HUD + bounding box renderer
├── tests/
│   ├── test_config.py                  # 4 config tests
│   ├── test_validation.py              # 4 validation tests
│   ├── test_lanes.py                   # 4 lane detection tests
│   ├── test_safety.py                  # 6 risk analyzer tests
│   └── run_smoke_test.py               # End-to-end BDD100K smoke test
├── scripts/
│   ├── setup_real_data.py              # BDD100K → real_images setup
│   └── generate_samples.py             # Synthetic test asset generator
├── models/
│   └── yolov8n.pt                      # YOLOv8 Nano weights (6.5 MB)
├── docs/
│   ├── ARCHITECTURE.md                 # Technical spec
│   ├── AWS_DEPLOYMENT.md               # ECS/Fargate deployment guide
│   ├── TESTING.md                      # Test protocols
│   └── DEMO_SCRIPT.md                  # 75-90s competition video script
├── Dockerfile                          # Python 3.11-slim container
├── docker-compose.yml
└── requirements.txt
```

---

## 🧪 Running Tests

```powershell
# Full test suite (18 tests)
D:\PROJECTS\OPENCV\venv\Scripts\python.exe -m pytest tests/ -v

# End-to-end smoke test on a real BDD100K image
D:\PROJECTS\OPENCV\venv\Scripts\python.exe tests/run_smoke_test.py
```

**Expected output**: `18 passed` + annotated output in `outputs/smoke_test_output.jpg`

---

## 📊 Dashboard Tabs

| Tab | Purpose |
|:----|:--------|
| **🏠 Dashboard** | System status, BDD100K gallery, feature overview |
| **📷 Image Analysis** | Upload or select BDD100K real images for analysis |
| **🎥 Video Processing** | Upload or select local videos for batch processing |
| **📹 Live Camera** | Real-time webcam detection and lane visualization |
| **📊 Analytics** | Session detection stats, alert log, FPS/latency charts |
| **📐 Architecture** | Pipeline diagram, features table, full ARCHITECTURE.md |

---

## 📦 Dataset Attribution

**Berkeley DeepDrive BDD100K** — 100,000 real-world road scene images
- Citation: *Yu, F. et al. (2020). BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning. CVPR.*
- License: BDD100K Dataset License Agreement

**Sample Road Video** — `assets/sample_videos/sample_road_driving.mp4`
- For Pexels urban intersection video: https://www.pexels.com/video/36926154/

---

## 🚀 Known Limitations

1. **First inference latency**: YOLOv8 cold-start on CPU takes ~10-15s. Subsequent frames ~300-800ms.
2. **Lane detection accuracy**: Requires clearly visible lane markings. Fades, shadows, and night scenes reduce accuracy.
3. **Depth estimates**: All proximity alerts are 2D perspective heuristics — not physically calibrated metric distances.
4. **Live webcam in Streamlit**: Due to Streamlit's server-client architecture, webcam access requires running locally (not on cloud).

---

## 🗺️ Future Roadmap

- [ ] Monocular depth estimation (MiDaS / Depth-Anything v2) for metric-scale distance
- [ ] GTSRB fine-tuned traffic sign classifier (43 classes)
- [ ] TensorRT / ONNX optimization for Jetson Orin edge deployment
- [ ] AWS ECS/Fargate cloud deployment (Dockerfile ready)

---

*OpenCV AI Competition 2026 — Powered by AWS*
