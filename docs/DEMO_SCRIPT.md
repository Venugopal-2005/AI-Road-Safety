# Competition Video Demo Script: AI-Road-Safety System

**OpenCV AI Competition 2026 powered by AWS**  
**Target Duration**: 75–90 seconds  
**Format**: Screen capture recording with voiceover commentary

---

### [00:00 - 00:15] Introduction & Competition Hook

**Visual**:
- Display Streamlit web application running at `http://localhost:8501`.
- Camera focuses on the header: *"AI-Powered Road Safety & Pedestrian Detection System — OpenCV AI Competition 2026 powered by AWS"*.

**Spoken Audio**:
> *"Welcome! This is our entry for the OpenCV AI Competition 2026 powered by AWS: the AI-Powered Road Safety and Pedestrian Detection System. Every year, millions of collisions occur due to driver distraction and delayed hazard perception. Our goal was to build a modular, explainable computer vision prototype that combines deep learning object detection with classical geometric lane perception to protect vulnerable road users and alert drivers to imminent road hazards."*

---

### [00:15 - 00:35] Core Computer Vision & Image Analysis

**Visual**:
- Navigate to **"Single Image Analysis"** tab.
- Select the preset **"Urban Pedestrians (urban_pedestrian.jpg)"** or upload a driving scene.
- Show side-by-side view: original vs annotated output with HUD overlay.
- Point out the red critical alert banner: `🚨 CRITICAL: PEDESTRIAN IN FORWARD PATH`.
- Highlight telemetry counters: Vulnerable Users: 1, Lane Status: KEEPING, Latency in ms.

**Spoken Audio**:
> *"Here in the image analysis view, the system processes a road scene. Using an Ultralytics YOLOv8 detector running on CPU, we detect pedestrians, cyclists, and vehicles. Simultaneously, OpenCV performs edge detection, trapezoidal region-of-interest masking, and line fitting to extract the vehicle's driving corridor. Notice how our heuristic risk engine detects this pedestrian directly in the forward ego path, instantly escalating the telemetry HUD to a vivid CRITICAL warning."*

---

### [00:35 - 00:55] Dynamic Video Processing & Lane Keeping

**Visual**:
- Switch to **"Video File Processing"** tab.
- Select `sample_road_driving.mp4` and click **"Start Video Processing Pipeline"**.
- Show the frame-by-frame progress bar advancing smoothly.
- Play the finished video: show the green lane corridor dynamically tracking the road, vehicle centroid tracking IDs, and lane departure alerts triggering when the vehicle drifts.

**Spoken Audio**:
> *"In video mode, the system processes dashcam footage frame by frame. Our centroid tracker maintains object identities and computes bounding box growth rates. Classical Hough transforms identify lane boundaries with exponential moving average smoothing to eliminate road jitter. If the car drifts beyond our safety threshold, the system triggers an immediate Lane Drift Alert. Once complete, drivers can download the annotated MP4 directly from the browser."*

---

### [00:55 - 01:15] Architecture, Cloud Deployment & Safety Scope

**Visual**:
- Switch to **"Architecture & AWS Blueprint"** tab.
- Highlight the pipeline data flow diagram, modular class architecture, and the AWS deployment path (ECS/Fargate, S3 telemetry).
- Briefly show the automated unit test suite (`pytest`) executing with 18 passing tests in the terminal.

**Spoken Audio**:
> *"Under the hood, the system is designed with strict modularity, type hints, and full test coverage—18 automated unit tests verify our algorithms without needing GPU or internet access. For cloud scalability, we provide full Docker containerization and documented AWS deployment blueprints for EC2 and ECS. We clearly distinguish our visual heuristic warnings from certified collision avoidance systems, creating an explainable, robust foundation for future ADAS research."*

---

### [01:15 - 01:25] Conclusion & Call to Action

**Visual**:
- Return to the home screen showing the telemetry dashboard.
- Display GitHub repository link and OpenCV Competition attribution badge.

**Spoken Audio**:
> *"Thank you to the OpenCV team and AWS for empowering developers to make our roads safer. Check out our open-source codebase and architecture documentation on GitHub!"*
