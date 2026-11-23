# Pretrained Models Directory

This directory stores deep learning model weights used by the road safety vision pipeline.

## Default Object Detector: YOLOv8 Nano (`yolov8n.pt`)

The system uses Ultralytics YOLOv8 nano by default, pre-trained on the COCO dataset (80 classes, including pedestrians, cars, bicycles, buses, trucks, motorcycles, traffic lights, and stop signs).

### Model Licensing & Attribution
- **Architecture**: Ultralytics YOLOv8 / YOLO11
- **License**: AGPL-3.0 (Ultralytics Open Source) or Enterprise License.
- **Dataset**: COCO (Common Objects in Context), Creative Commons Attribution 4.0 License.

### Automated Download (Online)
When connected to the internet, running the application will automatically download `yolov8n.pt` (approx. 6.2 MB) to the working directory or cache upon first inference:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Manual Download (Offline or Air-Gapped Environments)
If working in an offline environment or air-gapped machine:
1. Download `yolov8n.pt` from the official Ultralytics GitHub release:
   `https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt`
2. Place the file directly in this folder:
   `models/yolov8n.pt`
3. The system automatically searches `models/yolov8n.pt`, `yolov8n.pt`, and system caches.

### Fallback Behavior
If weights are completely absent and the machine is offline:
- The `ObjectDetector` logs a warning and enters graceful fallback mode.
- Lane detection, classical sign color/contour analysis, and input validation continue to operate without crashing.
- The Streamlit interface displays an informative banner with step-by-step instructions.
