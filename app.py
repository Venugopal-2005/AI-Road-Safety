"""Streamlit Web Application: AI-Powered Road Safety and Pedestrian Detection System.

Developed for the OpenCV AI Competition 2026 powered by AWS.
Integrates real-world BDD100K dataset images as demonstration presets.
"""

from __future__ import annotations

import io
import os
import random
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Project root-relative path resolution (works from any working directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# Real-world asset paths (BDD100K + videos)
BDD100K_DIR = PROJECT_ROOT / "assets" / "real_images" / "bdd100k"
REAL_VIDEOS_DIR = PROJECT_ROOT / "assets" / "real_videos"
SAMPLE_VIDEOS_DIR = PROJECT_ROOT / "assets" / "sample_videos"

# Fall back model path
MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"
MODEL_PATH_ALT = PROJECT_ROOT / "models" / "yolov8n.pt"

# ---------------------------------------------------------------------------
# Lazy import of project src (after path resolution is set)
# ---------------------------------------------------------------------------
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.road_safety.config import AppConfig, get_default_config
from src.road_safety.detection.detector import DetectionResult
from src.road_safety.processing.pipeline import ProcessedFrame, RoadSafetyPipeline
from src.road_safety.processing.video_io import VideoProcessor, VideoReader, VideoWriter
from src.road_safety.safety.risk_analyzer import HazardLevel

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Road Safety System | OpenCV AI Competition 2026",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2c 0%, #121620 100%);
        border: 1px solid #2a3346;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #8b9bb4;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .alert-banner-safe {
        background-color: #143823;
        border-left: 5px solid #22c55e;
        padding: 12px 16px;
        border-radius: 4px;
        color: #86efac;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .alert-banner-caution {
        background-color: #422d06;
        border-left: 5px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 4px;
        color: #fde68a;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .alert-banner-critical {
        background-color: #450a0a;
        border-left: 5px solid #ef4444;
        padding: 12px 16px;
        border-radius: 4px;
        color: #fca5a5;
        font-weight: 700;
        margin-bottom: 15px;
    }
    .stDownloadButton button {
        background-color: #2563eb !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
    }
    .dataset-badge {
        background-color: #1e3a5f;
        border: 1px solid #2563eb;
        border-radius: 4px;
        padding: 4px 10px;
        color: #93c5fd;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state for Analytics
# ---------------------------------------------------------------------------
def _init_session_state():
    defaults = {
        "total_frames_analyzed": 0,
        "total_detections": 0,
        "total_pedestrians": 0,
        "total_vehicles": 0,
        "total_critical_alerts": 0,
        "total_caution_alerts": 0,
        "confidence_sum": 0.0,
        "confidence_count": 0,
        "latency_history": [],
        "fps_history": [],
        "alert_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _record_processed_frame(processed: ProcessedFrame, source_label: str = ""):
    """Update session analytics with results from a processed frame."""
    st.session_state["total_frames_analyzed"] += 1
    peds = sum(1 for d in processed.detections if d.is_vulnerable_user)
    vehs = sum(1 for d in processed.detections if d.is_vehicle)
    st.session_state["total_detections"] += len(processed.detections)
    st.session_state["total_pedestrians"] += peds
    st.session_state["total_vehicles"] += vehs

    for d in processed.detections:
        st.session_state["confidence_sum"] += d.confidence
        st.session_state["confidence_count"] += 1

    if processed.risk:
        if processed.risk.overall_level == HazardLevel.CRITICAL_WARNING:
            st.session_state["total_critical_alerts"] += 1
            if source_label:
                st.session_state["alert_log"].append(
                    {"level": "CRITICAL", "msg": processed.risk.primary_alert_message, "src": source_label}
                )
        elif processed.risk.overall_level == HazardLevel.CAUTION:
            st.session_state["total_caution_alerts"] += 1

    if processed.latency_ms > 0:
        st.session_state["latency_history"].append(processed.latency_ms)
        # Keep rolling window
        if len(st.session_state["latency_history"]) > 300:
            st.session_state["latency_history"] = st.session_state["latency_history"][-300:]

    if processed.fps > 0:
        st.session_state["fps_history"].append(processed.fps)
        if len(st.session_state["fps_history"]) > 300:
            st.session_state["fps_history"] = st.session_state["fps_history"][-300:]


# ---------------------------------------------------------------------------
# BDD100K Image Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_bdd100k_image_list() -> List[Path]:
    """Return sorted list of available BDD100K real-world images."""
    if BDD100K_DIR.is_dir():
        images = sorted(BDD100K_DIR.glob("*.jpg"))
        return images
    return []


def pick_random_bdd100k_images(n: int = 12) -> List[Path]:
    images = get_bdd100k_image_list()
    if not images:
        return []
    random.seed(99)
    return random.sample(images, min(n, len(images)))


# ---------------------------------------------------------------------------
# Pipeline Loader (cached per config parameters)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI vision pipeline...")
def load_pipeline(
    conf: float,
    iou: float,
    enable_lanes: bool,
    enable_tracking: bool,
    model_path_str: str,
) -> RoadSafetyPipeline:
    """Initialize and cache the core road safety vision pipeline."""
    config = get_default_config(PROJECT_ROOT / "configs" / "config.yaml")
    config.detection.confidence_threshold = conf
    config.detection.iou_threshold = iou
    config.detection.enable_tracking = enable_tracking
    config.lanes.enabled = enable_lanes
    # Resolve model path
    mp = Path(model_path_str)
    if mp.is_file():
        config.detection.model_path = str(mp)
        config.detection.model_name = mp.name
    return RoadSafetyPipeline(config=config)


# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------
def render_risk_banner(processed: ProcessedFrame):
    risk = processed.risk
    if not risk:
        return
    if risk.overall_level == HazardLevel.CRITICAL_WARNING:
        st.markdown(
            f"<div class='alert-banner-critical'>🚨 {risk.primary_alert_message}</div>",
            unsafe_allow_html=True,
        )
    elif risk.overall_level == HazardLevel.CAUTION:
        st.markdown(
            f"<div class='alert-banner-caution'>⚠️ {risk.primary_alert_message}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='alert-banner-safe'>✅ {risk.primary_alert_message}</div>",
            unsafe_allow_html=True,
        )


def render_detection_metrics(processed: ProcessedFrame):
    ped_count = sum(1 for d in processed.detections if d.is_vulnerable_user)
    veh_count = sum(1 for d in processed.detections if d.is_vehicle)
    sign_count = len(processed.signs) if processed.signs else 0

    lane_status_str = "KEEPING"
    if processed.lane_result:
        if not processed.lane_result.lane_detected:
            lane_status_str = "NOT DETECTED"
        elif processed.lane_result.lane_departure_warning:
            lane_status_str = f"DRIFT {processed.lane_result.departure_direction}"

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("🚶 Vulnerable Users", ped_count)
    m2.metric("🚗 Motor Vehicles", veh_count)
    m3.metric("🚦 Signs / Signals", sign_count)
    m4.metric("🛣️ Lane Status", lane_status_str)
    m5.metric("⏱️ Latency", f"{processed.latency_ms:.1f} ms")
    status_label = processed.risk.overall_level.label if processed.risk else "SAFE"
    m6.metric("⚠️ Hazard Level", status_label)


def render_detections_table(detections: List[DetectionResult]):
    if not detections:
        st.info("No objects detected in this frame.")
        return
    table_data = []
    for d in detections:
        table_data.append({
            "Track ID": d.track_id if d.track_id is not None else "-",
            "Class": d.class_name,
            "Type": "VRU (Pedestrian/Cyclist)" if d.is_vulnerable_user else ("Vehicle" if d.is_vehicle else "Sign/Signal"),
            "Confidence": f"{d.confidence:.0%}",
            "Box (x1,y1,x2,y2)": str(d.box),
            "Depth Proxy": f"{d.depth_proxy:.2f}",
        })
    st.dataframe(table_data, use_container_width=True)


def render_header():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🚗 AI-Powered Road Safety & Pedestrian Detection System")
        st.markdown(
            "**OpenCV AI Competition 2026 powered by AWS** | Real-Time Scene Analysis, "
            "Lane Keeping Assistance & Explainable VRU Hazard Warnings.\n\n"
            "📦 **Dataset**: Berkeley DeepDrive BDD100K (100K real-world road images)"
        )
    with col2:
        st.markdown(
            """
            <div style="text-align: right; padding-top: 10px;">
                <span style="background-color: #ff9900; color: #111; font-weight: bold; padding: 4px 8px; border-radius: 4px;">AWS Powered</span>&nbsp;
                <span style="background-color: #0077b6; color: #fff; font-weight: bold; padding: 4px 8px; border-radius: 4px;">OpenCV 2026</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        "> ⚠️ **PROTOTYPE DISCLAIMER**: *This application provides visual heuristic warnings based on 2D camera geometry "
        "and is not a certified vehicle collision prevention or ADAS safety-critical system.*"
    )
    st.divider()


# ---------------------------------------------------------------------------
# TAB: Dashboard Overview
# ---------------------------------------------------------------------------
def tab_dashboard_overview(pipeline: RoadSafetyPipeline):
    st.subheader("🏠 System Dashboard & Overview")

    # System status cards
    col1, col2, col3, col4 = st.columns(4)
    import torch
    cuda_ok = torch.cuda.is_available()

    with col1:
        st.markdown(
            f"""<div class='metric-card'>
            <div class='metric-title'>Execution Device</div>
            <div class='metric-value'>{'🟢 CUDA GPU' if cuda_ok else '🟡 CPU'}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        model_ok = pipeline.detector.is_loaded
        st.markdown(
            f"""<div class='metric-card'>
            <div class='metric-title'>YOLO Model Status</div>
            <div class='metric-value'>{'✅ Loaded' if model_ok else '❌ Not Loaded'}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        bdd_count = len(get_bdd100k_image_list())
        st.markdown(
            f"""<div class='metric-card'>
            <div class='metric-title'>BDD100K Demo Images</div>
            <div class='metric-value'>🖼️ {bdd_count}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        video_ok = (REAL_VIDEOS_DIR / "urban_intersection.mp4").is_file() or \
                   (SAMPLE_VIDEOS_DIR / "sample_road_driving.mp4").is_file()
        st.markdown(
            f"""<div class='metric-card'>
            <div class='metric-title'>Demo Video</div>
            <div class='metric-value'>{'✅ Available' if video_ok else '❌ Missing'}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.divider()

    # Feature overview
    st.markdown("### 🔑 Implemented Features")
    feat_col1, feat_col2 = st.columns(2)
    with feat_col1:
        st.markdown("""
        | Feature | Status |
        |:--------|:------:|
        | YOLOv8 Pedestrian & Cyclist Detection | ✅ |
        | YOLOv8 Vehicle Detection (Car, Bus, Truck) | ✅ |
        | Multi-Object Centroid Tracker (IDs) | ✅ |
        | Bbox Expansion Rate (Closing Speed Proxy) | ✅ |
        | Classical Canny + Hough Lane Detection | ✅ |
        | EMA Temporal Lane Smoothing (α=0.70) | ✅ |
        """)
    with feat_col2:
        st.markdown("""
        | Feature | Status |
        |:--------|:------:|
        | Lane Departure Warning (LDW) | ✅ |
        | COCO Traffic Sign Detection | ✅ |
        | HSV Color/Shape Sign Heuristics | ✅ |
        | 3-Tier SAFE/CAUTION/CRITICAL Alerts | ✅ |
        | Automotive Telemetry HUD Overlay | ✅ |
        | BDD100K Real-World Dataset Integration | ✅ |
        """)

    st.divider()

    # BDD100K sample gallery
    st.markdown("### 🖼️ BDD100K Real-World Road Dataset Preview")
    st.markdown(
        "<span class='dataset-badge'>📦 Berkeley DeepDrive BDD100K — Real Road Footage</span>",
        unsafe_allow_html=True,
    )

    gallery_images = pick_random_bdd100k_images(8)
    if gallery_images:
        cols = st.columns(4)
        for i, img_path in enumerate(gallery_images[:8]):
            with cols[i % 4]:
                img = Image.open(img_path)
                st.image(img, caption=img_path.stem[:20], use_container_width=True)
    else:
        st.info("Run `python scripts/setup_real_data.py` to populate BDD100K demo images.")

    st.divider()
    st.markdown("### 🔖 Project Disclaimer & Attribution")
    st.info(
        "🔬 **Research Prototype**: This system uses visual heuristics from 2D monocular cameras. "
        "All depth estimates and proximity alerts are geometric approximations — not physically calibrated "
        "collision distance measurements. Do not use for autonomous vehicle control.\n\n"
        "📚 **Dataset Attribution**: BDD100K images courtesy of Berkeley DeepDrive. "
        "Licensed under BDD100K Dataset License Agreement."
    )


# ---------------------------------------------------------------------------
# TAB: Image Analysis
# ---------------------------------------------------------------------------
def tab_image_analysis(pipeline: RoadSafetyPipeline):
    st.subheader("📷 Road Scene Image Analysis")

    upload_col, preset_col = st.columns([2, 1])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload a road scene image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="image_uploader",
            help="Upload any dashcam or road scene image.",
        )

    # Build real-world BDD100K preset list
    bdd_images = get_bdd100k_image_list()
    preset_options = ["None (Upload your own)"]
    bdd_name_map: Dict[str, Path] = {}

    if bdd_images:
        with preset_col:
            # Show first 20 BDD100K images as named presets
            sample_presets = sorted(bdd_images)[:20]
            for p in sample_presets:
                label = f"BDD100K: {p.stem}"
                preset_options.append(label)
                bdd_name_map[label] = p

    with preset_col:
        selected_preset = st.selectbox(
            "Or choose a real-world BDD100K demo image:",
            preset_options,
            help="Select from real-world road scene images from the BDD100K dataset.",
        )

    target_frame: Optional[np.ndarray] = None
    source_label = ""

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        target_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        source_label = uploaded_file.name
    elif selected_preset != "None (Upload your own)" and selected_preset in bdd_name_map:
        img_path = bdd_name_map[selected_preset]
        target_frame = cv2.imread(str(img_path))
        source_label = img_path.name
        st.markdown(
            "<span class='dataset-badge'>📦 BDD100K Real-World Road Image</span>",
            unsafe_allow_html=True,
        )

    if target_frame is not None:
        with st.spinner("Analyzing road scene with AI vision pipeline..."):
            processed = pipeline.process_image(target_frame)

        _record_processed_frame(processed, source_label)

        render_risk_banner(processed)
        render_detection_metrics(processed)

        # Side-by-side image display
        img_c1, img_c2 = st.columns(2)
        with img_c1:
            st.markdown("**📸 Original Input Frame**")
            st.image(cv2.cvtColor(processed.original_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        with img_c2:
            st.markdown("**🎯 Annotated Road Safety Output (HUD Overlay)**")
            st.image(cv2.cvtColor(processed.annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Download annotated image
        is_success, buffer = cv2.imencode(".jpg", processed.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if is_success:
            st.download_button(
                label="📥 Download Annotated Image",
                data=buffer.tobytes(),
                file_name="annotated_road_safety.jpg",
                mime="image/jpeg",
            )

        # Detailed detections
        with st.expander("🔍 Detected Objects Detail Table"):
            render_detections_table(processed.detections)

        # Safety alerts breakdown
        if processed.risk and processed.risk.alerts:
            with st.expander("🚨 Active Safety Alerts Detail"):
                for alert in processed.risk.alerts:
                    color = "red" if alert.level == HazardLevel.CRITICAL_WARNING else ("orange" if alert.level == HazardLevel.CAUTION else "green")
                    st.markdown(f"- **:{color}[{alert.level.label}]** — {alert.message}")

    elif bdd_images:
        st.markdown(
            "<span class='dataset-badge'>📦 BDD100K Real-World Road Images Available — Select a preset above</span>",
            unsafe_allow_html=True,
        )
        st.info("👆 Upload your own road scene image or select a BDD100K real-world preset above.")
        # Show mini gallery
        gallery = pick_random_bdd100k_images(4)
        if gallery:
            g_cols = st.columns(4)
            for i, gp in enumerate(gallery):
                with g_cols[i]:
                    st.image(str(gp), caption=gp.stem[:18], use_container_width=True)
    else:
        st.info("👆 Upload a road scene image to begin analysis. Run `python scripts/setup_real_data.py` to load BDD100K demo images.")


# ---------------------------------------------------------------------------
# TAB: Video Processing
# ---------------------------------------------------------------------------
def tab_video_processing(pipeline: RoadSafetyPipeline, frame_stride: int):
    st.subheader("🎥 Video File Batch Processing")
    v_col1, v_col2 = st.columns([2, 1])

    with v_col1:
        uploaded_video = st.file_uploader(
            "Upload a driving video (.mp4, .avi, .mov)",
            type=["mp4", "avi", "mov"],
            key="video_uploader",
        )

    # Build local video presets
    local_vids: Dict[str, Path] = {}
    real_vid = REAL_VIDEOS_DIR / "urban_intersection.mp4"
    sample_vid = SAMPLE_VIDEOS_DIR / "sample_road_driving.mp4"
    if real_vid.is_file():
        local_vids["Urban Intersection (Real Traffic Video)"] = real_vid
    if sample_vid.is_file():
        local_vids["Sample Road Driving (Synthetic Demo)"] = sample_vid

    vid_preset_options = ["None (Upload custom video)"] + list(local_vids.keys())

    with v_col2:
        selected_vid_preset = st.selectbox(
            "Or choose a local demo video:",
            vid_preset_options,
        )

    video_input_path: Optional[str] = None

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()
        video_input_path = tfile.name
        st.info(f"📂 Loaded uploaded video: `{uploaded_video.name}`")
    elif selected_vid_preset != "None (Upload custom video)" and selected_vid_preset in local_vids:
        video_input_path = str(local_vids[selected_vid_preset])
        st.info(f"📂 Loaded local video: `{local_vids[selected_vid_preset].name}`")
        if "urban_intersection" in selected_vid_preset.lower() or "real" in selected_vid_preset.lower():
            st.markdown(
                "<span class='dataset-badge'>🎬 Real-World Traffic Video</span>",
                unsafe_allow_html=True,
            )

    if video_input_path:
        # Show video metadata
        try:
            import cv2 as _cv2
            cap = _cv2.VideoCapture(video_input_path)
            vw = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
            vfps = cap.get(_cv2.CAP_PROP_FPS)
            vtot = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            meta_col1.metric("Resolution", f"{vw}×{vh}")
            meta_col2.metric("Source FPS", f"{vfps:.1f}")
            meta_col3.metric("Total Frames", vtot)
        except Exception:
            pass

        start_process = st.button("🚀 Start Video Processing Pipeline", type="primary")

        if start_process:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            output_temp.close()
            output_path = output_temp.name

            processor = VideoProcessor(pipeline=pipeline)

            frame_analytics = {"pedestrians": 0, "vehicles": 0, "critical": 0, "caution": 0, "latencies": []}

            def update_progress(curr: int, total: int, lat: float):
                frac = min(1.0, float(curr) / float(max(1, total)))
                progress_bar.progress(frac)
                status_text.text(
                    f"Processing frame {curr}/{total} ({frac * 100:.1f}%) | Latency: {lat:.1f} ms"
                )
                frame_analytics["latencies"].append(lat)

            t0 = time.time()
            try:
                meta = processor.process_video(
                    input_path=video_input_path,
                    output_path=output_path,
                    frame_stride=frame_stride,
                    progress_callback=update_progress,
                )
                elapsed = time.time() - t0
                progress_bar.progress(1.0)
                status_text.empty()
                st.success(
                    f"✅ Video processing complete in **{elapsed:.1f}s** — "
                    f"{meta.total_frames} source frames at {meta.fps:.1f} fps."
                )

                # Play and download processed video
                try:
                    with open(output_path, "rb") as vf:
                        video_bytes = vf.read()
                    st.video(video_bytes)
                    st.download_button(
                        label="📥 Download Annotated Video (.mp4)",
                        data=video_bytes,
                        file_name="annotated_road_safety.mp4",
                        mime="video/mp4",
                    )
                except Exception as ve:
                    st.warning(f"Video playback error (file still downloadable): {ve}")
                    with open(output_path, "rb") as vf:
                        st.download_button(
                            label="📥 Download Annotated Video (.mp4)",
                            data=vf.read(),
                            file_name="annotated_road_safety.mp4",
                            mime="video/mp4",
                        )

                # Update session analytics
                avg_lat = np.mean(frame_analytics["latencies"]) if frame_analytics["latencies"] else 0
                st.session_state["total_frames_analyzed"] += meta.total_frames
                if avg_lat > 0:
                    st.session_state["latency_history"].extend(frame_analytics["latencies"][-50:])

            except Exception as e:
                st.error(f"❌ Error during video processing: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("👆 Upload a video file or select a local demo video to begin.")


# ---------------------------------------------------------------------------
# TAB: Live Webcam
# ---------------------------------------------------------------------------
def tab_live_webcam(pipeline: RoadSafetyPipeline):
    st.subheader("📹 Live Camera Feed Analysis")
    st.markdown(
        "Connect a USB or laptop webcam and click **Start Webcam** for real-time "
        "object detection, lane visualization, and pedestrian safety alerts."
    )

    cam_index = st.number_input("Camera Device Index", min_value=0, max_value=10, value=0, step=1)
    webcam_active = st.checkbox("▶ Start Webcam Capture", value=False)

    webcam_display = st.empty()
    hud_metrics = st.empty()

    if webcam_active:
        cap = cv2.VideoCapture(int(cam_index))
        if not cap.isOpened():
            st.warning(
                f"⚠️ Unable to access camera device {cam_index}. "
                "Ensure your camera is connected and permissions are granted."
            )
            return

        stop_button = st.button("⏹ Stop Stream")
        pipeline.reset()
        frame_count = 0

        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret or frame is None:
                st.warning("Failed to grab camera frame.")
                break

            processed = pipeline.process_frame(frame)
            _record_processed_frame(processed, f"webcam_frame_{frame_count}")
            frame_count += 1

            rgb_frame = cv2.cvtColor(processed.annotated_frame, cv2.COLOR_BGR2RGB)
            webcam_display.image(rgb_frame, use_container_width=True, caption=f"Frame {frame_count}")

            # Live telemetry
            ped_c = sum(1 for d in processed.detections if d.is_vulnerable_user)
            veh_c = sum(1 for d in processed.detections if d.is_vehicle)
            level = processed.risk.overall_level.label if processed.risk else "SAFE"
            hud_metrics.markdown(
                f"**FPS**: `{processed.fps:.1f}` | **VRU**: `{ped_c}` | "
                f"**VEH**: `{veh_c}` | **Alert**: `{level}` | **Latency**: `{processed.latency_ms:.0f}ms`"
            )

            time.sleep(0.03)

        cap.release()
        st.info("Webcam stream stopped.")
    else:
        st.info("Enable the checkbox above to start the live webcam feed.")


# ---------------------------------------------------------------------------
# TAB: Analytics
# ---------------------------------------------------------------------------
def tab_analytics():
    st.subheader("📊 Session Analytics & Detection Statistics")

    ss = st.session_state
    total_frames = ss.get("total_frames_analyzed", 0)
    total_det = ss.get("total_detections", 0)
    total_ped = ss.get("total_pedestrians", 0)
    total_veh = ss.get("total_vehicles", 0)
    total_crit = ss.get("total_critical_alerts", 0)
    total_caut = ss.get("total_caution_alerts", 0)
    conf_sum = ss.get("confidence_sum", 0.0)
    conf_cnt = ss.get("confidence_count", 0)
    lat_hist = ss.get("latency_history", [])
    fps_hist = ss.get("fps_history", [])

    if total_frames == 0:
        st.info(
            "📈 No frames have been analyzed yet in this session.\n\n"
            "Go to **Image Analysis** or **Video Processing** to start generating statistics."
        )
        return

    # Summary Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎞️ Total Frames Analyzed", total_frames)
    c2.metric("🔍 Total Detections", total_det)
    c3.metric("🚶 Total VRU Detections", total_ped)
    c4.metric("🚗 Total Vehicle Detections", total_veh)

    c5, c6, c7, c8 = st.columns(4)
    avg_conf = (conf_sum / conf_cnt * 100) if conf_cnt > 0 else 0.0
    avg_lat = np.mean(lat_hist) if lat_hist else 0.0
    avg_fps = np.mean(fps_hist) if fps_hist else 0.0
    det_per_frame = total_det / total_frames if total_frames > 0 else 0.0

    c5.metric("🎯 Avg Confidence", f"{avg_conf:.1f}%")
    c6.metric("⏱️ Avg Latency", f"{avg_lat:.1f} ms")
    c7.metric("⚡ Avg FPS", f"{avg_fps:.1f}")
    c8.metric("📦 Avg Objects/Frame", f"{det_per_frame:.1f}")

    st.divider()

    # Safety alert summary
    st.markdown("### 🚨 Safety Alert Summary")
    a1, a2, a3 = st.columns(3)
    a1.metric("🔴 Critical Warnings", total_crit)
    a2.metric("🟡 Caution Alerts", total_caut)
    safe_frames = total_frames - total_crit - total_caut
    a3.metric("🟢 Safe Frames", max(0, safe_frames))

    # Recent alert log
    alert_log = ss.get("alert_log", [])
    if alert_log:
        st.markdown("### 📋 Recent Critical Alert Log")
        for entry in alert_log[-10:][::-1]:
            st.markdown(f"- 🔴 **{entry['level']}** — {entry['msg']} *(source: `{entry['src']}`)*")

    # Latency chart
    if lat_hist:
        st.divider()
        st.markdown("### ⏱️ Processing Latency History")
        import pandas as pd
        df = pd.DataFrame({"Latency (ms)": lat_hist[-150:]})
        st.line_chart(df)

    # FPS chart
    if fps_hist:
        st.markdown("### ⚡ Pipeline FPS History")
        import pandas as pd
        df2 = pd.DataFrame({"FPS": fps_hist[-150:]})
        st.line_chart(df2)

    # Reset button
    st.divider()
    if st.button("🔄 Reset Session Analytics"):
        for k in [
            "total_frames_analyzed", "total_detections", "total_pedestrians",
            "total_vehicles", "total_critical_alerts", "total_caution_alerts",
            "confidence_sum", "confidence_count", "latency_history", "fps_history", "alert_log"
        ]:
            st.session_state[k] = [] if isinstance(st.session_state.get(k), list) else (0.0 if isinstance(st.session_state.get(k), float) else 0)
        st.success("Session analytics reset.")
        st.rerun()


# ---------------------------------------------------------------------------
# TAB: System Architecture & Docs
# ---------------------------------------------------------------------------
def tab_architecture():
    st.subheader("📐 System Architecture & Competition Blueprint")

    st.markdown("""
### Pipeline Architecture Flow

```
INPUT (Image / Video / Webcam)
     │
     ▼
Validation & Normalization Layer
     │
     ├──────────────────────────────────────┐
     │                                      │
     ▼                                      ▼
YOLOv8 Object Detector              Classical Lane Detector
(COCO: Person, Bike, Car,          (Grayscale → Gaussian Blur →
 Moto, Bus, Truck, Signs)           Canny Edge → ROI Mask →
     │                               HoughLinesP → Polyfit x(y))
     ▼                                      │
Centroid & Trajectory Tracker               ▼
(Track IDs + Bbox Growth Rate)        Lane Center + EMA Smoothing
     │                                      │
     └──────────────────┬───────────────────┘
                        │
                        ▼
                Traffic Sign Engine
              (COCO Signs + HSV Heuristics)
                        │
                        ▼
            Heuristic Risk Analyzer
         (Ego-Corridor Proximity Test,
          Depth Proxy, 3-Tier Alerts)
                        │
                        ▼
         Visualization & HUD Renderer
       (Lane Polygon, Boxes, Telemetry)
                        │
                        ▼
         Streamlit Dashboard + Download
```
""")

    st.markdown("---")
    st.markdown("""
### Implemented vs Future Work

| Feature | Status | Technology |
|:--------|:------:|:-----------|
| Pedestrian & Cyclist Detection | ✅ | YOLOv8n COCO (class 0, 1) |
| Vehicle Detection | ✅ | YOLOv8n COCO (class 2, 3, 5, 7) |
| Lane Detection & LDW | ✅ | Canny + Hough + EMA Smoothing |
| Multi-Object Tracking | ✅ | Centroid Distance Tracker |
| Traffic Sign Detection | ✅ | COCO + HSV Shape Heuristics |
| 3-Tier Hazard Classification | ✅ | Geometric Ego-Corridor Analysis |
| BDD100K Dataset Integration | ✅ | 8,927 Real-World Road Images |
| Automotive HUD Overlay | ✅ | OpenCV Alpha-Blended Rendering |
| Custom GTSRB Sign Classifier | ⏳ | Extensible hook ready |
| Monocular Metric Depth | ⏳ | Requires MiDaS / Depth-Anything |
| LiDAR/Radar Sensor Fusion | ⏳ | Multi-modal calibrated hardware |
| AWS ECS/Fargate Deployment | ⏳ | Docker container ready |
""")

    # Read ARCHITECTURE.md if available
    arch_path = PROJECT_ROOT / "docs" / "ARCHITECTURE.md"
    if arch_path.is_file():
        with st.expander("📄 Full Architecture Specification (ARCHITECTURE.md)"):
            st.markdown(arch_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    _init_session_state()
    render_header()

    # -------------------------------------------------------------------------
    # Sidebar Controls
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ System Configuration")

        import torch
        cuda_ok = torch.cuda.is_available()
        device_badge = "🟢 CUDA GPU" if cuda_ok else "🟡 CPU (Optimized)"
        st.markdown(f"**Execution Device**: {device_badge}")

        st.subheader("Object Detection")
        # Resolve default model path
        default_model = str(MODEL_PATH) if MODEL_PATH.is_file() else (
            str(MODEL_PATH_ALT) if MODEL_PATH_ALT.is_file() else "yolov8n.pt"
        )
        model_path_input = st.text_input("Model Path", value=default_model, help="Path to .pt YOLO weights file.")
        conf_threshold = st.slider("Confidence Threshold", 0.10, 0.90, 0.35, 0.05)
        iou_threshold = st.slider("IoU NMS Threshold", 0.20, 0.80, 0.45, 0.05)

        st.subheader("Vision Features")
        enable_lanes = st.checkbox("Lane-Marking Detection", value=True)
        enable_tracking = st.checkbox("Multi-Object Tracking (IDs)", value=True)
        show_hud = st.checkbox("Show HUD Telemetry Bar", value=True)
        show_ego_corridor = st.checkbox("Show Ego Path Corridor", value=True)

        st.subheader("Performance")
        frame_stride = st.slider(
            "Video Frame Stride",
            1, 4, 1,
            help="1 = Process every frame. 2 = Skip every other frame (2× faster on CPU).",
        )

        st.divider()
        st.markdown("### 📚 Docs")
        st.markdown("- Architecture Guide: `docs/ARCHITECTURE.md`")
        st.markdown("- AWS Deployment: `docs/AWS_DEPLOYMENT.md`")
        st.markdown("- Dataset Setup: `python scripts/setup_real_data.py`")

        st.divider()
        st.markdown("### 🧪 Session Stats")
        st.metric("Frames Analyzed", st.session_state.get("total_frames_analyzed", 0))
        st.metric("Total Detections", st.session_state.get("total_detections", 0))

    # Load pipeline (cached)
    pipeline = load_pipeline(
        conf_threshold, iou_threshold, enable_lanes, enable_tracking, model_path_input
    )
    pipeline.visualizer.config.show_telemetry_hud = show_hud
    pipeline.visualizer.config.show_ego_corridor = show_ego_corridor

    # -------------------------------------------------------------------------
    # Navigation Tabs
    # -------------------------------------------------------------------------
    tab_dash, tab_img, tab_vid, tab_cam, tab_anal, tab_docs = st.tabs([
        "🏠 Dashboard",
        "📷 Image Analysis",
        "🎥 Video Processing",
        "📹 Live Camera",
        "📊 Analytics",
        "📐 Architecture",
    ])

    with tab_dash:
        tab_dashboard_overview(pipeline)

    with tab_img:
        tab_image_analysis(pipeline)

    with tab_vid:
        tab_video_processing(pipeline, frame_stride)

    with tab_cam:
        tab_live_webcam(pipeline)

    with tab_anal:
        tab_analytics()

    with tab_docs:
        tab_architecture()


if __name__ == "__main__":
    main()
