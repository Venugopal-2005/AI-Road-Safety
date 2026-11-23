"""End-to-end smoke test script for image and video processing pipeline."""

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
from src.road_safety.config import get_default_config
from src.road_safety.processing.pipeline import RoadSafetyPipeline
from src.road_safety.processing.video_io import VideoProcessor


def test_smoke_image_pipeline():
    print("[1/3] Running Image Pipeline Smoke Test...")
    config = get_default_config()
    pipeline = RoadSafetyPipeline(config=config)

    sample_img_path = REPO_ROOT / "assets" / "sample_images" / "highway_lanes.jpg"
    assert sample_img_path.is_file(), f"Missing test image: {sample_img_path}"

    processed = pipeline.process_image(sample_img_path)
    assert processed.annotated_frame is not None
    assert processed.annotated_frame.shape == (720, 1280, 3)
    assert processed.lane_result is not None
    assert processed.lane_result.lane_detected is True
    print(f" -> Image processed successfully! Latency: {processed.latency_ms:.1f}ms")

    out_img = REPO_ROOT / "outputs" / "smoke_test_image.jpg"
    out_img.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), processed.annotated_frame)
    assert out_img.is_file() and out_img.stat().st_size > 0
    print(f" -> Saved annotated image to: {out_img}")


def test_smoke_video_pipeline():
    print("[2/3] Running Video Pipeline Smoke Test...")
    config = get_default_config()
    pipeline = RoadSafetyPipeline(config=config)
    processor = VideoProcessor(pipeline=pipeline)

    sample_vid_path = REPO_ROOT / "assets" / "sample_videos" / "sample_road_driving.mp4"
    assert sample_vid_path.is_file(), f"Missing test video: {sample_vid_path}"

    out_vid = REPO_ROOT / "outputs" / "smoke_test_video.mp4"
    out_vid.parent.mkdir(parents=True, exist_ok=True)

    meta = processor.process_video(
        input_path=sample_vid_path,
        output_path=out_vid,
        frame_stride=2,
    )
    assert out_vid.is_file() and out_vid.stat().st_size > 0
    print(f" -> Processed {meta.total_frames} frames to {out_vid} (size: {out_vid.stat().st_size} bytes)")


def test_app_compilation():
    print("[3/3] Checking app.py syntax and import compatibility...")
    app_path = REPO_ROOT / "app.py"
    with open(app_path, "r", encoding="utf-8") as f:
        code = f.read()
    compile(code, "app.py", "exec")
    print(" -> app.py syntax is clean and valid!")


if __name__ == "__main__":
    test_smoke_image_pipeline()
    test_smoke_video_pipeline()
    test_app_compilation()
    print("\n✅ All smoke tests PASSED successfully!")
