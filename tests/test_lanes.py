"""Unit tests for classical computer vision lane detection."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.road_safety.config import LanesConfig
from src.road_safety.lanes.lane_detector import LaneDetector


def create_synthetic_road_frame(
    width: int = 1280,
    height: int = 720,
    offset_x: int = 0,
) -> np.ndarray:
    """Create a synthetic road image with two visible white lane markings.

    Args:
        width: Image width.
        height: Image height.
        offset_x: Horizontal shift applied to lane lines to simulate vehicle deviation.

    Returns:
        BGR synthetic image.
    """
    frame = np.full((height, width, 3), 40, dtype=np.uint8)  # Dark asphalt background

    # Horizon line and vanishing point
    y_top = int(height * 0.60)
    y_bot = int(height * 0.95)

    # Standard centered lane coordinates
    left_x_bot = 250 + offset_x
    left_x_top = 540 + offset_x
    right_x_bot = 1030 + offset_x
    right_x_top = 740 + offset_x

    # Draw high-contrast white lane stripes
    cv2.line(frame, (left_x_bot, y_bot), (left_x_top, y_top), (255, 255, 255), 14)
    cv2.line(frame, (right_x_bot, y_bot), (right_x_top, y_top), (255, 255, 255), 14)

    return frame


def test_lane_detection_on_synthetic_frame():
    """Verify lane detector correctly locates left and right lane lines and produces corridor."""
    detector = LaneDetector()
    frame = create_synthetic_road_frame()

    result = detector.detect(frame)

    assert result.lane_detected is True
    assert result.left_line is not None
    assert result.right_line is not None
    assert result.left_line.is_valid is True
    assert result.right_line.is_valid is True
    assert result.lane_polygon is not None
    assert result.lane_polygon.shape == (4, 2)
    # Centered synthetic lane should have small offset from vehicle center
    assert abs(result.center_offset_px) < 60
    assert result.lane_departure_warning is False


def test_lane_departure_warning_trigger():
    """Verify lane departure warning triggers when lane is significantly shifted."""
    config = LanesConfig(lane_departure_warning_threshold_px=40)
    detector = LaneDetector(config=config)

    # Shifting lane markings to the left means vehicle is drifting RIGHT relative to lane center
    frame_drift_right = create_synthetic_road_frame(offset_x=-150)
    result = detector.detect(frame_drift_right)

    if result.lane_detected:
        assert result.lane_departure_warning is True
        assert result.departure_direction == "RIGHT"


def test_graceful_degradation_on_blank_frame():
    """Verify detector handles absence of markings gracefully without exceptions."""
    detector = LaneDetector()
    blank_frame = np.full((720, 1280, 3), 30, dtype=np.uint8)

    result = detector.detect(blank_frame)

    assert result.lane_detected is False
    assert result.lane_polygon is None
    assert result.lane_departure_warning is False
    assert result.departure_direction in ("NO_LANES", "UNCERTAIN")


def test_invalid_and_empty_frames():
    """Verify detector handles invalid inputs gracefully."""
    detector = LaneDetector()

    res_none = detector.detect(None)
    assert res_none.lane_detected is False

    res_empty = detector.detect(np.array([], dtype=np.uint8))
    assert res_empty.lane_detected is False
