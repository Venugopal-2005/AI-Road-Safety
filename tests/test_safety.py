"""Unit tests for heuristic safety risk analysis and alert generation."""

from __future__ import annotations

import pytest

from src.road_safety.config import SafetyConfig
from src.road_safety.detection.detector import DetectionResult
from src.road_safety.lanes.lane_detector import LaneDetectionResult
from src.road_safety.safety.risk_analyzer import HazardLevel, RiskAnalyzer
from src.road_safety.signs.sign_detector import TrafficSignResult


@pytest.fixture
def risk_analyzer():
    cfg = SafetyConfig(
        ego_corridor_width_ratio=0.36,
        pedestrian_critical_distance_ratio=0.70,
        pedestrian_caution_distance_ratio=0.50,
        vehicle_critical_distance_ratio=0.78,
        vehicle_caution_distance_ratio=0.58,
    )
    return RiskAnalyzer(config=cfg)


def test_clear_path_safe(risk_analyzer):
    """Empty road scene should evaluate to SAFE."""
    assessment = risk_analyzer.analyze(
        frame_shape=(720, 1280),
        detections=[],
    )
    assert assessment.overall_level == HazardLevel.SAFE
    assert len(assessment.alerts) == 0
    assert "CLEAR" in assessment.primary_alert_message.upper()
    assert "PROTOTYPE HEURISTIC" in assessment.disclaimer.upper()


def test_pedestrian_critical_in_path(risk_analyzer):
    """Pedestrian centered in forward ego corridor and close to camera triggers CRITICAL."""
    # Frame width 1280, height 720. Center is x=640.
    # Pedestrian at bottom center (640, 580) -> depth proxy = 580/720 = 0.805 >= 0.70
    ped = DetectionResult(
        box=(600, 400, 680, 580),
        class_id=0,
        class_name="person",
        confidence=0.91,
        bottom_center=(640, 580),
        depth_proxy=580.0 / 720.0,
    )

    assessment = risk_analyzer.analyze(
        frame_shape=(720, 1280),
        detections=[ped],
    )

    assert assessment.overall_level == HazardLevel.CRITICAL_WARNING
    assert any(a.category == "PEDESTRIAN" for a in assessment.alerts)
    assert "CRITICAL" in assessment.primary_alert_message


def test_pedestrian_caution_distance(risk_analyzer):
    """Pedestrian at moderate distance (depth_proxy >= 0.50, < 0.70) triggers CAUTION."""
    ped = DetectionResult(
        box=(300, 250, 360, 400),
        class_id=0,
        class_name="person",
        confidence=0.88,
        bottom_center=(330, 400),
        depth_proxy=400.0 / 720.0,  # ~0.555
    )

    assessment = risk_analyzer.analyze(
        frame_shape=(720, 1280),
        detections=[ped],
    )

    assert assessment.overall_level == HazardLevel.CAUTION
    assert any(a.category == "PEDESTRIAN" for a in assessment.alerts)


def test_vehicle_critical_proximity(risk_analyzer):
    """Lead vehicle directly in front with large proximity triggers CRITICAL."""
    # Vehicle centered at (640, 600) -> depth_proxy = 600/720 = 0.833 >= 0.78
    car = DetectionResult(
        box=(490, 380, 790, 600),
        class_id=2,
        class_name="car",
        confidence=0.94,
        bottom_center=(640, 600),
        depth_proxy=600.0 / 720.0,
    )

    assessment = risk_analyzer.analyze(
        frame_shape=(720, 1280),
        detections=[car],
    )

    assert assessment.overall_level == HazardLevel.CRITICAL_WARNING
    assert any(a.category == "VEHICLE" for a in assessment.alerts)


def test_lane_departure_alert_integration(risk_analyzer):
    """Lane departure warning creates a CAUTION alert."""
    lane_res = LaneDetectionResult(
        lane_detected=True,
        lane_departure_warning=True,
        departure_direction="RIGHT",
        center_offset_px=65.0,
    )

    assessment = risk_analyzer.analyze(
        frame_shape=(720, 1280),
        detections=[],
        lane_result=lane_res,
    )

    assert assessment.overall_level == HazardLevel.CAUTION
    assert any(a.category == "LANE_DEPARTURE" for a in assessment.alerts)
    assert "RIGHT" in assessment.primary_alert_message


def test_traffic_sign_alert_integration(risk_analyzer):
    """Traffic sign warning triggers notice alert."""
    sign = TrafficSignResult(
        box=(800, 100, 880, 180),
        sign_type="Stop Sign",
        confidence=0.90,
        source="YOLO_COCO",
        actionable_warning="STOP REQUIRED AHEAD",
    )

    assessment = risk_analyzer.analyze(
        frame_shape=(720, 1280),
        detections=[],
        signs=[sign],
    )

    assert assessment.overall_level == HazardLevel.CAUTION
    assert any(a.category == "TRAFFIC_CONTROL" for a in assessment.alerts)
