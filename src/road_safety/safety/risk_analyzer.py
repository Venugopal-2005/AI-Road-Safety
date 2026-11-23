"""Heuristic risk analysis and visual hazard alert assessment."""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from ..config import SafetyConfig

if TYPE_CHECKING:
    from ..detection.detector import DetectionResult
    from ..lanes.lane_detector import LaneDetectionResult
    from ..signs.sign_detector import TrafficSignResult


class HazardLevel(enum.Enum):
    """Categorical hazard severity levels."""
    SAFE = 0
    CAUTION = 1
    CRITICAL_WARNING = 2

    @property
    def label(self) -> str:
        return self.name

    @property
    def color_bgr(self) -> Tuple[int, int, int]:
        """Color representation in BGR for visual overlays."""
        if self == HazardLevel.CRITICAL_WARNING:
            return (0, 0, 240)      # Vivid Red
        if self == HazardLevel.CAUTION:
            return (0, 190, 255)    # Amber / Yellow-Orange
        return (0, 200, 0)          # Green


@dataclass
class SafetyAlert:
    """Individual heuristic safety warning."""
    level: HazardLevel
    category: str  # "PEDESTRIAN", "VEHICLE", "LANE_DEPARTURE", "TRAFFIC_CONTROL"
    message: str
    target_box: Optional[Tuple[int, int, int, int]] = None
    track_id: Optional[int] = None
    details: Dict[str, float | str] = field(default_factory=dict)


@dataclass
class RiskAssessment:
    """Consolidated situational risk assessment for an analyzed frame."""
    overall_level: HazardLevel = HazardLevel.SAFE
    alerts: List[SafetyAlert] = field(default_factory=list)
    primary_alert_message: str = "PATH CLEAR - SYSTEM MONITORING"
    disclaimer: str = (
        "PROTOTYPE HEURISTIC ESTIMATE ONLY. Not a calibrated vehicle collision prevention system."
    )

    @property
    def alert_color_bgr(self) -> Tuple[int, int, int]:
        return self.overall_level.color_bgr


class RiskAnalyzer:
    """Evaluates spatial proximity, corridor overlap, and trajectory to generate hazard alerts.

    Note:
        These alerts are visual heuristic estimations derived from 2D pixel geometry,
        not physical 3D radar/lidar distance measurements.
    """

    def __init__(
        self,
        config: Optional[SafetyConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config or SafetyConfig()
        self.logger = logger or logging.getLogger("road_safety.safety")

    def analyze(
        self,
        frame_shape: Tuple[int, int],
        detections: List[DetectionResult],
        lane_result: Optional[LaneDetectionResult] = None,
        signs: Optional[List[TrafficSignResult]] = None,
    ) -> RiskAssessment:
        """Evaluate scene elements and determine active hazard level.

        Args:
            frame_shape: (height, width) of the analyzed frame.
            detections: Tracked/detected objects in the frame.
            lane_result: Output from lane detection pipeline.
            signs: Recognized signs and signals.

        Returns:
            RiskAssessment object containing severity level and descriptive alerts.
        """
        h, w = frame_shape[:2]
        alerts: List[SafetyAlert] = []

        # Define ego vehicle trajectory corridor in pixel coordinates
        corridor_half_w = (w * self.config.ego_corridor_width_ratio) / 2.0
        ego_center_x = w / 2.0
        corridor_x_min = ego_center_x - corridor_half_w
        corridor_x_max = ego_center_x + corridor_half_w

        # 1. Evaluate Pedestrians and Cyclists (Vulnerable Road Users)
        for det in detections:
            if det.is_vulnerable_user:
                bc_x, bc_y = det.bottom_center
                depth_ratio = det.depth_proxy

                in_corridor = corridor_x_min <= bc_x <= corridor_x_max

                if in_corridor and depth_ratio >= self.config.pedestrian_critical_distance_ratio:
                    alerts.append(
                        SafetyAlert(
                            level=HazardLevel.CRITICAL_WARNING,
                            category="PEDESTRIAN",
                            message=f"CRITICAL: {det.class_name.upper()} IN FORWARD PATH",
                            target_box=det.box,
                            track_id=det.track_id,
                            details={
                                "depth_ratio": round(depth_ratio, 2),
                                "in_corridor": True,
                            },
                        )
                    )
                elif depth_ratio >= self.config.pedestrian_caution_distance_ratio:
                    alerts.append(
                        SafetyAlert(
                            level=HazardLevel.CAUTION,
                            category="PEDESTRIAN",
                            message=f"CAUTION: {det.class_name.upper()} NEAR ROADWAY",
                            target_box=det.box,
                            track_id=det.track_id,
                            details={
                                "depth_ratio": round(depth_ratio, 2),
                                "in_corridor": in_corridor,
                            },
                        )
                    )

        # 2. Evaluate Motor Vehicles (Cars, Trucks, Buses, Motorcycles)
        for det in detections:
            if det.is_vehicle:
                bc_x, bc_y = det.bottom_center
                depth_ratio = det.depth_proxy
                in_corridor = corridor_x_min <= bc_x <= corridor_x_max

                if in_corridor and depth_ratio >= self.config.vehicle_critical_distance_ratio:
                    alerts.append(
                        SafetyAlert(
                            level=HazardLevel.CRITICAL_WARNING,
                            category="VEHICLE",
                            message=f"CRITICAL: HIGH PROXIMITY TO FORWARD {det.class_name.upper()}",
                            target_box=det.box,
                            track_id=det.track_id,
                            details={"depth_ratio": round(depth_ratio, 2)},
                        )
                    )
                elif in_corridor and depth_ratio >= self.config.vehicle_caution_distance_ratio:
                    alerts.append(
                        SafetyAlert(
                            level=HazardLevel.CAUTION,
                            category="VEHICLE",
                            message=f"CAUTION: FORWARD {det.class_name.upper()} AHEAD",
                            target_box=det.box,
                            track_id=det.track_id,
                            details={"depth_ratio": round(depth_ratio, 2)},
                        )
                    )

        # 3. Evaluate Lane Departure Warning
        if lane_result and lane_result.lane_departure_warning:
            direction = lane_result.departure_direction
            alerts.append(
                SafetyAlert(
                    level=HazardLevel.CAUTION,
                    category="LANE_DEPARTURE",
                    message=f"LANE DRIFT: DEVIATION TOWARD {direction}",
                    details={"offset_px": round(lane_result.center_offset_px, 1)},
                )
            )

        # 4. Evaluate Traffic Signals and Stop Signs
        if signs:
            for s in signs:
                if s.actionable_warning:
                    alerts.append(
                        SafetyAlert(
                            level=HazardLevel.CAUTION,
                            category="TRAFFIC_CONTROL",
                            message=f"NOTICE: {s.actionable_warning}",
                            target_box=s.box,
                            details={"sign_type": s.sign_type},
                        )
                    )

        # Consolidate overall hazard level
        if any(a.level == HazardLevel.CRITICAL_WARNING for a in alerts):
            overall = HazardLevel.CRITICAL_WARNING
            crit = next(a for a in alerts if a.level == HazardLevel.CRITICAL_WARNING)
            primary_msg = crit.message
        elif any(a.level == HazardLevel.CAUTION for a in alerts):
            overall = HazardLevel.CAUTION
            caut = next(a for a in alerts if a.level == HazardLevel.CAUTION)
            primary_msg = caut.message
        else:
            overall = HazardLevel.SAFE
            primary_msg = "PATH CLEAR - SYSTEM MONITORING"

        return RiskAssessment(
            overall_level=overall,
            alerts=alerts,
            primary_alert_message=primary_msg,
        )
