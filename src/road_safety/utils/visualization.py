"""Heads-Up Display (HUD) and computer vision visualization utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import numpy as np

from ..config import VisualizationConfig
from ..safety.risk_analyzer import HazardLevel

if TYPE_CHECKING:
    from ..detection.detector import DetectionResult
    from ..lanes.lane_detector import LaneDetectionResult
    from ..safety.risk_analyzer import RiskAssessment
    from ..signs.sign_detector import TrafficSignResult


class Visualizer:
    """Renders telemetry, safety alerts, lane polygons, and bounding boxes onto frames."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()

    def draw_scene(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        lane_result: Optional[LaneDetectionResult] = None,
        signs: Optional[List[TrafficSignResult]] = None,
        risk: Optional[RiskAssessment] = None,
        fps: float = 0.0,
        latency_ms: float = 0.0,
        device_label: str = "CPU",
    ) -> np.ndarray:
        """Render complete road safety annotations and HUD onto a copy of the input frame."""
        output = frame.copy()
        h, w = output.shape[:2]

        # 1. Draw Lane Corridor and Boundaries
        if self.config.show_lane_overlay and lane_result:
            output = self._draw_lanes(output, lane_result)

        # 2. Draw Ego Trajectory Corridor Guidelines
        if self.config.show_ego_corridor:
            output = self._draw_ego_corridor(output)

        # 3. Draw Object Bounding Boxes & Tracking
        if self.config.show_bounding_boxes and detections:
            output = self._draw_detections(output, detections, risk)

        # 4. Draw Traffic Signs & Signals
        if signs:
            output = self._draw_signs(output, signs)

        # 5. Draw Top Telemetry Banner & Safety Alerts HUD
        if self.config.show_telemetry_hud:
            output = self._draw_hud(output, detections, lane_result, risk, fps, latency_ms, device_label)

        return output

    def _draw_lanes(self, frame: np.ndarray, lane: LaneDetectionResult) -> np.ndarray:
        """Draw lane corridor polygon and boundary lines."""
        if lane.lane_detected and lane.lane_polygon is not None:
            overlay = frame.copy()
            corridor_color = (0, 200, 0) if not lane.lane_departure_warning else (0, 165, 255)
            cv2.fillPoly(overlay, [lane.lane_polygon], corridor_color)
            cv2.addWeighted(overlay, self.config.overlay_alpha, frame, 1.0 - self.config.overlay_alpha, 0, frame)

        # Draw left boundary line
        if lane.left_line and lane.left_line.is_valid:
            pt1, pt2 = lane.left_line.get_points()
            cv2.line(frame, pt1, pt2, (255, 180, 0), 3, cv2.LINE_AA)

        # Draw right boundary line
        if lane.right_line and lane.right_line.is_valid:
            pt1, pt2 = lane.right_line.get_points()
            cv2.line(frame, pt1, pt2, (255, 180, 0), 3, cv2.LINE_AA)

        return frame

    def _draw_ego_corridor(self, frame: np.ndarray) -> np.ndarray:
        """Draw dashed guidelines representing the immediate forward vehicle path."""
        h, w = frame.shape[:2]
        cw = int(w * 0.36 / 2)
        center_x = w // 2
        y_start = int(h * 0.65)
        y_end = int(h * 0.95)

        cv2.line(frame, (center_x - cw, y_start), (center_x - cw, y_end), (200, 200, 200), 1, cv2.LINE_4)
        cv2.line(frame, (center_x + cw, y_start), (center_x + cw, y_end), (200, 200, 200), 1, cv2.LINE_4)
        return frame

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        risk: Optional[RiskAssessment],
    ) -> np.ndarray:
        """Draw labeled bounding boxes with risk-dependent styling."""
        crit_boxes = set()
        caut_boxes = set()
        if risk:
            for alert in risk.alerts:
                if alert.target_box:
                    if alert.level == HazardLevel.CRITICAL_WARNING:
                        crit_boxes.add(alert.target_box)
                    elif alert.level == HazardLevel.CAUTION:
                        caut_boxes.add(alert.target_box)

        for det in detections:
            x1, y1, x2, y2 = det.box

            if det.box in crit_boxes:
                box_color = (0, 0, 240)       # Red (Critical hazard)
                thickness = 3
            elif det.box in caut_boxes:
                box_color = (0, 190, 255)     # Amber/Yellow (Caution)
                thickness = 2
            elif det.is_vulnerable_user:
                box_color = (255, 120, 0)     # Blue-Cyan for pedestrians
                thickness = 2
            else:
                box_color = (0, 220, 100)     # Soft Green for regular vehicles
                thickness = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
            cv2.circle(frame, det.bottom_center, 4, box_color, -1)

            tag_parts = [det.class_name]
            if det.track_id is not None:
                tag_parts.insert(0, f"ID:{det.track_id}")
            if self.config.show_confidence:
                tag_parts.append(f"{int(det.confidence * 100)}%")

            label = " | ".join(tag_parts)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

            tag_y1 = max(0, y1 - th - baseline - 4)
            tag_y2 = y1
            cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 6, tag_y2), box_color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 3, y1 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255) if box_color != (0, 190, 255) else (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return frame

    def _draw_signs(self, frame: np.ndarray, signs: List[TrafficSignResult]) -> np.ndarray:
        """Draw traffic signs and signals."""
        for sign in signs:
            x1, y1, x2, y2 = sign.box
            color = (200, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"[SIGN] {sign.sign_type}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, label, (x1, max(15, y1 - 5)), font, 0.45, color, 1, cv2.LINE_AA)

        return frame

    def _draw_hud(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        lane_result: Optional[LaneDetectionResult],
        risk: Optional[RiskAssessment],
        fps: float,
        latency_ms: float,
        device_label: str,
    ) -> np.ndarray:
        """Render top telemetry HUD bar with active safety alerts."""
        h, w = frame.shape[:2]
        hud_h = 58

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        cv2.line(frame, (0, hud_h), (w, hud_h), (50, 55, 68), 1)

        overall_level = risk.overall_level if risk else HazardLevel.SAFE
        status_color = overall_level.color_bgr
        status_text = overall_level.label

        badge_w = 150
        cv2.rectangle(frame, (10, 10), (10 + badge_w, 48), status_color, -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, status_text, (20, 35), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        alert_msg = risk.primary_alert_message if risk else "PATH CLEAR"
        cv2.putText(frame, alert_msg, (175, 35), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        ped_count = sum(1 for d in detections if d.is_vulnerable_user)
        veh_count = sum(1 for d in detections if d.is_vehicle)

        lane_status = "KEEPING"
        if lane_result:
            if not lane_result.lane_detected:
                lane_status = "NO LANES"
            elif lane_result.lane_departure_warning:
                lane_status = f"DRIFT {lane_result.departure_direction}"

        stats_line1 = f"VRU: {ped_count} | VEH: {veh_count} | LANE: {lane_status}"
        stats_line2 = f"FPS: {fps:.1f} | {latency_ms:.1f}ms | {device_label}"

        (sw1, _), _ = cv2.getTextSize(stats_line1, font, 0.45, 1)
        (sw2, _), _ = cv2.getTextSize(stats_line2, font, 0.42, 1)

        cv2.putText(frame, stats_line1, (w - max(sw1, sw2) - 15, 25), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, stats_line2, (w - max(sw1, sw2) - 15, 46), font, 0.42, (160, 170, 185), 1, cv2.LINE_AA)

        return frame
