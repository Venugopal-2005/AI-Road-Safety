"""Traffic sign and signal detection using YOLO COCO classes and color/shape heuristics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config import SignsConfig
from ..detection.detector import DetectionResult
from ..utils.validation import validate_frame


@dataclass
class TrafficSignResult:
    """Represents a detected traffic sign or signal."""
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    sign_type: str
    confidence: float
    source: str  # "YOLO_COCO", "COLOR_SHAPE_HEURISTIC", "CUSTOM_CLASSIFIER"
    shape: Optional[str] = None  # "OCTAGON", "CIRCLE", "TRIANGLE", "RECTANGLE"
    actionable_warning: Optional[str] = None


class TrafficSignDetector:
    """Detects and categorizes road traffic signs and signals.

    Features:
    - Integrates native YOLO COCO classes: Stop Sign (11), Traffic Light (9).
    - Color/geometry contour analysis for potential regulatory/warning sign candidates.
    - Extensible architecture hook for custom GTSRB (German Traffic Sign Recognition Benchmark) models.
    """

    def __init__(
        self,
        config: Optional[SignsConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config or SignsConfig()
        self.logger = logger or logging.getLogger("road_safety.signs")

    def process_from_detections(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
    ) -> List[TrafficSignResult]:
        """Extract traffic control elements from model detections and color heuristics.

        Args:
            frame: OpenCV BGR image array.
            detections: Pre-filtered detections from the primary object detector.

        Returns:
            List of structured TrafficSignResult objects.
        """
        results: List[TrafficSignResult] = []

        # 1. Ingest model-based signs from YOLO COCO
        for det in detections:
            if det.class_id == 11:  # stop sign
                results.append(
                    TrafficSignResult(
                        box=det.box,
                        sign_type="Stop Sign",
                        confidence=det.confidence,
                        source="YOLO_COCO",
                        shape="OCTAGON",
                        actionable_warning="STOP REQUIRED AHEAD",
                    )
                )
            elif det.class_id == 9:  # traffic light
                results.append(
                    TrafficSignResult(
                        box=det.box,
                        sign_type="Traffic Light",
                        confidence=det.confidence,
                        source="YOLO_COCO",
                        shape="RECTANGLE",
                        actionable_warning="MONITOR TRAFFIC SIGNAL",
                    )
                )

        # 2. Optional heuristic color/shape region proposal for red circular/octagonal signs
        if self.config.enable_heuristic_signs:
            heuristic_signs = self._detect_candidate_sign_contours(frame)
            # Add only candidates that do not substantially overlap existing model detections
            for cand in heuristic_signs:
                if not self._overlaps_existing(cand.box, [r.box for r in results]):
                    results.append(cand)

        return results

    def _detect_candidate_sign_contours(self, frame: np.ndarray) -> List[TrafficSignResult]:
        """Detect candidate sign regions based on red color thresholding and polygonal approximation."""
        candidates: List[TrafficSignResult] = []
        is_valid, _ = validate_frame(frame)
        if not is_valid:
            return candidates

        h, w = frame.shape[:2]
        # Restrict heuristic sign search to upper and middle portions of road scene
        roi_h = int(h * 0.70)
        roi = frame[:roi_h, :]

        # Convert to HSV color space
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Red hue wraps around 0 and 180 in OpenCV HSV
        mask1 = cv2.inRange(hsv, np.array([0, 80, 70]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([165, 80, 70]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.config.min_contour_area or area > self.config.max_contour_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = float(bw) / float(bh) if bh > 0 else 0.0

            # Traffic signs generally have near 1:1 aspect ratio (circular, triangular, octagonal)
            if not (0.75 <= aspect_ratio <= 1.35):
                continue

            # Approximate polygonal shape
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            num_vertices = len(approx)

            shape = "UNKNOWN"
            sign_desc = "Candidate Sign"
            if num_vertices == 3:
                shape = "TRIANGLE"
                sign_desc = "Warning/Yield Sign Candidate"
            elif num_vertices == 8 or (num_vertices >= 7 and num_vertices <= 9):
                shape = "OCTAGON"
                sign_desc = "Potential Stop Sign Candidate"
            elif num_vertices > 6:
                shape = "CIRCLE"
                sign_desc = "Regulatory Sign Candidate"

            candidates.append(
                TrafficSignResult(
                    box=(x, y, x + bw, y + bh),
                    sign_type=sign_desc,
                    confidence=0.55,
                    source="COLOR_SHAPE_HEURISTIC",
                    shape=shape,
                    actionable_warning="OBSERVE ROAD SIGN",
                )
            )

        return candidates

    @staticmethod
    def _overlaps_existing(
        box: Tuple[int, int, int, int], existing_boxes: List[Tuple[int, int, int, int]]
    ) -> bool:
        """Check if candidate box significantly overlaps any existing box."""
        bx1, by1, bx2, by2 = box
        b_area = max(1, (bx2 - bx1) * (by2 - by1))

        for ex1, ey1, ex2, ey2 in existing_boxes:
            ix1 = max(bx1, ex1)
            iy1 = max(by1, ey1)
            ix2 = min(bx2, ex2)
            iy2 = min(by2, ey2)

            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                iou = inter_area / float(b_area)
                if iou > 0.30:
                    return True

        return False
