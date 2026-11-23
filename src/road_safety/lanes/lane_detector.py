"""Classical computer vision lane detection using Canny edge, ROI masking, and Hough transforms."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config import LanesConfig
from ..utils.validation import validate_frame


@dataclass
class LaneBoundary:
    """Represents a fitted lane boundary line (left or right)."""
    slope: float
    intercept: float
    x_bottom: int
    x_top: int
    y_bottom: int
    y_top: int
    is_valid: bool = False

    def get_points(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return endpoints ((x_bottom, y_bottom), (x_top, y_top))."""
        return ((self.x_bottom, self.y_bottom), (self.x_top, self.y_top))


@dataclass
class LaneDetectionResult:
    """Structured output from lane detection pipeline."""
    lane_detected: bool = False
    left_line: Optional[LaneBoundary] = None
    right_line: Optional[LaneBoundary] = None
    lane_center_px: Optional[int] = None
    vehicle_center_px: int = 0
    center_offset_px: float = 0.0
    lane_departure_warning: bool = False
    departure_direction: str = "CENTER"  # "CENTER", "LEFT", "RIGHT"
    lane_polygon: Optional[np.ndarray] = None  # (4, 2) array for corridor fill
    confidence: float = 0.0


class LaneDetector:
    """Robust lane detection using classical OpenCV filtering and line fitting."""

    def __init__(
        self,
        config: Optional[LanesConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config or LanesConfig()
        self.logger = logger or logging.getLogger("road_safety.lanes")

        # Temporal smoothing state across frames
        self.prev_left_line: Optional[LaneBoundary] = None
        self.prev_right_line: Optional[LaneBoundary] = None

    def reset_state(self) -> None:
        """Reset temporal state for a new video stream or clean test."""
        self.prev_left_line = None
        self.prev_right_line = None

    def _get_roi_mask(self, img_shape: Tuple[int, int]) -> np.ndarray:
        """Construct polygonal Region of Interest mask based on normalized vertices."""
        h, w = img_shape[:2]
        vertices = np.array(
            [[int(x * w), int(y * h)] for x, y in self.config.roi_vertices],
            dtype=np.int32,
        )
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [vertices], 255)
        return mask

    def _filter_and_fit_lines(
        self,
        lines: Optional[np.ndarray],
        y_bottom: int,
        y_top: int,
        img_width: int,
    ) -> Tuple[Optional[LaneBoundary], Optional[LaneBoundary]]:
        """Separate line segments into left and right lane candidates and fit lines."""
        if lines is None or len(lines) == 0:
            return None, None

        left_points_x: List[int] = []
        left_points_y: List[int] = []
        left_weights: List[float] = []

        right_points_x: List[int] = []
        right_points_y: List[int] = []
        right_weights: List[float] = []

        img_center_x = img_width / 2.0

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue  # Ignore purely vertical lines to avoid div by zero

                slope = (y2 - y1) / (x2 - x1)
                length = float(np.hypot(x2 - x1, y2 - y1))

                # Discard lines with slopes outside expected range
                if abs(slope) < self.config.min_slope or abs(slope) > self.config.max_slope:
                    continue

                # In image coordinates, y increases downward:
                # Left lane typically has negative slope and is on the left half
                if slope < 0 and max(x1, x2) < img_center_x * 1.15:
                    left_points_x.extend([x1, x2])
                    left_points_y.extend([y1, y2])
                    left_weights.extend([length, length])
                # Right lane typically has positive slope and is on the right half
                elif slope > 0 and min(x1, x2) > img_center_x * 0.85:
                    right_points_x.extend([x1, x2])
                    right_points_y.extend([y1, y2])
                    right_weights.extend([length, length])

        left_boundary = self._fit_boundary(
            left_points_x, left_points_y, left_weights, y_bottom, y_top, is_left=True
        )
        right_boundary = self._fit_boundary(
            right_points_x, right_points_y, right_weights, y_bottom, y_top, is_left=False
        )

        return left_boundary, right_boundary

    def _fit_boundary(
        self,
        x_pts: List[int],
        y_pts: List[int],
        weights: List[float],
        y_bottom: int,
        y_top: int,
        is_left: bool,
    ) -> Optional[LaneBoundary]:
        """Fit a linear polynomial to points with outlier rejection and extrapolate."""
        if len(x_pts) < 2:
            return None

        try:
            # Fit line x as a function of y: x = slope * y + intercept
            # Using x(y) avoids infinite slope singularities for near-vertical road lines
            poly = np.polyfit(y_pts, x_pts, deg=1, w=weights)
            slope = float(poly[0])
            intercept = float(poly[1])

            x_bottom = int(poly[0] * y_bottom + poly[1])
            x_top = int(poly[0] * y_top + poly[1])

            boundary = LaneBoundary(
                slope=slope,
                intercept=intercept,
                x_bottom=x_bottom,
                x_top=x_top,
                y_bottom=y_bottom,
                y_top=y_top,
                is_valid=True,
            )

            # Apply temporal smoothing if previous boundary exists
            prev = self.prev_left_line if is_left else self.prev_right_line
            if prev is not None and prev.is_valid:
                alpha = self.config.temporal_smoothing
                boundary.x_bottom = int(alpha * prev.x_bottom + (1.0 - alpha) * boundary.x_bottom)
                boundary.x_top = int(alpha * prev.x_top + (1.0 - alpha) * boundary.x_top)

            if is_left:
                self.prev_left_line = boundary
            else:
                self.prev_right_line = boundary

            return boundary
        except Exception as e:
            self.logger.debug("Failed line fitting: %s", e)
            return None

    def detect(self, frame: np.ndarray) -> LaneDetectionResult:
        """Execute lane detection on a single frame.

        Args:
            frame: OpenCV BGR image array.

        Returns:
            LaneDetectionResult with detected boundaries, corridor polygon, and alerts.
        """
        is_valid, err = validate_frame(frame)
        if not is_valid:
            self.logger.warning("Invalid frame provided for lane detection: %s", err)
            return LaneDetectionResult()

        if not self.config.enabled:
            return LaneDetectionResult()

        h, w = frame.shape[:2]
        vehicle_center_x = w // 2

        # 1. Grayscale conversion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Gaussian smoothing to suppress high-frequency road texture noise
        k = self.config.gaussian_kernel
        k = k if k % 2 == 1 else k + 1
        blurred = cv2.GaussianBlur(gray, (k, k), 0)

        # 3. Canny edge detection
        edges = cv2.Canny(blurred, self.config.canny_low, self.config.canny_high)

        # 4. Region of interest masking
        roi_mask = self._get_roi_mask((h, w))
        masked_edges = cv2.bitwise_and(edges, roi_mask)

        # 5. Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=self.config.hough_rho,
            theta=np.radians(self.config.hough_theta_deg),
            threshold=self.config.hough_threshold,
            minLineLength=self.config.hough_min_line_len,
            maxLineGap=self.config.hough_max_line_gap,
        )

        y_bottom = int(h * self.config.roi_vertices[0][1])
        y_top = int(h * self.config.roi_vertices[1][1])

        # 6. Slope filtering and line fitting
        left_boundary, right_boundary = self._filter_and_fit_lines(
            lines, y_bottom, y_top, img_width=w
        )

        # Evaluate detection completeness
        lane_detected = (left_boundary is not None and left_boundary.is_valid) and (
            right_boundary is not None and right_boundary.is_valid
        )

        result = LaneDetectionResult(
            lane_detected=lane_detected,
            left_line=left_boundary,
            right_line=right_boundary,
            vehicle_center_px=vehicle_center_x,
        )

        if lane_detected:
            # Construct lane corridor polygon
            # [left_bottom, left_top, right_top, right_bottom]
            poly = np.array(
                [
                    [left_boundary.x_bottom, left_boundary.y_bottom],
                    [left_boundary.x_top, left_boundary.y_top],
                    [right_boundary.x_top, right_boundary.y_top],
                    [right_boundary.x_bottom, right_boundary.y_bottom],
                ],
                dtype=np.int32,
            )
            result.lane_polygon = poly

            # Compute lane center at bottom edge
            lane_center_x = (left_boundary.x_bottom + right_boundary.x_bottom) // 2
            result.lane_center_px = lane_center_x

            # Center offset: positive if vehicle is to the right of lane center (drifting right)
            offset = float(vehicle_center_x - lane_center_x)
            result.center_offset_px = offset
            result.confidence = 0.85

            # Lane Departure Warning (LDW) thresholding
            thresh = self.config.lane_departure_warning_threshold_px
            if abs(offset) > thresh:
                result.lane_departure_warning = True
                result.departure_direction = "RIGHT" if offset > 0 else "LEFT"
            else:
                result.lane_departure_warning = False
                result.departure_direction = "CENTER"

        elif left_boundary is not None or right_boundary is not None:
            # Partial lane detection
            result.confidence = 0.40
            result.lane_departure_warning = False
            result.departure_direction = "UNCERTAIN"
        else:
            # Degraded / missing markings
            result.confidence = 0.0
            result.departure_direction = "NO_LANES"

        return result
