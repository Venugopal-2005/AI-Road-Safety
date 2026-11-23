"""Unified road safety vision processing pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from ..config import AppConfig, get_default_config
from ..detection.detector import DetectionResult, ObjectDetector
from ..detection.tracker import SimpleCentroidTracker
from ..lanes.lane_detector import LaneDetectionResult, LaneDetector
from ..logging_config import setup_logger
from ..safety.risk_analyzer import RiskAnalyzer, RiskAssessment
from ..signs.sign_detector import TrafficSignDetector, TrafficSignResult
from ..utils.validation import validate_frame, validate_image_file
from ..utils.visualization import Visualizer


@dataclass
class ProcessedFrame:
    """Consolidated results of a single analyzed frame."""
    annotated_frame: np.ndarray
    original_frame: np.ndarray
    detections: List[DetectionResult] = field(default_factory=list)
    lane_result: Optional[LaneDetectionResult] = None
    signs: List[TrafficSignResult] = field(default_factory=list)
    risk: Optional[RiskAssessment] = None
    latency_ms: float = 0.0
    fps: float = 0.0
    frame_index: int = 0


class RoadSafetyPipeline:
    """Coordinates object detection, tracking, lane detection, sign recognition,

    risk analysis, and visualization in an end-to-end computer vision pipeline.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config or get_default_config()
        self.logger = logger or setup_logger(
            name="road_safety.pipeline",
            log_level=self.config.system.log_level,
            log_file=self.config.system.log_file,
        )

        # Initialize vision components
        self.detector = ObjectDetector(
            config=self.config.detection,
            system_config=self.config.system,
            logger=self.logger,
        )
        self.tracker = (
            SimpleCentroidTracker(max_disappeared=self.config.detection.tracker_max_age)
            if self.config.detection.enable_tracking
            else None
        )
        self.lane_detector = LaneDetector(
            config=self.config.lanes,
            logger=self.logger,
        )
        self.sign_detector = TrafficSignDetector(
            config=self.config.signs,
            logger=self.logger,
        )
        self.risk_analyzer = RiskAnalyzer(
            config=self.config.safety,
            logger=self.logger,
        )
        self.visualizer = Visualizer(
            config=self.config.visualization,
        )

        # Performance tracking state
        self._frame_count: int = 0
        self._fps_ema: float = 0.0
        self._last_timestamp: Optional[float] = None
        self._device_label = "CUDA" if self.config.system.resolved_device() == "cuda" else "CPU"

    def reset(self) -> None:
        """Reset temporal state across stream changes."""
        self._frame_count = 0
        self._fps_ema = 0.0
        self._last_timestamp = None
        if self.lane_detector:
            self.lane_detector.reset_state()
        if self.tracker:
            self.tracker = SimpleCentroidTracker(
                max_disappeared=self.config.detection.tracker_max_age
            )

    def process_frame(self, frame: np.ndarray) -> ProcessedFrame:
        """Process a single video frame or image array through the vision pipeline.

        Args:
            frame: OpenCV BGR image array.

        Returns:
            ProcessedFrame containing annotated image and telemetry results.
        """
        t_start = time.perf_counter()

        is_valid, err_msg = validate_frame(frame)
        if not is_valid:
            self.logger.warning("Frame validation rejected: %s", err_msg)
            # Return blank or copy if invalid
            safe_frame = frame if frame is not None and isinstance(frame, np.ndarray) else np.zeros((480, 640, 3), dtype=np.uint8)
            return ProcessedFrame(
                annotated_frame=safe_frame,
                original_frame=safe_frame,
            )

        # 1. Object Detection
        detections, det_time_ms = self.detector.detect(frame)

        # 2. Object Tracking
        if self.tracker is not None:
            try:
                detections = self.tracker.update(detections)
            except Exception as e:
                self.logger.warning("Tracking failed on frame %d: %s", self._frame_count, e)

        # 3. Lane Detection
        try:
            lane_result = self.lane_detector.detect(frame)
        except Exception as e:
            self.logger.warning("Lane detection error on frame %d: %s", self._frame_count, e)
            lane_result = LaneDetectionResult()

        # 4. Traffic Sign Detection
        try:
            signs = self.sign_detector.process_from_detections(frame, detections)
        except Exception as e:
            self.logger.warning("Sign detection error on frame %d: %s", self._frame_count, e)
            signs = []

        # 5. Safety Risk Analysis
        try:
            risk = self.risk_analyzer.analyze(
                frame_shape=frame.shape[:2],
                detections=detections,
                lane_result=lane_result,
                signs=signs,
            )
        except Exception as e:
            self.logger.warning("Risk analyzer error on frame %d: %s", self._frame_count, e)
            risk = RiskAssessment()

        # Compute pipeline timing
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        # Compute smoothed FPS
        if self._last_timestamp is not None:
            dt = t_end - self._last_timestamp
            if dt > 0:
                current_fps = 1.0 / dt
                if self._fps_ema == 0.0:
                    self._fps_ema = current_fps
                else:
                    self._fps_ema = 0.85 * self._fps_ema + 0.15 * current_fps
        self._last_timestamp = t_end
        self._frame_count += 1

        # 6. Visualization & Telemetry HUD
        annotated = self.visualizer.draw_scene(
            frame=frame,
            detections=detections,
            lane_result=lane_result,
            signs=signs,
            risk=risk,
            fps=self._fps_ema,
            latency_ms=latency_ms,
            device_label=self._device_label,
        )

        return ProcessedFrame(
            annotated_frame=annotated,
            original_frame=frame,
            detections=detections,
            lane_result=lane_result,
            signs=signs,
            risk=risk,
            latency_ms=latency_ms,
            fps=self._fps_ema,
            frame_index=self._frame_count,
        )

    def process_image(self, image_input: Union[str, Path, np.ndarray]) -> ProcessedFrame:
        """Process a static image from path or numpy array.

        Args:
            image_input: File path or existing BGR image array.

        Returns:
            ProcessedFrame output.
        """
        if isinstance(image_input, (str, Path)):
            is_valid, err = validate_image_file(image_input)
            if not is_valid:
                raise ValueError(f"Invalid image file: {err}")
            frame = cv2.imread(str(image_input))
        else:
            frame = image_input

        # Reset smoothing for static image analysis
        self.reset()
        return self.process_frame(frame)
