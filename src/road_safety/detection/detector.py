"""YOLO-based object detector for road safety scenes."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from ..config import DetectionConfig, SystemConfig
from ..utils.validation import validate_frame


# Standard COCO class names relevant to road scenes
COCO_ROAD_CLASSES: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
    11: "stop sign",
}


@dataclass
class DetectionResult:
    """Represents a single detected object in a frame."""
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None
    bottom_center: Tuple[int, int] = (0, 0)
    area: float = 0.0
    depth_proxy: float = 0.0  # Normalized y2 / height (1.0 = bottom edge, closest)

    @property
    def is_vulnerable_user(self) -> bool:
        """Indicate if the detected object is a pedestrian or cyclist."""
        return self.class_id in (0, 1)

    @property
    def is_vehicle(self) -> bool:
        """Indicate if the detected object is a motor vehicle."""
        return self.class_id in (2, 3, 5, 7)

    @property
    def is_traffic_control(self) -> bool:
        """Indicate if the detected object is a traffic light or stop sign."""
        return self.class_id in (9, 11)


class ObjectDetector:
    """Wrapper around Ultralytics YOLO for road scene object detection."""

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        system_config: Optional[SystemConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config or DetectionConfig()
        self.system_config = system_config or SystemConfig()
        self.logger = logger or logging.getLogger("road_safety.detector")
        self.device = self.system_config.resolved_device()
        self.model: Optional[Any] = None
        self.is_loaded = False

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Load the YOLO model from local path or standard download with error handling."""
        if not ULTRALYTICS_AVAILABLE:
            self.logger.error("Ultralytics package is not installed. YOLO detection disabled.")
            return

        model_identifier = self.config.model_name
        # Check if local model file exists at model_path
        if Path(self.config.model_path).is_file():
            model_identifier = self.config.model_path
        elif Path(self.config.model_name).is_file():
            model_identifier = self.config.model_name

        try:
            self.logger.info(
                "Initializing YOLO detector with '%s' on device '%s'...",
                model_identifier,
                self.device,
            )
            self.model = YOLO(model_identifier)
            self.is_loaded = True
            self.logger.info("YOLO model successfully loaded.")
        except Exception as e:
            self.logger.warning(
                "Failed to initialize YOLO model '%s': %s. "
                "Detector will operate in fallback mock mode.",
                model_identifier,
                e,
            )
            self.model = None
            self.is_loaded = False

    def detect(self, frame: np.ndarray) -> Tuple[List[DetectionResult], float]:
        """Run object detection on a single OpenCV BGR image frame.

        Args:
            frame: OpenCV BGR image array.

        Returns:
            Tuple of (list of DetectionResult, inference_time_ms).
        """
        is_valid, err = validate_frame(frame)
        if not is_valid:
            self.logger.warning("Invalid frame provided to detector: %s", err)
            return [], 0.0

        if not self.is_loaded or self.model is None:
            # Fallback when model is not available (e.g. offline unit test without weights)
            return [], 0.0

        h, w = frame.shape[:2]
        t0 = time.perf_counter()

        try:
            results = self.model.predict(
                source=frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.img_size,
                classes=self.config.target_classes,
                device=self.device,
                verbose=False,
            )
            inference_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            self.logger.error("Inference execution failed: %s", e)
            return [], 0.0

        detections: List[DetectionResult] = []
        if not results:
            return detections, inference_ms

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections, inference_ms

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)

        names = r.names if hasattr(r, "names") else COCO_ROAD_CLASSES

        for box, score, cid in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            # Clip coordinates to frame boundary
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            cname = names.get(cid, COCO_ROAD_CLASSES.get(cid, f"class_{cid}"))
            bottom_center = ((x1 + x2) // 2, y2)
            area = float((x2 - x1) * (y2 - y1))
            depth_proxy = float(y2 / h) if h > 0 else 0.0

            detections.append(
                DetectionResult(
                    box=(x1, y1, x2, y2),
                    class_id=cid,
                    class_name=cname,
                    confidence=float(score),
                    bottom_center=bottom_center,
                    area=area,
                    depth_proxy=depth_proxy,
                )
            )

        return detections, inference_ms
