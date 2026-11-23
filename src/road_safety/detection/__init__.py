"""Object detection and tracking modules for road safety."""

from .detector import DetectionResult, ObjectDetector
from .tracker import SimpleCentroidTracker

__all__ = ["DetectionResult", "ObjectDetector", "SimpleCentroidTracker"]
