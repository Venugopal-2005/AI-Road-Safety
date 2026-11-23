"""Configuration loader and schema validator for the Road Safety System."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import yaml


@dataclass
class SystemConfig:
    device: str = "auto"
    num_threads: int = 4
    log_level: str = "INFO"
    log_file: str = "outputs/road_safety.log"
    save_outputs: bool = True
    output_dir: str = "outputs"

    def resolved_device(self) -> str:
        """Resolve 'auto' to 'cuda' if GPU is available and verified, else 'cpu'."""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self.device


@dataclass
class DetectionConfig:
    model_name: str = "yolov8n.pt"
    model_path: str = "models/yolov8n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    img_size: int = 640
    target_classes: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 5, 7, 9, 11]
    )
    enable_tracking: bool = True
    tracker_max_age: int = 15
    tracker_min_hits: int = 3


@dataclass
class LanesConfig:
    enabled: bool = True
    canny_low: int = 50
    canny_high: int = 150
    gaussian_kernel: int = 5
    hough_rho: int = 2
    hough_theta_deg: float = 1.0
    hough_threshold: int = 40
    hough_min_line_len: int = 40
    hough_max_line_gap: int = 50
    roi_vertices: List[List[float]] = field(
        default_factory=lambda: [
            [0.05, 0.95],
            [0.42, 0.60],
            [0.58, 0.60],
            [0.95, 0.95],
        ]
    )
    min_slope: float = 0.35
    max_slope: float = 2.50
    temporal_smoothing: float = 0.70
    lane_departure_warning_threshold_px: int = 50


@dataclass
class SignsConfig:
    enabled: bool = True
    yolo_sign_classes: List[int] = field(default_factory=lambda: [9, 11])
    enable_heuristic_signs: bool = True
    min_contour_area: int = 400
    max_contour_area: int = 40000


@dataclass
class SafetyConfig:
    ego_corridor_width_ratio: float = 0.36
    pedestrian_critical_distance_ratio: float = 0.70
    pedestrian_caution_distance_ratio: float = 0.50
    vehicle_critical_distance_ratio: float = 0.78
    vehicle_caution_distance_ratio: float = 0.58
    enable_ttc_proxy: bool = True
    ttc_critical_frames: int = 10
    ttc_caution_frames: int = 25


@dataclass
class VisualizationConfig:
    show_bounding_boxes: bool = True
    show_labels: bool = True
    show_confidence: bool = True
    show_lane_overlay: bool = True
    show_ego_corridor: bool = True
    show_telemetry_hud: bool = True
    show_safety_alerts: bool = True
    box_line_thickness: int = 2
    font_scale: float = 0.6
    overlay_alpha: float = 0.35


@dataclass
class AppConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    lanes: LanesConfig = field(default_factory=LanesConfig)
    signs: SignsConfig = field(default_factory=SignsConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> AppConfig:
        """Load configuration from a YAML file, merging with default values."""
        path = Path(yaml_path)
        if not path.is_file():
            # Return defaults if file does not exist
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            system=SystemConfig(**data.get("system", {})),
            detection=DetectionConfig(**data.get("detection", {})),
            lanes=LanesConfig(**data.get("lanes", {})),
            signs=SignsConfig(**data.get("signs", {})),
            safety=SafetyConfig(**data.get("safety", {})),
            visualization=VisualizationConfig(**data.get("visualization", {})),
        )


def get_default_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """Convenience helper to retrieve initialized AppConfig."""
    if config_path:
        return AppConfig.from_yaml(config_path)

    # Attempt to locate standard config.yaml relative to current module or repo root
    candidates = [
        Path("configs/config.yaml"),
        Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml",
        Path("AI-Road-Safety/configs/config.yaml"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return AppConfig.from_yaml(candidate)

    return AppConfig()
