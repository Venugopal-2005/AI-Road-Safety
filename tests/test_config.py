"""Unit tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.road_safety.config import (
    AppConfig,
    DetectionConfig,
    LanesConfig,
    SafetyConfig,
    SignsConfig,
    SystemConfig,
    VisualizationConfig,
    get_default_config,
)


def test_default_config_instantiation():
    """Verify default configuration initializes with sensible defaults."""
    cfg = AppConfig()

    assert cfg.system.device in ("auto", "cpu", "cuda")
    assert cfg.detection.model_name == "yolov8n.pt"
    assert 0.0 < cfg.detection.confidence_threshold < 1.0
    assert cfg.lanes.enabled is True
    assert cfg.lanes.canny_low < cfg.lanes.canny_high
    assert len(cfg.lanes.roi_vertices) == 4
    assert cfg.safety.pedestrian_critical_distance_ratio > cfg.safety.pedestrian_caution_distance_ratio
    assert cfg.visualization.show_telemetry_hud is True


def test_resolved_device():
    """Verify resolved_device falls back to CPU if CUDA is not available."""
    sys_cfg = SystemConfig(device="cpu")
    assert sys_cfg.resolved_device() == "cpu"

    sys_cfg_auto = SystemConfig(device="auto")
    assert sys_cfg_auto.resolved_device() in ("cpu", "cuda")


def test_yaml_loading(tmp_path: Path):
    """Test loading configuration from a temporary YAML file."""
    custom_yaml = tmp_path / "custom_config.yaml"
    data = {
        "system": {"device": "cpu", "log_level": "DEBUG"},
        "detection": {"confidence_threshold": 0.55, "img_size": 480},
        "lanes": {"canny_low": 40, "canny_high": 120},
        "safety": {"ego_corridor_width_ratio": 0.40},
    }
    with open(custom_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)

    cfg = AppConfig.from_yaml(custom_yaml)
    assert cfg.system.device == "cpu"
    assert cfg.system.log_level == "DEBUG"
    assert cfg.detection.confidence_threshold == 0.55
    assert cfg.detection.img_size == 480
    assert cfg.lanes.canny_low == 40
    assert cfg.safety.ego_corridor_width_ratio == 0.40


def test_yaml_missing_file_fallback():
    """Verify that a nonexistent file returns default configuration without crashing."""
    cfg = AppConfig.from_yaml("nonexistent_file_path_12345.yaml")
    assert cfg.detection.confidence_threshold == 0.35
    assert cfg.system.save_outputs is True
