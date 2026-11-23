"""Unit tests for frame and file validation utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.road_safety.utils.validation import (
    validate_frame,
    validate_image_file,
    validate_video_file,
)


def test_validate_frame_valid():
    """Verify standard BGR and grayscale frames pass validation."""
    bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    ok, err = validate_frame(bgr)
    assert ok is True
    assert err == ""

    gray = np.zeros((480, 640), dtype=np.uint8)
    ok_g, err_g = validate_frame(gray)
    assert ok_g is True
    assert err_g == ""


def test_validate_frame_invalid_inputs():
    """Verify invalid frames are properly rejected."""
    # None
    ok, err = validate_frame(None)
    assert ok is False
    assert "None" in err

    # Wrong type
    ok, err = validate_frame("not an array")  # type: ignore
    assert ok is False
    assert "numpy.ndarray" in err

    # Empty array
    ok, err = validate_frame(np.array([], dtype=np.uint8))
    assert ok is False
    assert "size 0" in err

    # Unsupported dimensions (1D or 4D)
    ok, err = validate_frame(np.zeros((100,), dtype=np.uint8))
    assert ok is False
    assert "Unsupported dimensions" in err

    # Dimensions too small (< 10x10)
    ok, err = validate_frame(np.zeros((5, 5, 3), dtype=np.uint8))
    assert ok is False
    assert "too small" in err


def test_validate_image_file(tmp_path: Path):
    """Test image file integrity and format checks."""
    # Nonexistent file
    ok, err = validate_image_file(tmp_path / "missing.jpg")
    assert ok is False
    assert "not found" in err

    # Unsupported extension
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("hello", encoding="utf-8")
    ok, err = validate_image_file(txt_file)
    assert ok is False
    assert "Unsupported image extension" in err

    # Empty file (0 bytes)
    empty_img = tmp_path / "empty.jpg"
    empty_img.touch()
    ok, err = validate_image_file(empty_img)
    assert ok is False
    assert "empty" in err

    # Valid image file
    valid_img_path = tmp_path / "valid.png"
    sample = np.full((100, 100, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(valid_img_path), sample)
    ok, err = validate_image_file(valid_img_path)
    assert ok is True
    assert err == ""


def test_validate_video_file(tmp_path: Path):
    """Test video file format and existence checks."""
    # Nonexistent
    ok, err = validate_video_file(tmp_path / "missing.mp4")
    assert ok is False
    assert "not found" in err

    # Unsupported extension
    bad_ext = tmp_path / "sample.pdf"
    bad_ext.write_text("fake pdf", encoding="utf-8")
    ok, err = validate_video_file(bad_ext)
    assert ok is False
    assert "Unsupported video extension" in err

    # Empty file
    empty_video = tmp_path / "empty.mp4"
    empty_video.touch()
    ok, err = validate_video_file(empty_video)
    assert ok is False
    assert "empty" in err
