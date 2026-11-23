"""Input validation routines for images, video frames, and media files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class ValidationError(Exception):
    """Raised when an input frame or file fails integrity checks."""
    pass


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def validate_frame(frame: Optional[np.ndarray]) -> Tuple[bool, str]:
    """Validate that a frame is a valid non-empty OpenCV BGR image array.

    Args:
        frame: Candidate numpy array.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if frame is None:
        return False, "Frame is None"

    if not isinstance(frame, np.ndarray):
        return False, f"Expected numpy.ndarray, got {type(frame).__name__}"

    if frame.size == 0:
        return False, "Frame has size 0 (empty array)"

    if frame.ndim not in (2, 3):
        return False, f"Unsupported dimensions: ndim={frame.ndim}. Expected 2 or 3"

    h, w = frame.shape[:2]
    if h < 10 or w < 10:
        return False, f"Frame dimensions ({w}x{h}) are too small for processing (minimum 10x10)"

    return True, ""


def validate_image_file(file_path: str | Path) -> Tuple[bool, str]:
    """Validate that an image file exists, has a valid extension, and can be decoded.

    Args:
        file_path: Path to candidate image.

    Returns:
        Tuple of (is_valid, error_message).
    """
    path = Path(file_path)
    if not path.is_file():
        return False, f"Image file not found: {file_path}"

    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return False, (
            f"Unsupported image extension '{path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    if path.stat().st_size == 0:
        return False, f"Image file is empty (0 bytes): {file_path}"

    # Verify OpenCV can decode the image
    img = cv2.imread(str(path))
    if img is None:
        return False, f"OpenCV failed to decode image: {file_path} (corrupt or unreadable format)"

    return True, ""


def validate_video_file(file_path: str | Path) -> Tuple[bool, str]:
    """Validate that a video file exists, is supported, and has readable frames.

    Args:
        file_path: Path to candidate video.

    Returns:
        Tuple of (is_valid, error_message).
    """
    path = Path(file_path)
    if not path.is_file():
        return False, f"Video file not found: {file_path}"

    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        return False, (
            f"Unsupported video extension '{path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_VIDEO_EXTENSIONS)}"
        )

    if path.stat().st_size == 0:
        return False, f"Video file is empty (0 bytes): {file_path}"

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, f"OpenCV VideoCapture failed to open: {file_path}"

    ret, test_frame = cap.read()
    cap.release()

    if not ret or test_frame is None:
        return False, f"Video container opened but contains no readable frames: {file_path}"

    return True, ""
