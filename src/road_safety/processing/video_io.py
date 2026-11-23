"""Video reading, writing, and batch processing utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, Optional, Tuple

import cv2
import numpy as np

from ..logging_config import setup_logger
from ..utils.validation import validate_video_file
from .pipeline import ProcessedFrame, RoadSafetyPipeline


@dataclass
class VideoMetadata:
    """Metadata container for video files."""
    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float


class VideoReader:
    """Safely opens and iterates frames from a video source."""

    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
        is_valid, err = validate_video_file(self.video_path)
        if not is_valid:
            raise ValueError(f"Invalid video file: {err}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.video_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0.0

    @property
    def metadata(self) -> VideoMetadata:
        return VideoMetadata(
            width=self.width,
            height=self.height,
            fps=self.fps,
            total_frames=self.total_frames,
            duration_sec=self.duration_sec,
        )

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Yield frames sequentially."""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break
                yield frame
        finally:
            self.release()

    def release(self) -> None:
        """Release underlying OpenCV VideoCapture."""
        if self.cap and self.cap.isOpened():
            self.cap.release()


class VideoWriter:
    """Writes video frames to an MP4 file with codec fallback for browser playback."""

    def __init__(
        self,
        output_path: str | Path,
        fps: float = 30.0,
        frame_size: Tuple[int, int] = (1280, 720),
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frame_size = frame_size

        # Try H.264 / avc1 first, fallback to mp4v
        codecs_to_try = ["avc1", "mp4v", "MJPG"]
        self.writer: Optional[cv2.VideoWriter] = None

        for codec in codecs_to_try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                self.frame_size,
            )
            if writer.isOpened():
                self.writer = writer
                break

        if self.writer is None or not self.writer.isOpened():
            # Final fallback
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                self.frame_size,
            )

    def write(self, frame: np.ndarray) -> None:
        """Write a single BGR frame."""
        if self.writer is None:
            raise RuntimeError("VideoWriter is not initialized")

        # Resize if dimensions do not match target frame size
        h, w = frame.shape[:2]
        if (w, h) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)

        self.writer.write(frame)

    def release(self) -> None:
        """Close and flush video file."""
        if self.writer and self.writer.isOpened():
            self.writer.release()


class VideoProcessor:
    """Processes an entire video through the RoadSafetyPipeline with progress reporting."""

    def __init__(
        self,
        pipeline: RoadSafetyPipeline,
        logger: Optional[logging.Logger] = None,
    ):
        self.pipeline = pipeline
        self.logger = logger or setup_logger(name="road_safety.video_processor")

    def process_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
        frame_stride: int = 1,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> VideoMetadata:
        """Run pipeline on every (or strided) frame and save annotated video.

        Args:
            input_path: Source video file path.
            output_path: Target annotated video destination.
            frame_stride: Process every Nth frame (useful for CPU speedup).
            progress_callback: Optional callback func(current_frame, total_frames, elapsed_sec).

        Returns:
            VideoMetadata for the processed video.
        """
        reader = VideoReader(input_path)
        meta = reader.metadata

        # Output video will have effective fps = original_fps / frame_stride if skipping
        effective_fps = max(1.0, meta.fps / float(frame_stride))
        writer = VideoWriter(
            output_path=output_path,
            fps=effective_fps,
            frame_size=(meta.width, meta.height),
        )

        self.pipeline.reset()
        current_idx = 0
        processed_count = 0

        self.logger.info(
            "Starting video processing: %s -> %s (%d frames, %.1f fps, stride=%d)",
            input_path,
            output_path,
            meta.total_frames,
            meta.fps,
            frame_stride,
        )

        try:
            for frame in reader.frames():
                current_idx += 1
                if current_idx % frame_stride != 0:
                    continue

                processed = self.pipeline.process_frame(frame)
                writer.write(processed.annotated_frame)
                processed_count += 1

                if progress_callback is not None:
                    progress_callback(current_idx, meta.total_frames, processed.latency_ms)

        finally:
            reader.release()
            writer.release()

        self.logger.info("Video processing finished. Wrote %d frames.", processed_count)
        return meta
