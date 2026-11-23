"""Lightweight multi-object tracker based on centroid distance and bounding box IoU."""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detector import DetectionResult


@dataclass
class Track:
    """Represents a tracked object across video frames."""
    track_id: int
    class_id: int
    class_name: str
    box: Tuple[int, int, int, int]
    centroid: Tuple[int, int]
    history: collections.deque = field(default_factory=lambda: collections.deque(maxlen=30))
    area_history: collections.deque = field(default_factory=lambda: collections.deque(maxlen=15))
    disappeared_count: int = 0
    hits: int = 1

    @property
    def growth_rate(self) -> float:
        """Estimate bounding box area expansion rate (proxy for approaching speed).

        A value > 1.0 indicates an approaching object growing larger in view.
        """
        if len(self.area_history) < 3:
            return 1.0
        earlier = np.mean(list(self.area_history)[:2])
        recent = np.mean(list(self.area_history)[-2:])
        if earlier <= 0:
            return 1.0
        return float(recent / earlier)

    @property
    def velocity_vector(self) -> Tuple[float, float]:
        """Estimated pixel velocity (dx, dy) per frame over recent observations."""
        if len(self.history) < 2:
            return (0.0, 0.0)
        p1 = self.history[0]
        p2 = self.history[-1]
        dt = max(1, len(self.history) - 1)
        return ((p2[0] - p1[0]) / dt, (p2[1] - p1[1]) / dt)


class SimpleCentroidTracker:
    """Multi-object tracker associating detections using centroid Euclidean distance."""

    def __init__(self, max_disappeared: int = 15, max_distance_px: float = 120.0):
        self.next_track_id = 1
        self.tracks: Dict[int, Track] = {}
        self.max_disappeared = max_disappeared
        self.max_distance_px = max_distance_px

    def _register(self, detection: DetectionResult) -> int:
        """Register a new tracked object."""
        tid = self.next_track_id
        self.next_track_id += 1
        x1, y1, x2, y2 = detection.box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        area = detection.area

        track = Track(
            track_id=tid,
            class_id=detection.class_id,
            class_name=detection.class_name,
            box=detection.box,
            centroid=(cx, cy),
        )
        track.history.append((cx, cy))
        track.area_history.append(area)

        self.tracks[tid] = track
        return tid

    def _deregister(self, track_id: int) -> None:
        """Remove a track that has expired."""
        if track_id in self.tracks:
            del self.tracks[track_id]

    def update(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Update tracks with new detections from the current frame.

        Args:
            detections: List of DetectionResult from the current frame.

        Returns:
            Updated detections with `track_id` assigned where matched.
        """
        # If no detections exist in this frame
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id].disappeared_count += 1
                if self.tracks[track_id].disappeared_count > self.max_disappeared:
                    self._deregister(track_id)
            return []

        # If no tracks are currently active, register all detections
        if len(self.tracks) == 0:
            for det in detections:
                tid = self._register(det)
                det.track_id = tid
            return detections

        # Compute centroids of incoming detections
        input_centroids = np.zeros((len(detections), 2), dtype=int)
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.box
            input_centroids[i] = ((x1 + x2) // 2, (y1 + y2) // 2)

        track_ids = list(self.tracks.keys())
        track_centroids = np.array([self.tracks[tid].centroid for tid in track_ids])

        # Compute distance matrix between active tracks and new detections
        # shape: (len(track_ids), len(detections))
        dist_matrix = np.linalg.norm(
            track_centroids[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2
        )

        rows = dist_matrix.min(axis=1).argsort()
        cols = dist_matrix.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            # Check distance threshold
            if dist_matrix[row, col] > self.max_distance_px:
                continue

            tid = track_ids[row]
            det = detections[col]

            # Update track
            track = self.tracks[tid]
            cx, cy = int(input_centroids[col][0]), int(input_centroids[col][1])
            track.centroid = (cx, cy)
            track.box = det.box
            track.history.append((cx, cy))
            track.area_history.append(det.area)
            track.disappeared_count = 0
            track.hits += 1

            det.track_id = tid
            used_rows.add(row)
            used_cols.add(col)

        # Unused tracks increment disappeared count
        unused_rows = set(range(dist_matrix.shape[0])) - used_rows
        for row in unused_rows:
            tid = track_ids[row]
            self.tracks[tid].disappeared_count += 1
            if self.tracks[tid].disappeared_count > self.max_disappeared:
                self._deregister(tid)

        # Unused detections are registered as new tracks
        unused_cols = set(range(dist_matrix.shape[1])) - used_cols
        for col in unused_cols:
            det = detections[col]
            tid = self._register(det)
            det.track_id = tid

        return detections

    def get_track(self, track_id: int) -> Optional[Track]:
        """Retrieve a specific track by its ID."""
        return self.tracks.get(track_id)
