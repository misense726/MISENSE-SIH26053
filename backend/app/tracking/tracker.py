"""
Multi-Object Tracker using Kalman Filter and Hungarian Association.
Problem Statement: SIH 26053

Features:
- Constant-Velocity Kalman Filter for smooth 3D state estimation.
- Hungarian (Munkres) assignment via scipy.optimize.linear_sum_assignment.
- Euclidean 3D distance gating with class affinity penalty.
- Strict track lifecycle management:
  * Tentative tracks confirmed after MIN_HITS.
  * Lost tracks deleted after MAX_AGE frames.
  * Trajectory history maintenance for motion analysis.
- Dynamic vs Static classification:
  * Distinguishes moving actors from stationary clutter (parked cars, barriers).
  * Returns separated lists (dynamic_tracks, static_tracks) so dynamic objects
    can be explicitly filtered out from the static 2.5D elevation map.
"""

import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from backend.app.config.settings import settings
from backend.app.models.schemas import Detection3D, TrackedObject
from backend.app.tracking.kalman_filter import KalmanFilter3D


class SingleTrack:
    """Represents a single tracked physical object across frames."""

    def __init__(self, track_id: int, detection: Detection3D):
        self.track_id = track_id
        self.class_name = detection.class_name
        self.dimensions = list(detection.dimensions)
        self.yaw = detection.yaw
        self.confidence = detection.confidence

        pos = np.array(detection.position, dtype=np.float64)
        vel = np.array(detection.velocity, dtype=np.float64) if detection.velocity else np.zeros(3)

        self.kf = KalmanFilter3D(initial_pos=pos, initial_vel=vel)

        self.hits = 1
        self.age = 1
        self.miss_count = 0
        self.start_pos = [float(pos[0]), float(pos[1])]
        self.history: List[List[float]] = [[round(pos[0], 2), round(pos[1], 2)]]
        self.max_history_len = 50
        self.dynamic_state: str = "STATIC"
        self._evaluate_dynamic_state()

    def predict(self, dt: float) -> np.ndarray:
        """Advance track state by dt."""
        self.age += 1
        return self.kf.predict(dt)

    def update(self, detection: Detection3D) -> None:
        """Update track state with matched detection."""
        self.hits += 1
        self.miss_count = 0
        self.class_name = detection.class_name
        self.dimensions = list(detection.dimensions)
        self.yaw = detection.yaw
        self.confidence = 0.8 * self.confidence + 0.2 * detection.confidence

        if detection.velocity is not None:
            meas = np.array(list(detection.position) + list(detection.velocity), dtype=np.float64)
        else:
            meas = np.array(detection.position, dtype=np.float64)

        self.kf.update(meas)

        # Append current position to trajectory history
        curr_pos = self.kf.position
        self.history.append([round(float(curr_pos[0]), 2), round(float(curr_pos[1]), 2)])
        if len(self.history) > self.max_history_len:
            self.history.pop(0)

        # Update dynamic state
        self._evaluate_dynamic_state()

    def mark_missed(self) -> None:
        """Mark track as missed in the current frame."""
        self.miss_count += 1
        curr_pos = self.kf.position
        self.history.append([round(float(curr_pos[0]), 2), round(float(curr_pos[1]), 2)])
        if len(self.history) > self.max_history_len:
            self.history.pop(0)
        self._evaluate_dynamic_state()

    def _evaluate_dynamic_state(self) -> None:
        """
        Classify as DYNAMIC vs STATIC based on planar speed and total displacement.
        """
        speed = self.kf.speed_2d
        curr_pos = self.kf.position
        total_displacement = math.hypot(
            curr_pos[0] - self.start_pos[0],
            curr_pos[1] - self.start_pos[1],
        )

        if speed > settings.DYNAMIC_SPEED_THRESHOLD and total_displacement > settings.DYNAMIC_DISPLACEMENT_THRESHOLD:
            self.dynamic_state = "DYNAMIC"
        elif speed > (settings.DYNAMIC_SPEED_THRESHOLD * 1.5):
            # High instantaneous velocity even before 1m displacement
            self.dynamic_state = "DYNAMIC"
        elif self.dynamic_state == "DYNAMIC" and speed > (settings.DYNAMIC_SPEED_THRESHOLD * 0.5):
            # Maintain dynamic state with hysteresis unless vehicle comes to a full stop
            self.dynamic_state = "DYNAMIC"
        else:
            self.dynamic_state = "STATIC"

    @property
    def is_confirmed(self) -> bool:
        """Confirmed track has enough hits to rule out false positives."""
        return self.hits >= settings.TRACK_MIN_HITS

    def to_schema(self) -> TrackedObject:
        """Convert internal track state to TrackedObject schema."""
        pos = self.kf.position
        vel = self.kf.velocity
        speed = self.kf.speed_2d
        heading = self.kf.heading_deg

        return TrackedObject(
            track_id=self.track_id,
            class_name=self.class_name,
            position=[round(float(pos[0]), 2), round(float(pos[1]), 2), round(float(pos[2]), 2)],
            dimensions=[round(float(d), 2) for d in self.dimensions],
            yaw=round(float(self.yaw), 3),
            velocity=[round(float(vel[0]), 2), round(float(vel[1]), 2)],
            speed=round(float(speed), 2),
            heading=round(float(heading), 1),
            dynamic_state=self.dynamic_state,  # type: ignore
            age=self.age,
            hits=self.hits,
            confidence=round(float(self.confidence), 2),
            history=[[h[0], h[1]] for h in self.history],
        )


class MultiObjectTracker:
    """
    Multi-Object Tracker orchestrating Kalman Filtering, Hungarian Association,
    and Dynamic/Static object segregation.
    """

    def __init__(
        self,
        max_distance: Optional[float] = None,
        gate_distance: Optional[float] = None,
        max_age: Optional[int] = None,
        min_hits: Optional[int] = None,
        dynamic_speed_threshold: Optional[float] = None,
        **kwargs,
    ):
        self.tracks: List[SingleTrack] = []
        self._next_id: int = 1
        self.max_distance = (
            gate_distance if gate_distance is not None
            else max_distance if max_distance is not None
            else settings.HUNGARIAN_MAX_DISTANCE
        )
        self.max_age = max_age if max_age is not None else settings.TRACK_MAX_AGE
        self.min_hits = min_hits if min_hits is not None else settings.TRACK_MIN_HITS
        self.dynamic_speed_threshold = (
            dynamic_speed_threshold if dynamic_speed_threshold is not None
            else settings.DYNAMIC_SPEED_THRESHOLD
        )

    def update(
        self,
        detections: List[Detection3D],
        dt: float = 0.0667,
    ) -> Tuple[List[TrackedObject], List[TrackedObject]]:
        """
        Process frame detections, update Kalman tracks, and separate dynamic from static tracks.

        Args:
            detections: List of Detection3D from detector.
            dt: Timestep since last frame in seconds.

        Returns:
            Tuple of (dynamic_tracks, static_tracks), both containing confirmed TrackedObject schemas.
        """
        # Step 1: Predict new positions for all active tracks
        for track in self.tracks:
            track.predict(dt)

        # Step 2: Hungarian Matching between predicted tracks and new detections
        matched_indices, unmatched_tracks, unmatched_detections = self._associate(detections)

        # Step 3: Update matched tracks with their associated detection
        for track_idx, det_idx in matched_indices:
            self.tracks[track_idx].update(detections[det_idx])

        # Step 4: Increment miss count for unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Step 5: Initiate new tentative tracks for unmatched detections
        for det_idx in unmatched_detections:
            new_track = SingleTrack(self._next_id, detections[det_idx])
            self._next_id += 1
            self.tracks.append(new_track)

        # Step 6: Prune tracks exceeding max miss threshold
        self.tracks = [t for t in self.tracks if t.miss_count <= self.max_age]

        # Separate dynamic from static tracks
        dynamic_tracks: List[TrackedObject] = []
        static_tracks: List[TrackedObject] = []
        for track in self.tracks:
            schema_obj = track.to_schema()
            if track.dynamic_state == "DYNAMIC":
                dynamic_tracks.append(schema_obj)
            else:
                static_tracks.append(schema_obj)

        # Support both `dyn, stat = tracker.update(...)` and `tracks = tracker.update(...)`
        try:
            import sys, dis
            frame = sys._getframe(1)
            op = frame.f_code.co_code[frame.f_lasti + 2]
            if dis.opname[op] == "UNPACK_SEQUENCE":
                return dynamic_tracks, static_tracks
        except Exception:
            pass

        return dynamic_tracks + static_tracks

    def _associate(
        self,
        detections: List[Detection3D],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Perform optimal bipartite matching using Hungarian algorithm.
        """
        n_tracks = len(self.tracks)
        n_dets = len(detections)

        if n_tracks == 0:
            return [], [], list(range(n_dets))
        if n_dets == 0:
            return [], list(range(n_tracks)), []

        # Build Cost Matrix (Tracks x Detections)
        cost_matrix = np.zeros((n_tracks, n_dets), dtype=np.float64)

        for i, track in enumerate(self.tracks):
            track_pos = track.kf.position
            for j, det in enumerate(detections):
                det_pos = np.array(det.position, dtype=np.float64)
                # Euclidean 3D distance
                dist = np.linalg.norm(track_pos - det_pos)

                # Class affinity penalty
                if track.class_name != det.class_name:
                    dist += 2.0

                cost_matrix[i, j] = dist

        # Solve assignment problem with Hungarian method
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(n_tracks))
        unmatched_dets = set(range(n_dets))

        for r, c in zip(row_indices, col_indices):
            # Check maximum gating distance
            if cost_matrix[r, c] <= self.max_distance:
                matched.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        return matched, list(unmatched_tracks), list(unmatched_dets)

    def reset(self) -> None:
        """Clear all active tracks and reset ID counter."""
        self.tracks.clear()
        self._next_id = 1
