"""
Test suite for Kalman Filter and Multi-Object Tracker.
Problem Statement: SIH 26053 - MI Sense
"""

import pytest
import numpy as np
from backend.app.models.schemas import Detection3D
from backend.app.tracking.kalman_filter import KalmanFilter3D
from backend.app.tracking.tracker import MultiObjectTracker


def test_kalman_filter_prediction_and_update():
    """Verify that Kalman filter accurately tracks linear velocity and position."""
    initial_pos = np.array([0.0, 0.0, 0.0])
    initial_vel = np.array([10.0, 0.0, 0.0])  # 10 m/s in X
    kf = KalmanFilter3D(initial_pos=initial_pos, initial_vel=initial_vel)

    # Predict after dt = 0.1s
    dt = 0.1
    predicted = kf.predict(dt)
    assert abs(predicted[0] - 1.0) < 0.1  # 10 * 0.1 = 1.0m

    # Measurement at [1.02, 0.01, 0.0]
    measurement = np.array([1.02, 0.01, 0.0])
    updated = kf.update(measurement)
    assert abs(updated[0] - 1.02) < 0.05


def test_tracker_id_persistence_and_dynamic_classification():
    """Verify that moving objects preserve Track ID and classify as DYNAMIC."""
    tracker = MultiObjectTracker(gate_distance=3.5, max_age=5, min_hits=2)

    # Frame 1: Vehicle appears at (0, 10, 0)
    det1 = [
        Detection3D(
            id="sim_1",
            class_name="vehicle",
            confidence=0.92,
            position=[0.0, 10.0, 0.5],
            dimensions=[4.5, 2.0, 1.5],
            yaw=1.57,
            velocity=[0.0, 10.0, 0.0],  # 10 m/s forward (+Y)
            detector_source="Simulated DSVT Adapter",
        )
    ]
    tracks1 = tracker.update(det1, dt=0.1)
    assert len(tracks1) == 1
    track_id = tracks1[0].track_id

    # Frame 2: Vehicle moved to (0, 11, 0)
    det2 = [
        Detection3D(
            id="sim_2",
            class_name="vehicle",
            confidence=0.94,
            position=[0.0, 11.0, 0.5],
            dimensions=[4.5, 2.0, 1.5],
            yaw=1.57,
            velocity=[0.0, 10.0, 0.0],
            detector_source="Simulated DSVT Adapter",
        )
    ]
    tracks2 = tracker.update(det2, dt=0.1)
    assert len(tracks2) == 1
    # Check Track ID persistence
    assert tracks2[0].track_id == track_id, "Track ID changed across continuous frames!"

    # Frame 3: Vehicle moved to (0, 12, 0)
    det3 = [
        Detection3D(
            id="sim_3",
            class_name="vehicle",
            confidence=0.95,
            position=[0.0, 12.0, 0.5],
            dimensions=[4.5, 2.0, 1.5],
            yaw=1.57,
            velocity=[0.0, 10.0, 0.0],
            detector_source="Simulated DSVT Adapter",
        )
    ]
    tracks3 = tracker.update(det3, dt=0.1)
    assert len(tracks3) == 1
    assert tracks3[0].track_id == track_id
    assert tracks3[0].dynamic_state == "DYNAMIC", "Moving vehicle was not classified as DYNAMIC"
    assert tracks3[0].speed > 5.0


def test_static_obstacle_classification():
    """Verify that stationary obstacles classify as STATIC."""
    tracker = MultiObjectTracker(gate_distance=3.5, max_age=5, min_hits=2)

    # Stationary barrier at (5, 15, 0)
    for _ in range(4):
        dets = [
            Detection3D(
                id="barrier_1",
                class_name="static_obstacle",
                confidence=0.88,
                position=[5.0, 15.0, 0.3],
                dimensions=[1.0, 1.0, 0.8],
                yaw=0.0,
                velocity=[0.0, 0.0, 0.0],
                detector_source="Simulated DSVT Adapter",
            )
        ]
        tracks = tracker.update(dets, dt=0.1)

    assert len(tracks) == 1
    assert tracks[0].dynamic_state == "STATIC", "Stationary barrier should be STATIC"
    assert tracks[0].speed < 0.5
