"""Unit tests verifying Multi-Object Person Tracking for session-scoped anonymous tracking.

Covers:
- Consistent track ID across sequential frames
- Handling crossing paths without ID flip
- Brief occlusion and track recovery within grace period
- Track retirement and status transitions (active -> lost -> ended)
- ByteTrack low-confidence matching during partial occlusions
- Clean session epoch resets
- 20-30 simultaneous active tracks with low latency
- Interoperability with PersonDetection contracts and raw boxes
- Strict anonymity
"""
import time
from typing import List, Tuple
import numpy as np
import pytest

from services.face_contracts import PersonDetection, PersonTrack
from services.person_tracker import PersonTracker, compute_diou, compute_iou


def test_consistent_track_id_across_sequential_frames():
    """Verify that an individual walking across sequential frames maintains a stable ID."""
    tracker = PersonTracker(session_id="session_test_1", lifetime=2.5, confirm_hits=2)

    # Simulate 30 frames at 30 fps (dt = 0.0333s)
    fps = 30.0
    dt = 1.0 / fps
    start_x = 50.0
    speed_px_per_frame = 4.0  # ~120 px/s

    tracked_ids = []
    for i in range(30):
        t = i * dt
        x1 = start_x + i * speed_px_per_frame
        y1 = 80.0
        x2 = x1 + 50.0
        y2 = y1 + 120.0

        det = PersonDetection(bbox=(x1, y1, x2, y2), confidence=0.92)
        active = tracker.update([det], timestamp=t)

        assert len(active) == 1
        track = active[0]

        # Verify contract type
        assert isinstance(track, PersonTrack)
        tracked_ids.append(track.track_id)

        # Basic contract assertions
        assert track.session_id == "session_test_1"
        assert track.state == "active"
        assert track.first_seen == pytest.approx(0.0, abs=1e-4)
        assert track.last_seen == pytest.approx(t, abs=1e-4)
        assert track.duration == pytest.approx(t, abs=1e-4)
        assert track.hits == i + 1

        # Check confirmation status
        if i >= 1:
            assert track.confirmed is True

    # ID must be strictly identical across all 30 frames
    assert len(set(tracked_ids)) == 1
    assert tracked_ids[0] == 1

    # Check trajectory length and speed
    final_track = tracker.get_track(1)
    assert final_track is not None
    assert len(final_track.trajectory) == 30
    assert final_track.speed_px_s > 50.0  # Should detect motion velocity


def test_handling_crossing_paths_without_id_flip():
    """Verify two people crossing trajectories in opposite directions do not switch IDs."""
    tracker = PersonTracker(session_id="session_cross", lifetime=2.5, confirm_hits=2)

    # Person 1 starts at x=100 and moves right at +10 px/frame
    # Person 2 starts at x=500 and moves left at -10 px/frame
    # Crossing occurs at frame 20 (x=300)
    total_frames = 41
    dt = 0.1  # 10 fps

    person1_coords = []
    person2_coords = []

    for i in range(total_frames):
        t = i * dt
        # Person 1: moving right
        p1_x1 = 100.0 + i * 10.0
        p1_box = (p1_x1, 150.0, p1_x1 + 45.0, 270.0)

        # Person 2: moving left
        p2_x1 = 500.0 - i * 10.0
        p2_box = (p2_x1, 150.0, p2_x1 + 45.0, 270.0)

        dets = [
            PersonDetection(bbox=p1_box, confidence=0.88),
            PersonDetection(bbox=p2_box, confidence=0.89),
        ]

        active = tracker.update(dets, timestamp=t)
        assert len(active) == 2

        # Sort by x coordinate to map to left/right position
        sorted_tracks = sorted(active, key=lambda trk: trk.bbox[0])

        if i == 0:
            p1_id = sorted_tracks[0].track_id
            p2_id = sorted_tracks[1].track_id
            assert p1_id != p2_id
        else:
            # Locate tracks by their initial IDs
            t1 = tracker.get_track(p1_id)
            t2 = tracker.get_track(p2_id)
            assert t1 is not None and t2 is not None

            # Person 1 must be moving right (increasing x)
            person1_coords.append(t1.bbox[0])
            # Person 2 must be moving left (decreasing x)
            person2_coords.append(t2.bbox[0])

    # After crossing (frame 40):
    # Person 1 must now be at x ~ 500
    # Person 2 must now be at x ~ 100
    track1 = tracker.get_track(p1_id)
    track2 = tracker.get_track(p2_id)

    assert track1.bbox[0] > 400.0, "Person 1 should have crossed to the right"
    assert track2.bbox[0] < 200.0, "Person 2 should have crossed to the left"

    # Monotonicity check: verify no ID swaps occurred mid-flight
    assert person1_coords[-1] > person1_coords[0]
    assert person2_coords[-1] < person2_coords[0]


def test_brief_occlusion_and_track_recovery_within_grace_period():
    """Verify that a track transitions to 'lost' during occlusion and recovers without ID flip."""
    tracker = PersonTracker(session_id="session_occ", lifetime=2.5, confirm_hits=2)

    # 1. Person appears and is confirmed
    dt = 0.1
    for i in range(5):
        t = i * dt
        box = (100.0 + i * 2.0, 100.0, 150.0 + i * 2.0, 220.0)
        active = tracker.update([PersonDetection(bbox=box, confidence=0.90)], timestamp=t)
        assert len(active) == 1
        initial_id = active[0].track_id

    assert tracker.get_track(initial_id).state == "active"
    assert tracker.get_track(initial_id).confirmed is True

    # 2. Brief occlusion: person disappears for 8 frames (0.8s, within 2.5s lifetime)
    for i in range(5, 13):
        t = i * dt
        active = tracker.update([], timestamp=t)
        assert len(active) == 0

        # Status must transition to "lost"
        track = tracker.get_track(initial_id)
        assert track is not None
        assert track.state == "lost"
        assert initial_id in [trk.track_id for trk in tracker.lost_tracks]

    # 3. Person re-appears at frame 13 (t = 1.3s) near predicted position
    reappear_box = (100.0 + 13 * 2.0, 100.0, 150.0 + 13 * 2.0, 220.0)
    active = tracker.update([PersonDetection(bbox=reappear_box, confidence=0.91)], timestamp=1.3)

    assert len(active) == 1
    recovered_track = active[0]

    # Track ID must be preserved
    assert recovered_track.track_id == initial_id
    assert recovered_track.state == "active"
    assert recovered_track.hits == 6
    assert recovered_track.last_seen == pytest.approx(1.3, abs=1e-4)


def test_track_retirement_and_status_transitions():
    """Verify state transitions: active -> lost -> ended after lifetime expiration."""
    tracker = PersonTracker(session_id="session_retire", lifetime=2.0, confirm_hits=2)

    # Frame 1: Creation at t=0.0
    active = tracker.update([PersonDetection(bbox=(50, 50, 100, 150), confidence=0.85)], timestamp=0.0)
    assert len(active) == 1
    tid = active[0].track_id
    assert active[0].state == "active"

    # Frame 2: Hit at t=0.1 -> confirmed
    active = tracker.update([PersonDetection(bbox=(52, 50, 102, 150), confidence=0.87)], timestamp=0.1)
    assert active[0].state == "active"
    assert active[0].confirmed is True

    # Frame 3: Missing at t=0.2 -> transitions to "lost"
    active = tracker.update([], timestamp=0.2)
    assert len(active) == 0
    trk = tracker.get_track(tid)
    assert trk is not None
    assert trk.state == "lost"
    assert trk in tracker.lost_tracks

    # Frame 4: Still missing at t=1.5 (1.5 - 0.1 = 1.4s < 2.0s lifetime) -> remains "lost"
    tracker.update([], timestamp=1.5)
    trk = tracker.get_track(tid)
    assert trk is not None
    assert trk.state == "lost"

    # Frame 5: Missing at t=2.5 (2.5 - 0.1 = 2.4s > 2.0s lifetime) -> transitions to "ended"
    tracker.update([], timestamp=2.5)
    assert len(tracker.active_tracks) == 0
    assert len(tracker.lost_tracks) == 0

    retired_trk = tracker.get_track(tid)
    assert retired_trk is not None
    assert retired_trk.state == "ended"
    assert tid in [t.track_id for t in tracker.ended_tracks]


def test_bytetrack_low_confidence_recovery_during_occlusion():
    """Verify that ByteTrack associates low-score detections to prevent track loss."""
    tracker = PersonTracker(
        session_id="session_bytetrack",
        high_score_threshold=0.5,
        low_score_threshold=0.1,
        lifetime=2.5,
        confirm_hits=2,
    )

    # Frames 0-2: Strong detections
    for i in range(3):
        t = i * 0.1
        box = (50.0 + i * 2.0, 50.0, 100.0 + i * 2.0, 150.0)
        active = tracker.update([PersonDetection(bbox=box, confidence=0.90)], timestamp=t)
        assert len(active) == 1
        tid = active[0].track_id

    # Frame 3: Partial occlusion causes confidence to drop to 0.25 (< 0.5 high threshold)
    t = 3 * 0.1
    occ_box = (50.0 + 3 * 2.0, 50.0, 100.0 + 3 * 2.0, 150.0)
    low_conf_det = PersonDetection(bbox=occ_box, confidence=0.25)

    active = tracker.update([low_conf_det], timestamp=t)

    # Should match existing track via ByteTrack stage 2
    assert len(active) == 1
    assert active[0].track_id == tid
    assert active[0].state == "active"
    assert active[0].hits == 4


def test_reset_session_epochs_cleanly():
    """Verify that resetting session epochs clears state and starts fresh without leaks."""
    tracker = PersonTracker(session_id="epoch_001")

    # Session 1: Create 2 tracks
    dets1 = [
        PersonDetection(bbox=(10, 10, 50, 100), confidence=0.9),
        PersonDetection(bbox=(100, 10, 150, 100), confidence=0.9),
    ]
    active1 = tracker.update(dets1, timestamp=0.0)
    assert len(active1) == 2
    assert {t.session_id for t in active1} == {"epoch_001"}

    # Reset with new epoch
    tracker.reset_epoch("epoch_002")

    assert len(tracker.active_tracks) == 0
    assert len(tracker.lost_tracks) == 0
    assert len(tracker.ended_tracks) == 0

    # Session 2: Create new track
    dets2 = [PersonDetection(bbox=(20, 20, 60, 110), confidence=0.9)]
    active2 = tracker.update(dets2, timestamp=10.0)

    assert len(active2) == 1
    assert active2[0].session_id == "epoch_002"
    # Track ID counter starts fresh
    assert active2[0].track_id == 1
    assert active2[0].first_seen == 10.0


def test_multi_person_tracking_capacity_and_low_latency():
    """Verify system handles 20-30 simultaneous active tracks with low latency (<5ms/frame)."""
    tracker = PersonTracker(session_id="crowd_sim", lifetime=2.0, confirm_hits=2)

    num_people = 28
    num_frames = 40
    dt = 0.0333  # 30 fps

    # Generate initial positions and velocities for 28 people
    np.random.seed(42)
    initial_boxes = []
    velocities = []

    for i in range(num_people):
        x = float(np.random.uniform(50, 800))
        y = float(np.random.uniform(50, 500))
        w = float(np.random.uniform(35, 60))
        h = float(np.random.uniform(90, 140))
        initial_boxes.append([x, y, x + w, y + h])
        vx = float(np.random.uniform(-40, 40))  # px/s
        vy = float(np.random.uniform(-10, 10))
        velocities.append((vx, vy))

    # Pre-generate frame detections to measure pure tracker algorithm latency
    frame_batches = []
    for frame_idx in range(num_frames):
        t = frame_idx * dt
        current_dets = []
        for i in range(num_people):
            box = initial_boxes[i]
            vx, vy = velocities[i]
            x1 = box[0] + vx * t
            y1 = box[1] + vy * t
            x2 = box[2] + vx * t
            y2 = box[3] + vy * t
            current_dets.append(PersonDetection(bbox=(x1, y1, x2, y2), confidence=0.85))
        frame_batches.append((current_dets, t))

    # Benchmark tracking loop
    t0 = time.perf_counter()
    for current_dets, t in frame_batches:
        active = tracker.update(current_dets, timestamp=t)

    total_time = time.perf_counter() - t0
    avg_frame_ms = (total_time / num_frames) * 1000.0

    # Low latency requirement: well under 5ms per frame for 28 simultaneous tracks
    assert avg_frame_ms < 5.0, f"Average tracking latency too high: {avg_frame_ms:.2f} ms/frame"
    assert len(tracker.active_tracks) == num_people


def test_input_detection_formats_interoperability():
    """Verify flexible input formats (PersonDetection, raw boxes, numpy array, dicts)."""
    tracker = PersonTracker(session_id="interop_test")

    # 1. PersonDetection instances
    d1 = [PersonDetection(bbox=(10, 10, 50, 100), confidence=0.85)]
    out1 = tracker.update(d1, timestamp=0.0)
    assert len(out1) == 1

    # 2. Raw tuples [x1, y1, x2, y2, score]
    d2 = [(12.0, 10.0, 52.0, 100.0, 0.88)]
    out2 = tracker.update(d2, timestamp=0.1)
    assert len(out2) == 1
    assert out2[0].track_id == out1[0].track_id

    # 3. NumPy ndarray shape (1, 5)
    d3 = np.array([[14.0, 10.0, 54.0, 100.0, 0.91]], dtype=np.float32)
    out3 = tracker.update(d3, timestamp=0.2)
    assert len(out3) == 1
    assert out3[0].track_id == out1[0].track_id

    # 4. Dict format
    d4 = [{"bbox": (16.0, 10.0, 56.0, 100.0), "confidence": 0.89}]
    out4 = tracker.update(d4, timestamp=0.3)
    assert len(out4) == 1
    assert out4[0].track_id == out1[0].track_id


def test_strict_anonymity_and_contract_serialization():
    """Verify strictly anonymous tracking with contract serialization."""
    tracker = PersonTracker(session_id="session_anon_9")
    active = tracker.update([PersonDetection(bbox=(20, 30, 80, 160), confidence=0.93)], timestamp=1.5)

    assert len(active) == 1
    track = active[0]

    # Verify no biometric vectors exist on the track
    assert not hasattr(track, "face_embedding")
    assert not hasattr(track, "reid_features")
    assert not hasattr(track, "biometric_id")

    # Verify contract serialization to dict
    d = track.to_dict()
    assert d["track_id"] == track.track_id
    assert d["session_id"] == "session_anon_9"
    assert d["first_seen"] == 1.5
    assert d["last_seen"] == 1.5
    assert d["duration"] == 0.0
    assert d["state"] == "active"
    assert d["hits"] == 1
    assert isinstance(d["bbox"], list)
    assert len(d["bbox"]) == 4

    # Tracker state summary serialization
    st = tracker.state()
    assert st["session_id"] == "session_anon_9"
    assert st["active_tracks_count"] == 1
    assert len(st["tracks"]) == 1


def test_geometry_iou_and_diou_helpers():
    """Verify geometry utility functions for IoU and DIoU."""
    b1 = (0.0, 0.0, 10.0, 10.0)
    b2 = (0.0, 0.0, 10.0, 10.0)
    assert compute_iou(b1, b2) == pytest.approx(1.0)
    assert compute_diou(b1, b2) == pytest.approx(1.0)

    # Disjoint boxes
    b3 = (20.0, 20.0, 30.0, 30.0)
    assert compute_iou(b1, b3) == 0.0
    # DIoU should be negative for disjoint boxes
    assert compute_diou(b1, b3) < 0.0
