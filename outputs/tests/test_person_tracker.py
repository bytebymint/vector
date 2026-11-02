"""Unit tests verifying Face Lock-On & Relative Head Tracking for PersonTracker.

Mandate:
- Lock on detected face and keep tracking it until the person exits the screen entirely.
- The face box is anchored relative to the person's bounding box and moves in 100%
  lockstep with the person during camera pans and movements.
- Never drift into empty space when camera moves or face detector skips frames.
- Smoothly update relative head position via EMA when fresh face observations arrive.
"""
from typing import List, Tuple
import pytest

from services.face_contracts import PersonDetection, PersonTrack
from services.person_tracker import PersonTracker


def test_face_lock_on_and_lockstep_movement_during_camera_pan():
    """Verify that when a camera pans, the locked face box moves 100% in lockstep with the person."""
    tracker = PersonTracker(session_id="pan_test", lifetime=3.0, confirm_hits=1)

    # Frame 0: Person detected at (100, 100, 200, 400) [w=100, h=300]
    person_box_0 = (100.0, 100.0, 200.0, 400.0)
    # Face detected at (125, 120, 175, 180) [center=(150, 150), w=50, h=60]
    face_box_0 = (125.0, 120.0, 175.0, 180.0)

    # Update tracker with person and face
    tracks = tracker.update(
        detections=[PersonDetection(bbox=person_box_0, confidence=0.92)],
        timestamp=0.0,
        face_observations=[{"bbox": face_box_0, "confidence": 0.88}],
    )

    assert len(tracks) == 1
    track = tracks[0]
    assert track.locked_face is not None
    assert track.face_bbox is not None
    assert track.face_relative_box is not None

    # Expected relative coordinates:
    # rel_x1 = (125 - 100)/100 = 0.25
    # rel_y1 = (120 - 100)/300 = 0.0667
    # rel_x2 = (175 - 100)/100 = 0.75
    # rel_y2 = (180 - 100)/300 = 0.2667
    r0, r1, r2, r3 = track.face_relative_box
    assert r0 == pytest.approx(0.25, abs=1e-3)
    assert r1 == pytest.approx(0.0667, abs=1e-3)
    assert r2 == pytest.approx(0.75, abs=1e-3)
    assert r3 == pytest.approx(0.2667, abs=1e-3)

    # Initial face offset relative to person top-left
    init_offset_x1 = track.face_bbox[0] - track.bbox[0]
    init_offset_y1 = track.face_bbox[1] - track.bbox[1]
    init_offset_x2 = track.face_bbox[2] - track.bbox[0]
    init_offset_y2 = track.face_bbox[3] - track.bbox[1]

    # Simulate camera panning violently across 15 frames:
    # Camera pans by +45 px horizontally and +15 px vertically each frame.
    # Face detector SKIPS all these frames (no face observations passed).
    for frame_idx in range(1, 16):
        t = frame_idx * 0.0333
        dx = frame_idx * 45.0
        dy = frame_idx * 15.0

        p_box = (100.0 + dx, 100.0 + dy, 200.0 + dx, 400.0 + dy)

        # Update person track without any face detections
        tracks = tracker.update(
            detections=[PersonDetection(bbox=p_box, confidence=0.90)],
            timestamp=t,
            face_observations=None,  # Face detector skipped!
        )

        assert len(tracks) == 1
        curr_track = tracks[0]

        # The face MUST remain locked
        assert curr_track.locked_face is not None
        assert curr_track.face_bbox is not None

        # Verify 100% lockstep movement:
        # Offsets relative to person top-left must remain identical!
        curr_offset_x1 = curr_track.face_bbox[0] - curr_track.bbox[0]
        curr_offset_y1 = curr_track.face_bbox[1] - curr_track.bbox[1]
        curr_offset_x2 = curr_track.face_bbox[2] - curr_track.bbox[0]
        curr_offset_y2 = curr_track.face_bbox[3] - curr_track.bbox[1]

        assert curr_offset_x1 == pytest.approx(init_offset_x1, abs=0.5)
        assert curr_offset_y1 == pytest.approx(init_offset_y1, abs=0.5)
        assert curr_offset_x2 == pytest.approx(init_offset_x2, abs=0.5)
        assert curr_offset_y2 == pytest.approx(init_offset_y2, abs=0.5)

        # Verify absolute coordinates shifted by exactly dx, dy
        assert curr_track.face_bbox[0] == pytest.approx(face_box_0[0] + dx, abs=1.0)
        assert curr_track.face_bbox[1] == pytest.approx(face_box_0[1] + dy, abs=1.0)
        assert curr_track.face_bbox[2] == pytest.approx(face_box_0[2] + dx, abs=1.0)
        assert curr_track.face_bbox[3] == pytest.approx(face_box_0[3] + dy, abs=1.0)


def test_face_relative_box_ema_smoothing():
    """Verify that when fresh face detections arrive, relative box updates smoothly via EMA."""
    track = PersonTrack(
        track_id=1,
        session_id="test_ema",
        first_seen=0.0,
        last_seen=0.0,
        bbox=(100.0, 100.0, 200.0, 400.0),
        confidence=0.9,
    )

    # Initial face lock
    face_1 = (120.0, 120.0, 170.0, 180.0)
    track.lock_face(face_1, confidence=0.85, timestamp=0.0)

    old_rel = track.face_relative_box
    assert old_rel is not None

    # New face detection with slight shift
    face_2 = (130.0, 120.0, 180.0, 180.0)
    # With alpha=0.3: new_rel should be 0.7 * old_rel + 0.3 * target_rel
    track.lock_face(face_2, confidence=0.90, timestamp=0.1, alpha=0.3)

    new_rel = track.face_relative_box
    target_rel_0 = (130.0 - 100.0) / 100.0  # 0.30
    expected_rel_0 = 0.7 * old_rel[0] + 0.3 * target_rel_0

    assert new_rel[0] == pytest.approx(expected_rel_0, abs=1e-4)


def test_face_lock_maintained_until_person_ends():
    """Verify face lock is maintained through active and lost states until track ends."""
    tracker = PersonTracker(session_id="lifecycle_test", lifetime=2.0, confirm_hits=1)

    # Frame 1: Person and face appear
    tracks = tracker.update(
        detections=[PersonDetection(bbox=(100, 100, 200, 400), confidence=0.9)],
        timestamp=0.0,
        face_observations=[(125.0, 120.0, 175.0, 180.0, 0.9)],
    )
    tid = tracks[0].track_id
    assert tracks[0].locked_face is not None
    assert tracks[0].state == "active"

    # Frame 2: Person missing (simulating brief disappearance at t=0.5s < 2.0s lifetime)
    tracks = tracker.update(detections=[], timestamp=0.5, return_lost=True)
    lost_track = tracker.get_track(tid)
    assert lost_track is not None
    assert lost_track.state == "lost"
    # Even in lost state, locked face is preserved and updated from predicted body position
    assert lost_track.locked_face is not None
    assert lost_track.face_bbox is not None

    # Frame 3: Missing past lifetime (t=2.6s > 2.0s lifetime)
    tracker.update(detections=[], timestamp=2.6, return_lost=True)
    ended_track = tracker.get_track(tid)
    assert ended_track is not None
    assert ended_track.state == "ended"


def test_associate_faces_with_multiple_people():
    """Verify correct association of multiple faces to corresponding people."""
    tracker = PersonTracker(session_id="multi_test", lifetime=3.0, confirm_hits=1)

    person_left = (100.0, 100.0, 200.0, 400.0)
    person_right = (500.0, 100.0, 600.0, 400.0)

    face_left = (125.0, 120.0, 175.0, 180.0)
    face_right = (525.0, 120.0, 575.0, 180.0)

    tracks = tracker.update(
        detections=[
            PersonDetection(bbox=person_left, confidence=0.9),
            PersonDetection(bbox=person_right, confidence=0.9),
        ],
        timestamp=0.0,
        face_observations=[face_left, face_right],
    )

    assert len(tracks) == 2
    track_left = min(tracks, key=lambda t: t.bbox[0])
    track_right = max(tracks, key=lambda t: t.bbox[0])

    assert track_left.face_bbox is not None
    assert track_right.face_bbox is not None

    # Face left should be on person left (~125)
    assert track_left.face_bbox[0] == pytest.approx(125.0, abs=2.0)
    # Face right should be on person right (~525)
    assert track_right.face_bbox[0] == pytest.approx(525.0, abs=2.0)


def test_landmarks_lockstep_movement():
    """Verify facial landmarks move in exact lockstep with person body bounding box."""
    tracker = PersonTracker(session_id="landmark_test", confirm_hits=1)

    p_box = (100.0, 100.0, 200.0, 400.0)
    f_box = (120.0, 120.0, 180.0, 180.0)
    landmarks = [(135.0, 140.0), (165.0, 140.0), (150.0, 155.0), (140.0, 170.0), (160.0, 170.0)]

    tracker.update(
        detections=[PersonDetection(bbox=p_box, confidence=0.9)],
        timestamp=0.0,
        face_observations=[{"bbox": f_box, "confidence": 0.95, "landmarks": landmarks}],
    )

    t0 = tracker.active_tracks[0]
    assert len(t0.face_landmarks) == 5
    init_lm_0 = t0.face_landmarks[0]

    # Camera pans by dx=80, dy=40
    tracker.update(
        detections=[PersonDetection(bbox=(180.0, 140.0, 280.0, 440.0), confidence=0.9)],
        timestamp=0.1,
    )

    t1 = tracker.active_tracks[0]
    # Landmark 0 should have moved by dx=80, dy=40
    assert t1.face_landmarks[0][0] == pytest.approx(init_lm_0[0] + 80.0, abs=1.0)
    assert t1.face_landmarks[0][1] == pytest.approx(init_lm_0[1] + 40.0, abs=1.0)


def test_person_track_contract_serialization_with_locked_face():
    """Verify serialization to dictionary contains all locked face fields."""
    track = PersonTrack(
        track_id=42,
        session_id="contract_test",
        first_seen=10.0,
        last_seen=12.0,
        bbox=(50.0, 60.0, 150.0, 260.0),
        confidence=0.95,
    )

    track.lock_face(
        face_bbox=(70.0, 75.0, 130.0, 135.0),
        confidence=0.88,
        landmarks=[(85.0, 95.0), (115.0, 95.0)],
        timestamp=12.0,
    )

    d = track.to_dict()
    assert d["locked_face"] is not None
    assert d["locked_face"]["is_locked"] is True
    assert d["face_bbox"] == [70.0, 75.0, 130.0, 135.0]
    assert d["face_confidence"] == 0.88
    assert len(d["face_landmarks"]) == 2
    assert d["face_relative_box"] is not None
    assert d["face_last_seen"] == 12.0
