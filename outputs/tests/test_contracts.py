"""Unit tests verifying shared interface contracts."""
import numpy as np
from services.face_contracts import (
    FramePacket, PersonDetection, PersonTrack, FacePose,
    FaceQualityResult, FaceObservation, CaptureCandidate,
    CaptureRecord, PersonProfile, AnalysisState, ProcessingMetrics
)


def test_contracts_instantiation_and_serialization():
    # FramePacket
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = FramePacket(
        sequence=1, timestamp=100.0, image=img, source_id="cam1",
        epoch="session_1", width=640, height=480
    )
    d_frame = frame.to_dict()
    assert d_frame["sequence"] == 1
    assert d_frame["width"] == 640

    # PersonDetection
    det = PersonDetection(bbox=(10.0, 20.0, 100.0, 200.0), confidence=0.85)
    d_det = det.to_dict()
    assert d_det["confidence"] == 0.85
    assert len(d_det["bbox"]) == 4

    # PersonTrack
    track = PersonTrack(
        track_id=1, session_id="session_1:1", first_seen=10.0, last_seen=15.0,
        bbox=(10.0, 20.0, 100.0, 200.0), confidence=0.88, confirmed=True
    )
    assert track.duration == 5.0
    d_track = track.to_dict()
    assert d_track["duration"] == 5.0
    assert d_track["confirmed"] is True

    # FacePose
    pose = FacePose(yaw=-15.5, pitch=2.0, roll=0.5, category="left_angle", confidence=0.92)
    d_pose = pose.to_dict()
    assert d_pose["category"] == "left_angle"

    # FaceQualityResult
    quality = FaceQualityResult(
        score=84.5, sharpness=120.0, motion_blur=0.08, resolution_score=90.0,
        lighting_score=85.0, pose_score=80.0, size_score=85.0, is_acceptable=True
    )
    d_qual = quality.to_dict()
    assert d_qual["score"] == 84.5
    assert d_qual["is_acceptable"] is True

    # FaceObservation
    obs = FaceObservation(
        observation_id="obs_1", track_id=1, timestamp=12.0,
        bbox=(30.0, 25.0, 80.0, 75.0), confidence=0.95,
        landmarks=[(40.0, 40.0), (70.0, 40.0), (55.0, 55.0), (45.0, 68.0), (65.0, 68.0)],
        pose=pose, quality=quality
    )
    d_obs = obs.to_dict()
    assert d_obs["observation_id"] == "obs_1"
    assert len(d_obs["landmarks"]) == 5

    # CaptureCandidate
    candidate = CaptureCandidate(
        candidate_id="cand_1", track_id=1, timestamp=12.0, category="frontal",
        quality_score=88.0, pose=pose, crop_original=np.zeros((50, 50, 3), dtype=np.uint8),
        frame_sequence=1, bbox_in_frame=(30.0, 25.0, 80.0, 75.0)
    )
    assert candidate.quality_score == 88.0

    # CaptureRecord
    record = CaptureRecord(
        capture_id="cap_1", track_id=1, session_id="session_1:1", timestamp=12.0,
        category="frontal", quality_score=88.0, pose_category="frontal", yaw=0.0, pitch=0.0,
        original_path="data/captures/cap_1_orig.jpg", crop_width=120, crop_height=120
    )
    d_rec = record.to_dict()
    assert d_rec["capture_id"] == "cap_1"
    assert "/api/people/captures/cap_1/image" in d_rec["original_url"]

    # PersonProfile
    profile = PersonProfile(
        session_id="session_1:1", track_id=1, camera_id="cam1",
        first_seen=10.0, last_seen=15.0, duration=5.0, status="active",
        best_capture=record, captures_by_category={"frontal": record},
        total_observations=5, best_quality_score=88.0
    )
    d_prof = profile.to_dict()
    assert d_prof["session_id"] == "session_1:1"
    assert d_prof["best_capture"]["capture_id"] == "cap_1"

    # AnalysisState
    state = AnalysisState(is_running=True, fps=15.2, active_tracks_count=3)
    d_state = state.to_dict()
    assert d_state["is_running"] is True
    assert d_state["fps"] == 15.2

    # ProcessingMetrics
    metrics = ProcessingMetrics(frame_sequence=10, acquisition_fps=20.0, detection_ms=18.5)
    d_metrics = metrics.to_dict()
    assert d_metrics["detection_ms"] == 18.5
