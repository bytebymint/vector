"""End-to-end integration and reliability test suite for Multi-Person Facial Capture System.

Comprehensive test coverage verifying pipeline components under realistic edge conditions:
1. Simultaneous tracking of multiple people (5 to 10 synthetic tracks)
2. Face association with correct person track (avoiding wrong assignment)
3. Observable head pose angle classification (frontal, left, right, profile, uncertain)
4. Quality score evaluation and candidate upgrade (initial low quality -> upgraded high quality)
5. Candidate replacement logic across categories (overall, frontal, left_angle, right_angle, high_res)
6. Start / Stop lifecycle transitions and thread cleanliness
7. Safe profile retrieval and original image extraction
8. Graceful recovery under missing models, synthetic frame drops, and corrupted frames
9. Full end-to-end multi-person capture pipeline integration workflow
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import queue
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

from services.face_contracts import (
    AnalysisState,
    CaptureCandidate,
    CaptureRecord,
    FaceObservation,
    FacePose,
    FaceQualityResult,
    FramePacket,
    PersonDetection,
    PersonProfile,
    PersonTrack,
    ProcessingMetrics,
)
from services.person_tracker import PersonTracker
from services.face_detector import (
    FaceDetector,
    FaceDetectorConfig,
    box_iou,
    point_in_box,
)
from services.face_pose import (
    FacePoseEstimator,
    estimate_face_pose,
    categorize_orientation,
    project_reference_model,
)
from services.face_quality import (
    FaceQualityAssessor,
    QualityThresholds,
    QualityWeights,
)
from services.capture_selector import (
    CATEGORY_FRONTAL,
    CATEGORY_HIGH_RES,
    CATEGORY_LEFT_ANGLE,
    CATEGORY_OVERALL,
    CATEGORY_RIGHT_ANGLE,
    CaptureSelector,
    CaptureSelectorConfig,
    determine_pose_category,
)
from services.profile_repository import ProfileRepository
from services.analysis_lifecycle import (
    AnalysisLifecycleController,
    AnalysisLifecycleState,
)


# =============================================================================
# Clean Synthetic Data & Image Generation Utilities
# =============================================================================

def create_synthetic_canvas(
    width: int = 1280,
    height: int = 720,
    bg_color: Tuple[int, int, int] = (160, 160, 160),
) -> np.ndarray:
    """Create a uniform synthetic background canvas."""
    return np.full((height, width, 3), bg_color, dtype=np.uint8)


def render_synthetic_person(
    frame: np.ndarray,
    cx: int,
    cy: int,
    scale: float = 1.0,
    yaw: float = 0.0,
    blur: bool = False,
    noise: bool = False,
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    """Render a synthetic person with distinguishable head features and torso.
    
    Generates facial structure recognizable by both Haar and YuNet:
    - Head oval (skin tone)
    - Eyes (sclera + pupils) with yaw lateral offset
    - Nose bridge
    - Mouth arc
    - Torso/body rectangle underneath
    
    Returns:
        (person_bbox, face_bbox): Bounding boxes as (x1, y1, x2, y2).
    """
    rx = max(10, int(40 * scale))
    ry = max(12, int(50 * scale))
    fh, fw = frame.shape[:2]

    # Constrain center within frame
    cx = max(rx + 10, min(fw - rx - 10, int(cx)))
    cy = max(ry + 10, min(fh - 300, int(cy)))

    # 1. Head oval (skin tone)
    cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, (180, 210, 240), -1)

    # Lateral shift based on yaw angle
    yaw_shift = int(yaw / 5.0 * scale)

    # 2. Eyes
    eye_dx = max(4, int(15 * scale))
    eye_dy = int(-10 * scale)
    eye_r = max(2, int(6 * scale))
    pupil_r = max(1, int(3 * scale))

    left_eye_center = (cx - eye_dx + yaw_shift, cy + eye_dy)
    right_eye_center = (cx + eye_dx + yaw_shift, cy + eye_dy)

    cv2.circle(frame, left_eye_center, eye_r, (255, 255, 255), -1)
    cv2.circle(frame, right_eye_center, eye_r, (255, 255, 255), -1)
    cv2.circle(frame, left_eye_center, pupil_r, (30, 30, 30), -1)
    cv2.circle(frame, right_eye_center, pupil_r, (30, 30, 30), -1)

    # 3. Nose
    nose_y = cy - int(5 * scale)
    cv2.line(
        frame,
        (cx + yaw_shift, nose_y),
        (cx + yaw_shift, nose_y + max(4, int(15 * scale))),
        (130, 160, 190),
        max(1, int(2 * scale)),
    )

    # 4. Mouth
    mouth_y = cy + int(22 * scale)
    cv2.ellipse(
        frame,
        (cx + yaw_shift, mouth_y),
        (max(3, int(14 * scale)), max(2, int(6 * scale))),
        0,
        0,
        180,
        (40, 40, 180),
        -1,
    )

    # 5. Torso and legs (Person body)
    bw = max(20, int(50 * scale))
    bh = max(60, int(180 * scale))
    cv2.rectangle(
        frame,
        (cx - bw, cy + ry),
        (cx + bw, min(fh - 1, cy + ry + bh)),
        (80, 60, 50),
        -1,
    )

    face_bbox = (
        float(cx - rx),
        float(cy - ry),
        float(cx + rx),
        float(cy + ry),
    )
    person_bbox = (
        float(cx - bw - 10),
        float(cy - ry - 10),
        float(cx + bw + 10),
        float(min(fh - 1, cy + ry + bh + 10)),
    )

    if blur:
        fx1 = max(0, int(face_bbox[0]))
        fy1 = max(0, int(face_bbox[1]))
        fx2 = min(fw, int(face_bbox[2]))
        fy2 = min(fh, int(face_bbox[3]))
        if fx2 > fx1 and fy2 > fy1:
            frame[fy1:fy2, fx1:fx2] = cv2.GaussianBlur(
                frame[fy1:fy2, fx1:fx2], (25, 25), 12.0
            )

    if noise:
        fx1 = max(0, int(face_bbox[0]))
        fy1 = max(0, int(face_bbox[1]))
        fx2 = min(fw, int(face_bbox[2]))
        fy2 = min(fh, int(face_bbox[3]))
        if fx2 > fx1 and fy2 > fy1:
            noise_arr = np.random.normal(0, 25, (fy2 - fy1, fx2 - fx1, 3)).astype(np.int16)
            crop = frame[fy1:fy2, fx1:fx2].astype(np.int16) + noise_arr
            frame[fy1:fy2, fx1:fx2] = np.clip(crop, 0, 255).astype(np.uint8)

    return person_bbox, face_bbox


# =============================================================================
# 1. Simultaneous Tracking of Multiple People (5 to 10 synthetic tracks)
# =============================================================================

class TestMultiPersonSimultaneousTracking:
    """Verifies simultaneous tracking of 5 to 10 individuals across frames."""

    def test_simultaneous_tracking_8_to_10_people(self):
        """Track 10 independent people with distinct linear trajectories over 15 frames."""
        tracker = PersonTracker(session_id="multi_track_session", lifetime=2.0)
        num_people = 10

        # Define 10 distinct trajectories across the scene
        people = []
        for i in range(num_people):
            start_x = 80.0 + i * 110.0
            start_y = 150.0 + (i % 3) * 60.0
            vx = 6.0 * (1.0 if i % 2 == 0 else -1.0)
            vy = 3.0 * (1.0 if (i // 2) % 2 == 0 else -1.0)
            people.append({"start_x": start_x, "start_y": start_y, "vx": vx, "vy": vy})

        active_history: Dict[int, List[int]] = {}

        for frame_idx in range(15):
            t = frame_idx * 0.1
            detections = []
            for p in people:
                cur_x = p["start_x"] + p["vx"] * frame_idx
                cur_y = p["start_y"] + p["vy"] * frame_idx
                w, h = 70.0, 180.0
                bbox = (cur_x - w / 2, cur_y - h / 2, cur_x + w / 2, cur_y + h / 2)
                detections.append(PersonDetection(bbox=bbox, confidence=0.92))

            active_tracks = tracker.update(detections, timestamp=t)
            assert len(active_tracks) == num_people

            for trk in active_tracks:
                active_history.setdefault(trk.track_id, []).append(frame_idx)

        # Assert exactly 10 unique track IDs were created (no ID switching / proliferation)
        assert len(active_history) == num_people
        assert sorted(active_history.keys()) == list(range(1, num_people + 1))

        # Check track stability, hits, confirmation, and speed
        for track_id, frames in active_history.items():
            assert len(frames) == 15
            track = tracker.get_track(track_id)
            assert track is not None
            assert track.confirmed is True
            assert track.hits >= 15
            assert track.duration >= 1.3
            assert track.speed_px_s > 0.0
            assert len(track.trajectory) >= 15

    def test_crossing_trajectories_avoid_id_switch(self):
        """Verify that two people walking towards and past each other maintain their IDs."""
        tracker = PersonTracker(session_id="crossing_session", lifetime=2.0)

        # Subject A moves left-to-right (100 -> 400)
        # Subject B moves right-to-left (400 -> 100)
        # They intersect around frame 5 at x=250
        num_frames = 11
        for frame_idx in range(num_frames):
            t = frame_idx * 0.1
            xa = 100.0 + 30.0 * frame_idx
            xb = 400.0 - 30.0 * frame_idx
            ya = 220.0
            yb = 220.0

            det_a = PersonDetection(bbox=(xa - 35, ya - 90, xa + 35, ya + 90), confidence=0.95)
            det_b = PersonDetection(bbox=(xb - 35, yb - 90, xb + 35, yb + 90), confidence=0.95)

            active = tracker.update([det_a, det_b], timestamp=t)
            assert len(active) == 2

        track_1 = tracker.get_track(1)
        track_2 = tracker.get_track(2)
        assert track_1 is not None and track_2 is not None

        # Verify Track 1 finished on the right side (~400) and Track 2 on the left side (~100)
        t1_cx = (track_1.bbox[0] + track_1.bbox[2]) / 2.0
        t2_cx = (track_2.bbox[0] + track_2.bbox[2]) / 2.0
        assert t1_cx > 380.0
        assert t2_cx < 120.0

    def test_variable_speed_and_scale_tracks(self):
        """Tracker accurately handles subjects moving at disparate speeds and bounding box scales."""
        tracker = PersonTracker(session_id="variable_scale_session", lifetime=2.5)

        for frame_idx in range(10):
            t = frame_idx * 0.1
            # Fast subject: running at 15 px/frame (150 px/s) with smaller scale (distant)
            fast_x = 100.0 + 15.0 * frame_idx
            fast_bbox = (fast_x - 35, 100.0, fast_x + 35, 240.0)

            # Slow subject: walking at 2 px/frame (20 px/s) with large scale (near camera)
            slow_x = 500.0 + 2.0 * frame_idx
            slow_bbox = (slow_x - 60, 200.0, slow_x + 60, 500.0)

            dets = [
                PersonDetection(bbox=fast_bbox, confidence=0.88),
                PersonDetection(bbox=slow_bbox, confidence=0.95),
            ]
            active = tracker.update(dets, timestamp=t)
            assert len(active) == 2

        fast_trk = tracker.get_track(1)
        slow_trk = tracker.get_track(2)
        assert fast_trk is not None and slow_trk is not None
        assert fast_trk.speed_px_s > slow_trk.speed_px_s
        assert (fast_trk.bbox[2] - fast_trk.bbox[0]) < (slow_trk.bbox[2] - slow_trk.bbox[0])


# =============================================================================
# 2. Face Association with Correct Person Track (Avoiding Wrong Assignment)
# =============================================================================

class TestFaceTrackAssociation:
    """Verifies geometric face-to-person association and ambiguity rejection."""

    def test_multi_person_face_association_exact_match(self):
        """Verify 5 individuals simultaneously in a frame have their faces assigned to the correct track."""
        detector = FaceDetector(FaceDetectorConfig(backend="yunet", score_threshold=0.3))
        canvas = create_synthetic_canvas(1280, 720)

        tracks = []
        raw_faces = []

        # Position 5 persons across the scene
        positions = [
            (150, 180),
            (380, 180),
            (620, 180),
            (860, 180),
            (1100, 180),
        ]

        for i, (cx, cy) in enumerate(positions, start=1):
            p_box, f_box = render_synthetic_person(canvas, cx, cy, scale=1.0)
            track = PersonTrack(
                track_id=i,
                session_id="session_assoc",
                first_seen=0.0,
                last_seen=1.0,
                bbox=p_box,
                confidence=0.92,
                confirmed=True,
            )
            tracks.append(track)

            raw_faces.append({
                "bbox": f_box,
                "confidence": 0.95,
                "landmarks": [
                    (cx - 15, cy - 10),
                    (cx + 15, cy - 10),
                    (cx, cy + 5),
                    (cx - 12, cy + 22),
                    (cx + 12, cy + 22),
                ],
            })

        # Associate faces with tracks
        observations = detector.associate_faces(raw_faces, tracks, timestamp=1.0, frame=canvas)
        assert len(observations) == 5

        # Verify 1-to-1 exact track ID match
        assigned_track_ids = {obs.track_id for obs in observations}
        assert assigned_track_ids == {1, 2, 3, 4, 5}

        for obs in observations:
            track = next(t for t in tracks if t.track_id == obs.track_id)
            # Face center must be inside the upper region of the matched track
            fcx = (obs.bbox[0] + obs.bbox[2]) / 2.0
            fcy = (obs.bbox[1] + obs.bbox[3]) / 2.0
            assert point_in_box(fcx, fcy, track.bbox)

    def test_rejection_of_ambiguous_overlapping_persons(self):
        """Reject face association when two people overlap heavily to avoid wrong assignment."""
        detector = FaceDetector(FaceDetectorConfig(ambiguity_iou_threshold=0.25))

        # Two heavily overlapping person tracks (IoU ~ 0.50)
        track_1 = PersonTrack(
            track_id=10,
            session_id="ambig",
            first_seen=0.0,
            last_seen=1.0,
            bbox=(200.0, 100.0, 320.0, 400.0),
            confidence=0.90,
            confirmed=True,
        )
        track_2 = PersonTrack(
            track_id=20,
            session_id="ambig",
            first_seen=0.0,
            last_seen=1.0,
            bbox=(220.0, 100.0, 340.0, 400.0),
            confidence=0.90,
            confirmed=True,
        )

        # Face positioned right in the overlapping boundary
        shared_face = {
            "bbox": (230.0, 120.0, 310.0, 220.0),
            "confidence": 0.95,
            "landmarks": [(250, 150), (290, 150), (270, 170), (255, 195), (285, 195)],
        }

        observations = detector.associate_faces([shared_face], [track_1, track_2], timestamp=1.0)
        # Because the ownership is ambiguous and tracks overlap heavily, association MUST be rejected
        assert len(observations) == 0

    def test_rejection_of_out_of_person_and_lower_body_faces(self):
        """Reject detections located on lower body or in the background."""
        detector = FaceDetector(FaceDetectorConfig(upper_region_ratio=0.55))

        person = PersonTrack(
            track_id=1,
            session_id="bounds",
            first_seen=0.0,
            last_seen=1.0,
            bbox=(100.0, 100.0, 220.0, 400.0),
            confidence=0.9,
            confirmed=True,
        )

        # 1. Lower body face (y=320, lower 75% of person height)
        lower_face = {
            "bbox": (130.0, 300.0, 190.0, 370.0),
            "confidence": 0.9,
            "landmarks": [(145, 320), (175, 320), (160, 335), (150, 350), (170, 350)],
        }

        # 2. Distractor face in background (far from person)
        background_face = {
            "bbox": (500.0, 200.0, 560.0, 270.0),
            "confidence": 0.9,
            "landmarks": [(515, 220), (545, 220), (530, 235), (520, 250), (540, 250)],
        }

        obs_lower = detector.associate_faces([lower_face], [person], timestamp=1.0)
        assert len(obs_lower) == 0

        obs_bg = detector.associate_faces([background_face], [person], timestamp=1.0)
        assert len(obs_bg) == 0

    def test_one_to_one_face_matching_constraint(self):
        """Ensure at most one face observation is assigned per track per frame."""
        detector = FaceDetector(FaceDetectorConfig())

        person = PersonTrack(
            track_id=1,
            session_id="one_to_one",
            first_seen=0.0,
            last_seen=1.0,
            bbox=(200.0, 100.0, 340.0, 420.0),
            confidence=0.9,
            confirmed=True,
        )

        face_a = {
            "bbox": (230.0, 120.0, 310.0, 210.0),
            "confidence": 0.96,
            "landmarks": [(250, 150), (290, 150), (270, 165), (255, 185), (285, 185)],
        }
        face_b = {
            "bbox": (235.0, 125.0, 305.0, 205.0),
            "confidence": 0.70,
            "landmarks": [(250, 150), (290, 150), (270, 165), (255, 185), (285, 185)],
        }

        observations = detector.associate_faces([face_a, face_b], [person], timestamp=1.0)
        assert len(observations) == 1
        assert observations[0].track_id == 1
        assert observations[0].confidence == 0.96


# =============================================================================
# 3. Observable Head Pose Angle Classification (Frontal, Left, Right)
# =============================================================================

class TestHeadPoseClassification:
    """Verifies observable 3D head pose estimation and categorical classification."""

    def test_frontal_pose_classification(self):
        """Frontal angles (yaw within ±15 degrees and pitch within ±15 degrees) classify as frontal."""
        for test_yaw in [0.0, -10.0, 12.5]:
            landmarks = project_reference_model(yaw=test_yaw, pitch=0.0, roll=0.0)
            pose = estimate_face_pose(landmarks)

            assert pose.category == "frontal"
            assert -15.0 <= pose.yaw <= 15.0
            assert determine_pose_category(pose) == CATEGORY_FRONTAL

    def test_left_angle_pose_classification(self):
        """Left-facing angles (-45 <= yaw < -15) classify as left_angle."""
        for test_yaw in [-20.0, -30.0, -42.0]:
            landmarks = project_reference_model(yaw=test_yaw, pitch=0.0, roll=0.0)
            pose = estimate_face_pose(landmarks)

            assert pose.category == "left_angle"
            assert -45.0 <= pose.yaw < -15.0
            assert determine_pose_category(pose) == CATEGORY_LEFT_ANGLE

    def test_right_angle_pose_classification(self):
        """Right-facing angles (15 < yaw <= 45) classify as right_angle."""
        for test_yaw in [18.0, 30.0, 44.0]:
            landmarks = project_reference_model(yaw=test_yaw, pitch=0.0, roll=0.0)
            pose = estimate_face_pose(landmarks)

            assert pose.category == "right_angle"
            assert 15.0 < pose.yaw <= 45.0
            assert determine_pose_category(pose) == CATEGORY_RIGHT_ANGLE

    def test_profile_and_extreme_head_poses(self):
        """Extreme yaw angles (|yaw| > 45) classify as profile."""
        for test_yaw in [-60.0, 55.0]:
            landmarks = project_reference_model(yaw=test_yaw, pitch=0.0, roll=0.0)
            pose = estimate_face_pose(landmarks)

            assert pose.category == "profile"
            assert abs(pose.yaw) > 45.0
            # Extreme profiles do not qualify for frontal, left, or right capture categories
            assert determine_pose_category(pose) is None

    def test_distorted_and_degenerate_landmarks_return_uncertain(self):
        """Degenerate landmark geometries produce 'uncertain' category."""
        # 1. Inverted eye order (left eye positioned to the left of right eye)
        inverted = [(300, 200), (240, 200), (270, 230), (250, 260), (290, 260)]
        pose_inv = estimate_face_pose(inverted)
        assert pose_inv.category == "uncertain"

        # 2. Collapsed points (all zero)
        zeros = [(0.0, 0.0)] * 5
        pose_zero = estimate_face_pose(zeros)
        assert pose_zero.category == "uncertain"

        # 3. Mouth above eyes
        mouth_up = [(240, 250), (300, 250), (270, 230), (250, 180), (290, 180)]
        pose_mouth = estimate_face_pose(mouth_up)
        assert pose_mouth.category == "uncertain"


# =============================================================================
# 4. Quality Score Evaluation & Candidate Upgrade
# =============================================================================

class TestQualityEvaluationAndUpgrade:
    """Verifies multi-factor face quality assessment and candidate upgrade transitions."""

    def test_quality_assessor_scoring_factors(self):
        """Sharp, high-resolution crop scores substantially higher than blurred/low-res crop."""
        assessor = FaceQualityAssessor()

        # Crisp, high-contrast face crop with human skin tone
        sharp_crop = np.full((120, 120, 3), (160, 190, 215), dtype=np.uint8)
        cv2.circle(sharp_crop, (40, 40), 10, (255, 255, 255), -1)
        cv2.circle(sharp_crop, (80, 40), 10, (255, 255, 255), -1)
        cv2.line(sharp_crop, (60, 50), (60, 80), (30, 30, 30), 3)

        pose_f = FacePose(yaw=0.0, pitch=0.0, roll=0.0, category="frontal")
        q_sharp = assessor.assess(sharp_crop, pose=pose_f)

        # Severely blurred crop
        blurred_crop = cv2.GaussianBlur(sharp_crop, (31, 31), 15.0)
        q_blurred = assessor.assess(blurred_crop, pose=pose_f)

        assert q_sharp.score > q_blurred.score + 25.0
        assert q_sharp.sharpness > q_blurred.sharpness
        assert q_sharp.is_acceptable is True

    def test_candidate_upgrade_on_high_quality_observation(self):
        """Initial low quality capture is successfully upgraded when a higher quality face is observed."""
        selector = CaptureSelector(CaptureSelectorConfig(quality_delta_threshold=4.0))

        # Observation 1: Initial fair quality (score = 45.0)
        pose = FacePose(yaw=2.0, pitch=1.0, roll=0.0, category="frontal")
        q1 = FaceQualityResult(
            score=45.0, sharpness=25.0, motion_blur=0.35, resolution_score=50.0,
            lighting_score=55.0, pose_score=85.0, size_score=50.0, is_acceptable=True
        )
        obs1 = FaceObservation(
            observation_id="obs_1",
            track_id=1,
            timestamp=1.0,
            bbox=(100, 100, 180, 200),
            confidence=0.88,
            pose=pose,
            quality=q1,
            face_crop=np.full((80, 80, 3), 120, dtype=np.uint8),
        )

        decisions1 = selector.evaluate_observation(obs1, frame_sequence=1)
        overall_d1 = next(d for d in decisions1 if d.category == CATEGORY_OVERALL)
        assert overall_d1.is_selected is True
        assert overall_d1.should_save_disk is True
        assert overall_d1.previous_score is None

        # Observation 2: Upgraded high clarity observation (score = 86.0)
        q2 = FaceQualityResult(
            score=86.0, sharpness=130.0, motion_blur=0.04, resolution_score=90.0,
            lighting_score=88.0, pose_score=92.0, size_score=90.0, is_acceptable=True
        )
        obs2 = FaceObservation(
            observation_id="obs_2",
            track_id=1,
            timestamp=2.0,
            bbox=(100, 100, 220, 240),
            confidence=0.96,
            pose=pose,
            quality=q2,
            face_crop=np.full((120, 120, 3), 160, dtype=np.uint8),
        )

        decisions2 = selector.evaluate_observation(obs2, frame_sequence=2)
        overall_d2 = next(d for d in decisions2 if d.category == CATEGORY_OVERALL)
        assert overall_d2.is_selected is True
        assert overall_d2.should_save_disk is True
        assert overall_d2.quality_delta == pytest.approx(41.0, abs=0.1)
        assert overall_d2.previous_score == pytest.approx(45.0, abs=0.1)

        # Verify selector now holds the upgraded candidate
        captures = selector.get_best_captures(track_id=1)
        assert captures[CATEGORY_OVERALL].quality_score == 86.0

    def test_anti_jitter_threshold_suppresses_redundant_writes(self):
        """Quality changes smaller than delta_threshold (4.0) do not trigger disk writes."""
        selector = CaptureSelector(CaptureSelectorConfig(
            quality_delta_threshold=4.0,
            rate_limit_seconds=0.8,
            quality_jump_threshold=15.0,
        ))

        pose = FacePose(yaw=0.0, pitch=0.0, roll=0.0, category="frontal")
        crop = np.full((100, 100, 3), 140, dtype=np.uint8)

        # Baseline capture (score = 80.0)
        q_base = FaceQualityResult(score=80.0, sharpness=100.0, motion_blur=0.05, resolution_score=80.0,
                                   lighting_score=80.0, pose_score=90.0, size_score=80.0, is_acceptable=True)
        obs_base = FaceObservation(observation_id="base", track_id=2, timestamp=1.0, bbox=(50, 50, 150, 150),
                                   confidence=0.9, pose=pose, quality=q_base, face_crop=crop)
        d_base = selector.evaluate_observation(obs_base, frame_sequence=1)
        assert any(d.should_save_disk for d in d_base)

        # Jitter observation: score moves from 80.0 -> 82.0 (delta = 2.0 < 4.0 threshold)
        q_jitter = FaceQualityResult(score=82.0, sharpness=105.0, motion_blur=0.05, resolution_score=80.0,
                                     lighting_score=80.0, pose_score=90.0, size_score=80.0, is_acceptable=True)
        obs_jitter = FaceObservation(observation_id="jitter", track_id=2, timestamp=1.2, bbox=(50, 50, 150, 150),
                                     confidence=0.9, pose=pose, quality=q_jitter, face_crop=crop)
        d_jitter = selector.evaluate_observation(obs_jitter, frame_sequence=2)

        # Must NOT trigger disk write
        overall_jit = next(d for d in d_jitter if d.category == CATEGORY_OVERALL)
        assert overall_jit.should_save_disk is False


# =============================================================================
# 5. Candidate Replacement Logic Across Categories
# =============================================================================

class TestCategoryReplacementLogic:
    """Verifies capture selection across multiple angle and resolution categories."""

    def test_multi_category_capture_lifecycle(self):
        """Track multi-angle progression: frontal -> left angle -> right angle -> high res."""
        selector = CaptureSelector(CaptureSelectorConfig(buffer_size=5))
        track_id = 100

        # 1. Frontal frame (yaw = 0.0)
        pose_front = FacePose(yaw=0.0, pitch=0.0, roll=0.0, category="frontal")
        q_front = FaceQualityResult(score=75.0, sharpness=80.0, motion_blur=0.1, resolution_score=70.0,
                                    lighting_score=75.0, pose_score=90.0, size_score=70.0, is_acceptable=True)
        obs_front = FaceObservation(observation_id="front", track_id=track_id, timestamp=1.0,
                                    bbox=(100, 100, 180, 200), confidence=0.9, pose=pose_front,
                                    quality=q_front, face_crop=np.full((80, 80, 3), 150, dtype=np.uint8))
        d_front = selector.evaluate_observation(obs_front, frame_sequence=1)
        selected_cats = {d.category for d in d_front if d.is_selected}
        assert CATEGORY_OVERALL in selected_cats
        assert CATEGORY_FRONTAL in selected_cats

        # 2. Left angle frame (yaw = -30.0)
        pose_left = FacePose(yaw=-30.0, pitch=0.0, roll=0.0, category="left_angle")
        q_left = FaceQualityResult(score=72.0, sharpness=75.0, motion_blur=0.1, resolution_score=70.0,
                                   lighting_score=75.0, pose_score=85.0, size_score=70.0, is_acceptable=True)
        obs_left = FaceObservation(observation_id="left", track_id=track_id, timestamp=2.0,
                                   bbox=(100, 100, 180, 200), confidence=0.9, pose=pose_left,
                                   quality=q_left, face_crop=np.full((80, 80, 3), 150, dtype=np.uint8))
        d_left = selector.evaluate_observation(obs_left, frame_sequence=2)
        selected_cats_left = {d.category for d in d_left if d.is_selected}
        assert CATEGORY_LEFT_ANGLE in selected_cats_left

        # 3. Right angle frame (yaw = +32.0)
        pose_right = FacePose(yaw=32.0, pitch=0.0, roll=0.0, category="right_angle")
        q_right = FaceQualityResult(score=73.0, sharpness=76.0, motion_blur=0.1, resolution_score=70.0,
                                    lighting_score=75.0, pose_score=85.0, size_score=70.0, is_acceptable=True)
        obs_right = FaceObservation(observation_id="right", track_id=track_id, timestamp=3.0,
                                    bbox=(100, 100, 180, 200), confidence=0.9, pose=pose_right,
                                    quality=q_right, face_crop=np.full((80, 80, 3), 150, dtype=np.uint8))
        d_right = selector.evaluate_observation(obs_right, frame_sequence=3)
        selected_cats_right = {d.category for d in d_right if d.is_selected}
        assert CATEGORY_RIGHT_ANGLE in selected_cats_right

        # 4. Subject walks closer -> High resolution crop (160x160 pixels)
        q_highres = FaceQualityResult(score=82.0, sharpness=110.0, motion_blur=0.05, resolution_score=95.0,
                                      lighting_score=85.0, pose_score=90.0, size_score=95.0, is_acceptable=True)
        obs_highres = FaceObservation(observation_id="highres", track_id=track_id, timestamp=4.0,
                                      bbox=(100, 100, 260, 260), confidence=0.95, pose=pose_front,
                                      quality=q_highres, face_crop=np.full((160, 160, 3), 180, dtype=np.uint8))
        d_highres = selector.evaluate_observation(obs_highres, frame_sequence=4)
        selected_cats_highres = {d.category for d in d_highres if d.is_selected}
        assert CATEGORY_HIGH_RES in selected_cats_highres

        # Verify all distinct categories are saved for this track
        captures = selector.get_best_captures(track_id)
        assert CATEGORY_OVERALL in captures
        assert CATEGORY_FRONTAL in captures
        assert CATEGORY_LEFT_ANGLE in captures
        assert CATEGORY_RIGHT_ANGLE in captures
        assert CATEGORY_HIGH_RES in captures

    def test_bounded_candidate_buffer_capacity(self):
        """Verify candidate buffer never exceeds max size and retains highest scores."""
        buffer_size = 5
        selector = CaptureSelector(CaptureSelectorConfig(buffer_size=buffer_size))
        track_id = 200

        # Feed 12 observations with variable scores
        scores = [45.0, 70.0, 55.0, 85.0, 60.0, 92.0, 50.0, 88.0, 65.0, 95.0, 78.0, 82.0]
        pose = FacePose(yaw=0.0, pitch=0.0, roll=0.0, category="frontal")

        for seq, score in enumerate(scores, start=1):
            q = FaceQualityResult(score=score, sharpness=score, motion_blur=0.1, resolution_score=score,
                                  lighting_score=score, pose_score=score, size_score=score, is_acceptable=True)
            obs = FaceObservation(observation_id=f"buf_{seq}", track_id=track_id, timestamp=float(seq),
                                  bbox=(100, 100, 180, 200), confidence=0.9, pose=pose, quality=q,
                                  face_crop=np.full((80, 80, 3), 100, dtype=np.uint8))
            selector.evaluate_observation(obs, frame_sequence=seq)

        state = selector._tracks[track_id]
        assert len(state.candidate_buffer) == buffer_size

        # Buffer must hold the top 5 scores: 95.0, 92.0, 88.0, 85.0, 82.0 in descending order
        buffer_scores = [c.quality_score for c in state.candidate_buffer]
        expected_top_5 = sorted(scores, reverse=True)[:buffer_size]
        assert buffer_scores == expected_top_5

    def test_brief_visitor_preservation(self):
        """A person visible for only 2 frames still has their best capture preserved."""
        selector = CaptureSelector(CaptureSelectorConfig())
        track_id = 999

        pose = FacePose(yaw=0.0, pitch=0.0, roll=0.0, category="frontal")
        q1 = FaceQualityResult(score=60.0, sharpness=50.0, motion_blur=0.2, resolution_score=60.0,
                               lighting_score=60.0, pose_score=80.0, size_score=60.0, is_acceptable=True)
        obs1 = FaceObservation(observation_id="v1", track_id=track_id, timestamp=1.0,
                               bbox=(10, 10, 80, 80), confidence=0.85, pose=pose, quality=q1,
                               face_crop=np.full((60, 60, 3), 100, dtype=np.uint8))
        selector.evaluate_observation(obs1, frame_sequence=1)

        q2 = FaceQualityResult(score=78.0, sharpness=90.0, motion_blur=0.1, resolution_score=70.0,
                               lighting_score=75.0, pose_score=85.0, size_score=70.0, is_acceptable=True)
        obs2 = FaceObservation(observation_id="v2", track_id=track_id, timestamp=1.1,
                               bbox=(10, 10, 80, 80), confidence=0.92, pose=pose, quality=q2,
                               face_crop=np.full((70, 70, 3), 120, dtype=np.uint8))
        selector.evaluate_observation(obs2, frame_sequence=2)

        captures = selector.get_best_captures(track_id)
        assert len(captures) >= 1
        assert captures[CATEGORY_OVERALL].quality_score == 78.0


# =============================================================================
# 6. Start / Stop Lifecycle Transitions and Thread Cleanliness
# =============================================================================

class TestLifecycleTransitionsAndThreads:
    """Verifies operational state machine transitions, thread management, and worker cleanup."""

    def test_start_stop_lifecycle_state_transitions(self):
        """Verify lifecycle transitions from STOPPED -> STARTING -> RUNNING -> STOPPING -> STOPPED."""
        controller = AnalysisLifecycleController()
        assert controller.lifecycle_state == AnalysisLifecycleState.STOPPED
        assert controller.is_running is False

        # Start analysis
        started = controller.start_analysis(epoch_id="lifecycle_epoch_01")
        assert started is True
        assert controller.lifecycle_state == AnalysisLifecycleState.RUNNING
        assert controller.is_running is True
        assert controller.current_epoch == "lifecycle_epoch_01"

        # Stop analysis
        stopped = controller.stop_analysis(drain_timeout=1.0)
        assert stopped is True
        assert controller.lifecycle_state == AnalysisLifecycleState.STOPPED
        assert controller.is_running is False

    def test_idempotent_lifecycle_operations(self):
        """Calling start or stop repeatedly produces safe no-ops without corrupting state."""
        controller = AnalysisLifecycleController()

        controller.start_analysis(epoch_id="idem_epoch")
        assert controller.is_running is True

        # Second start call is a safe no-op
        second_start = controller.start_analysis(epoch_id="idem_epoch")
        assert second_start is False
        assert controller.is_running is True

        # Stop call
        controller.stop_analysis()
        assert controller.is_running is False

        # Second stop call is a safe no-op
        second_stop = controller.stop_analysis()
        assert second_stop is False
        assert controller.is_running is False

    def test_thread_cleanliness_and_worker_termination(self):
        """Registered background worker threads cleanly join and terminate upon stopping."""
        controller = AnalysisLifecycleController()
        worker_ran = threading.Event()

        def background_task(stop_event: threading.Event):
            while not stop_event.is_set():
                worker_ran.set()
                time.sleep(0.01)

        controller.register_worker(
            name="integration_test_worker",
            target=background_task,
            pass_stop_event=True,
            daemon=True,
        )

        controller.start_analysis()
        assert worker_ran.wait(timeout=1.0) is True
        assert "integration_test_worker" in controller._active_threads
        assert controller._active_threads["integration_test_worker"].is_alive()

        # Stop analysis and confirm the worker thread is terminated
        controller.stop_analysis(drain_timeout=1.0)
        assert len(controller._active_threads) == 0

    def test_safe_queue_draining_on_stop(self):
        """Pending disk capture write operations are drained safely prior to shutdown."""
        drained_items = []

        def mock_writer(item: Any):
            drained_items.append(item)

        controller = AnalysisLifecycleController(capture_writer=mock_writer)
        controller.start_analysis()

        # Enqueue 5 items
        for i in range(5):
            controller.enqueue_capture_write(f"capture_{i}")

        # Stop should drain the queue
        controller.stop_analysis(drain_timeout=2.0)
        assert len(drained_items) == 5
        assert drained_items == [f"capture_{i}" for i in range(5)]

    def test_restart_analysis_and_worker_error_recovery(self):
        """Restart cycles analysis cleanly and worker errors are captured."""
        controller = AnalysisLifecycleController()
        controller.start_analysis(epoch_id="first_epoch")
        assert controller.is_running is True
        assert controller.current_epoch == "first_epoch"

        # Atomic restart
        restarted = controller.restart_analysis(epoch_id="second_epoch")
        assert restarted is True
        assert controller.is_running is True
        assert controller.current_epoch == "second_epoch"

        # Worker error reporting
        controller.report_worker_error("sample_worker", RuntimeError("simulated error"))
        assert controller.lifecycle_state == AnalysisLifecycleState.ERROR
        assert "simulated error" in (controller.state.error or "")

        controller.stop_analysis()


# =============================================================================
# 7. Safe Profile Retrieval and Original Image Extraction
# =============================================================================

class TestProfileRetrievalAndStorage:
    """Verifies SQLite repository persistence, safe image extraction, and security."""

    @pytest.fixture
    def test_repo(self, tmp_path):
        """Construct a clean isolated ProfileRepository instance."""
        db_file = tmp_path / "traffic.sqlite3"
        storage_dir = tmp_path / "captures"
        repo = ProfileRepository(db_path=db_file, storage_dir=storage_dir)
        yield repo
        repo.close()

    def test_atomic_profile_and_captures_persistence(self, test_repo):
        """Capture candidate is persisted atomically with original crop into database and filesystem."""
        crop = np.zeros((80, 80, 3), dtype=np.uint8)
        cv2.circle(crop, (40, 40), 20, (0, 255, 128), -1)

        pose = FacePose(yaw=5.0, pitch=2.0, roll=0.0, category="frontal")
        candidate = CaptureCandidate(
            candidate_id="cand_test_01",
            track_id=15,
            timestamp=100.0,
            category=CATEGORY_FRONTAL,
            quality_score=85.5,
            pose=pose,
            crop_original=crop,
            frame_sequence=10,
            bbox_in_frame=(100, 100, 180, 180),
        )

        record = test_repo.save_candidate("session_cam1:15", candidate)
        assert record.capture_id == "cand_test_01"
        assert record.quality_score == 85.5
        assert Path(record.original_path).exists()

        # Retrieve profile
        profile = test_repo.get_profile("session_cam1:15")
        assert profile is not None
        assert profile.track_id == 15
        assert profile.best_capture is not None
        assert profile.best_capture.capture_id == "cand_test_01"
        assert profile.best_quality_score == 85.5

    def test_safe_image_extraction_array_and_bytes(self, test_repo):
        """Image retrieval via get_capture_image_array and get_capture_image_bytes."""
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        crop[15:45, 15:45] = (255, 180, 50)

        pose = FacePose(yaw=0.0, pitch=0.0, roll=0.0, category="frontal")
        candidate = CaptureCandidate(
            candidate_id="cand_extract_01",
            track_id=22,
            timestamp=200.0,
            category=CATEGORY_OVERALL,
            quality_score=89.0,
            pose=pose,
            crop_original=crop,
            frame_sequence=12,
            bbox_in_frame=(50, 50, 114, 114),
        )

        record = test_repo.save_candidate("session_extract:22", candidate)

        # 1. Image path
        img_path = test_repo.get_capture_image_path(record.capture_id)
        assert img_path is not None
        assert img_path.is_file()

        # 2. Image bytes
        img_bytes = test_repo.get_capture_image_bytes(record.capture_id)
        assert img_bytes is not None
        assert len(img_bytes) > 0
        assert img_bytes[:8] == b"\x89PNG\r\n\x1a\n"  # Lossless original, no JPEG re-compression

        # 3. Decoded numpy array
        extracted_arr = test_repo.get_capture_image_array(record.capture_id)
        assert extracted_arr is not None
        assert extracted_arr.shape == (64, 64, 3)

    def test_path_traversal_attack_prevention(self, test_repo):
        """Validate that path traversal sequences (../../) are blocked and rejected."""
        malicious_filename = "../../evil.jpg"
        test_img = np.zeros((10, 10, 3), dtype=np.uint8)

        # write_image_atomic sanitizes or raises on traversal
        with pytest.raises(ValueError):
            test_repo.validate_safe_path(Path("C:/Windows/System32/calc.exe"))

        with pytest.raises(ValueError):
            test_repo.validate_safe_path(test_repo.storage_dir / ".." / "outside.txt")

    def test_profile_filtering_sorting_and_pagination(self, test_repo):
        """Verify repository listing, sorting by quality, and search query filters."""
        # Create 4 profiles with distinct quality scores and timestamps
        for i, score in enumerate([50.0, 92.0, 75.0, 88.0], start=1):
            profile = PersonProfile(
                session_id=f"prof_{i}",
                track_id=i,
                camera_id="cam_main",
                first_seen=float(i * 10),
                last_seen=float(i * 10 + 5),
                duration=5.0,
                status="active" if i % 2 == 0 else "lost",
                best_quality_score=score,
            )
            test_repo.save_profile(profile)

        # Sort by quality desc
        sorted_by_q = test_repo.list_profiles(sort_by="quality", order="desc")
        assert len(sorted_by_q) == 4
        assert [p.best_quality_score for p in sorted_by_q] == [92.0, 88.0, 75.0, 50.0]

        # Filter by min_quality
        high_q = test_repo.list_profiles(min_quality=80.0)
        assert len(high_q) == 2
        assert {p.track_id for p in high_q} == {2, 4}


# =============================================================================
# 8. Graceful Recovery Under Missing Models or Frame Drops
# =============================================================================

class TestReliabilityAndFaultTolerance:
    """Verifies robustness against missing models, sequence skips, and frame corruptions."""

    def test_missing_yunet_model_fallback_to_haar(self):
        """FaceDetector seamlessly falls back to Haar Cascade if YuNet ONNX model is missing."""
        config = FaceDetectorConfig(
            model_path="non_existent_path_yunet_weights.onnx",
            backend="auto",
        )
        detector = FaceDetector(config)

        assert detector.active_backend == "haar"
        assert detector._haar is not None

        # Verify detector runs without throwing exceptions
        dummy_frame = create_synthetic_canvas(640, 480)
        obs = detector.detect(dummy_frame)
        assert isinstance(obs, list)

    def test_synthetic_frame_drops_and_sequence_gaps(self):
        """Tracker recovers state cleanly across large sequence number jumps and time skips."""
        tracker = PersonTracker(session_id="frame_drop_session", lifetime=3.0)

        # Confirm track over 2 frames so it acquires full lifetime
        det1 = PersonDetection(bbox=(100.0, 100.0, 180.0, 300.0), confidence=0.9)
        tracker.update([det1], timestamp=0.0)
        det2 = PersonDetection(bbox=(105.0, 100.0, 185.0, 300.0), confidence=0.9)
        active2 = tracker.update([det2], timestamp=0.1)
        assert len(active2) == 1
        assert active2[0].confirmed is True
        t_id = active2[0].track_id

        # Jump to Frame 15 at t=0.6 (dropped frames)
        # Person moved smoothly to (130, 100)
        det15 = PersonDetection(bbox=(130.0, 100.0, 210.0, 300.0), confidence=0.92)
        active15 = tracker.update([det15], timestamp=0.6)

        # Track ID must be retained
        assert len(active15) == 1
        assert active15[0].track_id == t_id
        assert active15[0].hits == 3

    def test_corrupt_and_empty_frames_handled_gracefully(self):
        """Empty or invalid frames do not raise uncaught exceptions."""
        detector = FaceDetector(FaceDetectorConfig(backend="haar"))

        # 1. None frame
        obs_none = detector.detect(None)
        assert obs_none == []

        # 2. Empty 0x0 frame
        empty_img = np.empty((0, 0, 3), dtype=np.uint8)
        obs_empty = detector.detect(empty_img)
        assert obs_empty == []

        # 3. 1x1 miniature image
        mini_img = np.zeros((1, 1, 3), dtype=np.uint8)
        obs_mini = detector.detect(mini_img)
        assert obs_mini == []

    def test_occlusion_recovery_within_lifetime(self):
        """A person temporarily occluded (unobserved) recovers their identity upon reappearing."""
        tracker = PersonTracker(session_id="occlusion_session", lifetime=2.0)

        # Step 1: Initial observation (Frames 1-3)
        for i in range(3):
            d = PersonDetection(bbox=(200.0, 150.0, 280.0, 350.0), confidence=0.9)
            active = tracker.update([d], timestamp=i * 0.1)
        assert active[0].confirmed is True
        tid = active[0].track_id

        # Step 2: Occlusion (Frames 4-7: person is unobserved for 0.4 seconds)
        for i in range(3, 7):
            active_lost = tracker.update([], timestamp=i * 0.1)
            assert len(active_lost) == 0

        # Step 3: Reappearance (Frame 8 at t=0.8, well within lifetime=2.0)
        reappear_det = PersonDetection(bbox=(210.0, 150.0, 290.0, 350.0), confidence=0.9)
        active_recovered = tracker.update([reappear_det], timestamp=0.8)

        assert len(active_recovered) == 1
        assert active_recovered[0].track_id == tid


# =============================================================================
# 9. Full End-to-End Pipeline Integration Workflow
# =============================================================================

class TestEndToEndPipelineIntegration:
    """Verifies complete end-to-end multi-person facial capture pipeline workflow."""

    def test_end_to_end_multi_person_capture_workflow(self, tmp_path):
        """Simulate complete 10-frame capture pipeline with 5 moving people and persistence."""
        db_file = tmp_path / "traffic_e2e.sqlite3"
        storage_dir = tmp_path / "captures_e2e"

        # 1. Initialize subsystems
        tracker = PersonTracker(session_id="e2e_session", lifetime=2.5)
        detector = FaceDetector(FaceDetectorConfig(backend="yunet", extract_face_crops=True, score_threshold=0.3))
        assessor = FaceQualityAssessor()
        selector = CaptureSelector(CaptureSelectorConfig(buffer_size=5, quality_delta_threshold=3.0))
        repo = ProfileRepository(db_path=db_file, storage_dir=storage_dir)

        controller = AnalysisLifecycleController(
            tracker=tracker,
            detector=detector,
            profile_storage=repo,
        )

        started = controller.start_analysis(epoch_id="e2e_epoch_01")
        assert started is True

        # 2. Simulate 5 people across 10 frames
        num_people = 5
        people_specs = [
            {"cx": 180 + i * 220, "cy": 180, "vx": 4.0 * (1 if i % 2 == 0 else -1), "yaw": (i - 2) * 15.0}
            for i in range(num_people)
        ]

        for frame_idx in range(10):
            timestamp = 100.0 + frame_idx * 0.1
            canvas = create_synthetic_canvas(1280, 720)

            # Render people into full-resolution canvas
            detections = []
            for spec in people_specs:
                cur_x = spec["cx"] + spec["vx"] * frame_idx
                cur_y = spec["cy"]
                # Gradually turn head to generate angle variation
                cur_yaw = spec["yaw"] + frame_idx * 2.0

                p_box, f_box = render_synthetic_person(
                    canvas, int(cur_x), int(cur_y), scale=1.0, yaw=cur_yaw
                )
                detections.append(PersonDetection(bbox=p_box, confidence=0.92))

            # Packet encapsulation
            packet = FramePacket(
                sequence=frame_idx,
                timestamp=timestamp,
                image=canvas,
                source_id="camera_main",
                epoch=controller.current_epoch,
                width=1280,
                height=720,
            )

            # Step A: Update multi-person tracker
            active_tracks = tracker.update(detections, timestamp=packet.timestamp)
            assert len(active_tracks) == num_people

            # Step B: Detect and associate faces to tracks
            observations = detector.detect(
                packet,
                tracks=active_tracks,
                timestamp=packet.timestamp,
                frame_sequence=packet.sequence,
            )

            # Step C: Evaluate pose and quality for each observation
            for obs in observations:
                if obs.landmarks:
                    obs.pose = estimate_face_pose(obs.landmarks, bbox=obs.bbox)

                if obs.face_crop is None and packet.image is not None:
                    fh, fw = packet.image.shape[:2]
                    cy1 = max(0, min(fh, int(obs.bbox[1])))
                    cy2 = max(0, min(fh, int(obs.bbox[3])))
                    cx1 = max(0, min(fw, int(obs.bbox[0])))
                    cx2 = max(0, min(fw, int(obs.bbox[2])))
                    if cy2 > cy1 and cx2 > cx1:
                        obs.face_crop = packet.image[cy1:cy2, cx1:cx2].copy()

                if obs.face_crop is not None:
                    obs.quality = assessor.assess(
                        obs.face_crop, pose=obs.pose, landmarks=obs.landmarks, bbox=obs.bbox
                    )

                # Step D: Capture Selection
                decisions = selector.evaluate_observation(
                    obs, frame_sequence=packet.sequence, frame_image=packet.image
                )

                # Step E: Persist selected captures to ProfileRepository
                for dec in decisions:
                    if dec.should_save_disk:
                        session_id = f"cam_main:{dec.track_id}"
                        repo.save_candidate(session_id, dec.candidate)

        # 3. Gracefully stop analysis
        controller.stop_analysis(drain_timeout=2.0)
        assert controller.is_running is False

        # 4. Verify persisted profiles in repository
        profiles = repo.list_profiles(include_captures=True)
        assert len(profiles) >= num_people

        for profile in profiles:
            assert profile.best_capture is not None
            assert profile.best_quality_score > 0.0

            # Verify image file extraction on disk
            best_cap_id = profile.best_capture.capture_id
            img_arr = repo.get_capture_image_array(best_cap_id)
            assert img_arr is not None
            assert img_arr.ndim == 3
            assert img_arr.shape[0] > 0 and img_arr.shape[1] > 0

        repo.close()
