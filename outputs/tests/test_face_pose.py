"""Unit tests for facial landmarks geometry and head pose estimation.

Verifies:
  - Frontal landmark configuration produces "frontal" category and yaw near 0
  - Left-turned landmarks produce "left_angle" category with negative yaw
  - Right-turned landmarks produce "right_angle" category with positive yaw
  - Extreme angle produces "profile"
  - Robust fallback when landmarks are partial or degenerate
  - Conformance to services.face_contracts.FacePose
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from services.face_contracts import FaceObservation, FacePose
from services.face_pose import (
    CANONICAL_3D_FACE_MODEL,
    FRONTAL_PITCH_THRESHOLD,
    FRONTAL_YAW_THRESHOLD,
    PROFILE_YAW_THRESHOLD,
    FacePoseEstimator,
    categorize_orientation,
    estimate_face_pose,
    estimate_pose_geometric,
    estimate_pose_pnp,
    parse_landmarks,
    project_reference_model,
    validate_face_landmarks,
)


@pytest.fixture
def frontal_landmarks():
    """Standard symmetric frontal face landmarks."""
    return [
        (40.0, 40.0),  # Right eye
        (70.0, 40.0),  # Left eye
        (55.0, 55.0),  # Nose tip (centered between 40 and 70)
        (45.0, 68.0),  # Right mouth corner
        (65.0, 68.0),  # Left mouth corner
    ]


@pytest.fixture
def left_turned_landmarks():
    """Face turned left: nose shifted left towards right eye in 2D projection."""
    return [
        (40.0, 40.0),  # Right eye
        (70.0, 40.0),  # Left eye
        (47.0, 55.0),  # Nose tip moved left
        (45.0, 68.0),  # Right mouth corner
        (65.0, 68.0),  # Left mouth corner
    ]


@pytest.fixture
def right_turned_landmarks():
    """Face turned right: nose shifted right towards left eye in 2D projection."""
    return [
        (40.0, 40.0),  # Right eye
        (70.0, 40.0),  # Left eye
        (63.0, 55.0),  # Nose tip moved right
        (45.0, 68.0),  # Right mouth corner
        (65.0, 68.0),  # Left mouth corner
    ]


@pytest.fixture
def profile_left_landmarks():
    """Face turned to extreme left angle (profile)."""
    return [
        (40.0, 40.0),
        (70.0, 40.0),
        (32.0, 55.0),  # Extreme left
        (45.0, 68.0),
        (65.0, 68.0),
    ]


@pytest.fixture
def profile_right_landmarks():
    """Face turned to extreme right angle (profile)."""
    return [
        (40.0, 40.0),
        (70.0, 40.0),
        (78.0, 55.0),  # Extreme right
        (45.0, 68.0),
        (65.0, 68.0),
    ]


# ---------------------------------------------------------------------------
# Requirement 6.1: Frontal landmark configuration
# ---------------------------------------------------------------------------

def test_frontal_landmark_configuration(frontal_landmarks):
    """Verify frontal configuration produces 'frontal' category and yaw near 0."""
    pose = estimate_face_pose(frontal_landmarks)

    assert isinstance(pose, FacePose)
    assert pose.category == "frontal"
    assert abs(pose.yaw) <= FRONTAL_YAW_THRESHOLD
    assert abs(pose.yaw) < 5.0  # Near zero
    assert abs(pose.pitch) <= FRONTAL_PITCH_THRESHOLD
    assert pose.confidence > 0.8

    # Serialization check conforming to face_contracts
    d = pose.to_dict()
    assert "yaw" in d and isinstance(d["yaw"], float)
    assert "pitch" in d and isinstance(d["pitch"], float)
    assert "roll" in d and isinstance(d["roll"], float)
    assert d["category"] == "frontal"
    assert "confidence" in d


def test_frontal_projected_model():
    """Verify ground-truth 3D projected frontal model gives near 0 pose."""
    pts = project_reference_model(yaw=0.0, pitch=0.0, roll=0.0)
    pose = estimate_face_pose(pts, method="pnp")

    assert pose.category == "frontal"
    assert abs(pose.yaw) < 1.0
    assert abs(pose.pitch) < 1.0
    assert abs(pose.roll) < 1.0
    assert pose.confidence >= 0.95


# ---------------------------------------------------------------------------
# Requirement 6.2: Left-turned landmarks
# ---------------------------------------------------------------------------

def test_left_turned_landmarks(left_turned_landmarks):
    """Verify left-turned landmarks produce 'left_angle' category with negative yaw."""
    pose = estimate_face_pose(left_turned_landmarks)

    assert isinstance(pose, FacePose)
    assert pose.category == "left_angle"
    assert pose.yaw < 0.0  # Negative left
    assert -45.0 <= pose.yaw < -15.0
    assert pose.confidence > 0.8


def test_left_turned_projected_angles():
    """Verify multiple left-angle orientations produced by 3D projection."""
    for true_yaw in [-20.0, -30.0, -40.0]:
        pts = project_reference_model(yaw=true_yaw)
        pose = estimate_face_pose(pts, method="pnp")

        assert pose.category == "left_angle"
        assert pose.yaw < 0.0
        assert -45.0 <= pose.yaw < -15.0


# ---------------------------------------------------------------------------
# Requirement 6.3: Right-turned landmarks
# ---------------------------------------------------------------------------

def test_right_turned_landmarks(right_turned_landmarks):
    """Verify right-turned landmarks produce 'right_angle' category with positive yaw."""
    pose = estimate_face_pose(right_turned_landmarks)

    assert isinstance(pose, FacePose)
    assert pose.category == "right_angle"
    assert pose.yaw > 0.0  # Positive right
    assert 15.0 < pose.yaw <= 45.0
    assert pose.confidence > 0.8


def test_right_turned_projected_angles():
    """Verify multiple right-angle orientations produced by 3D projection."""
    for true_yaw in [20.0, 30.0, 40.0]:
        pts = project_reference_model(yaw=true_yaw)
        pose = estimate_face_pose(pts, method="pnp")

        assert pose.category == "right_angle"
        assert pose.yaw > 0.0
        assert 15.0 < pose.yaw <= 45.0


# ---------------------------------------------------------------------------
# Requirement 6.4: Extreme angle produces "profile"
# ---------------------------------------------------------------------------

def test_extreme_angle_profile_left(profile_left_landmarks):
    """Verify extreme left angle produces 'profile' category."""
    pose = estimate_face_pose(profile_left_landmarks)

    assert isinstance(pose, FacePose)
    assert pose.category == "profile"
    assert abs(pose.yaw) > PROFILE_YAW_THRESHOLD
    assert pose.yaw < -45.0


def test_extreme_angle_profile_right(profile_right_landmarks):
    """Verify extreme right angle produces 'profile' category."""
    pose = estimate_face_pose(profile_right_landmarks)

    assert isinstance(pose, FacePose)
    assert pose.category == "profile"
    assert abs(pose.yaw) > PROFILE_YAW_THRESHOLD
    assert pose.yaw > 45.0


def test_projected_profile_angles():
    """Verify 3D projected profile angles (yaw > 45 deg) classify as profile."""
    for extreme_yaw in [-55.0, -65.0, 55.0, 65.0]:
        pts = project_reference_model(yaw=extreme_yaw)
        pose = estimate_face_pose(pts, method="pnp")

        assert pose.category == "profile"
        assert abs(pose.yaw) > PROFILE_YAW_THRESHOLD


# ---------------------------------------------------------------------------
# Requirement 6.5: Robust fallback when landmarks are partial or degenerate
# ---------------------------------------------------------------------------

def test_none_landmarks():
    """Verify None landmarks return uncertain FacePose without throwing."""
    pose = estimate_face_pose(None)
    assert isinstance(pose, FacePose)
    assert pose.category == "uncertain"
    assert pose.confidence == 0.0


def test_empty_landmarks():
    """Verify empty landmarks list returns uncertain FacePose."""
    pose = estimate_face_pose([])
    assert isinstance(pose, FacePose)
    assert pose.category == "uncertain"
    assert pose.confidence == 0.0


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_partial_landmarks(count):
    """Verify partial landmark lists (< 5 points) return uncertain FacePose."""
    pts = [(40.0, 40.0), (70.0, 40.0), (55.0, 55.0), (45.0, 68.0)][:count]
    pose = estimate_face_pose(pts)
    assert pose.category == "uncertain"
    assert pose.confidence == 0.0


def test_degenerate_identical_points():
    """Verify identical collapsed landmarks (eye distance 0) return uncertain."""
    degenerate_pts = [(50.0, 50.0)] * 5
    pose = estimate_face_pose(degenerate_pts)
    assert pose.category == "uncertain"
    assert pose.confidence == 0.0


def test_eye_distance_too_close():
    """Verify points with eye distance smaller than threshold return uncertain."""
    close_pts = [
        (50.0, 50.0),
        (51.0, 50.0),  # Eye dist = 1.0 px < MIN_EYE_DISTANCE_PX
        (50.5, 52.0),
        (50.0, 55.0),
        (51.0, 55.0),
    ]
    pose = estimate_face_pose(close_pts)
    assert pose.category == "uncertain"
    assert pose.confidence == 0.0


def test_nan_and_inf_coordinates():
    """Verify landmarks containing NaN or Inf gracefully return uncertain."""
    nan_pts = [
        (float("nan"), 40.0),
        (70.0, 40.0),
        (55.0, 55.0),
        (45.0, 68.0),
        (65.0, 68.0),
    ]
    pose_nan = estimate_face_pose(nan_pts)
    assert pose_nan.category == "uncertain"
    assert pose_nan.confidence == 0.0

    inf_pts = [
        (40.0, 40.0),
        (float("inf"), 40.0),
        (55.0, 55.0),
        (45.0, 68.0),
        (65.0, 68.0),
    ]
    pose_inf = estimate_face_pose(inf_pts)
    assert pose_inf.category == "uncertain"
    assert pose_inf.confidence == 0.0


def test_inverted_eye_order():
    """Verify inverted eye order (right eye placed to the right of left eye) is rejected."""
    inverted_pts = [
        (70.0, 40.0),  # Inverted: right eye at x=70
        (40.0, 40.0),  # Inverted: left eye at x=40
        (55.0, 55.0),
        (45.0, 68.0),
        (65.0, 68.0),
    ]
    pose = estimate_face_pose(inverted_pts)
    assert pose.category == "uncertain"
    assert pose.confidence == 0.0


def test_mouth_above_eyes():
    """Verify landmarks with mouth placed above eyes are rejected."""
    inverted_y_pts = [
        (40.0, 70.0),  # Eyes at y=70
        (70.0, 70.0),
        (55.0, 55.0),
        (45.0, 30.0),  # Mouth at y=30 (above eyes)
        (65.0, 30.0),
    ]
    pose = estimate_face_pose(inverted_y_pts)
    assert pose.category == "uncertain"
    assert pose.confidence == 0.0


def test_landmarks_outside_bounding_box(frontal_landmarks):
    """Verify landmarks located far outside given bounding box return uncertain."""
    # Bounding box located at (200, 200, 300, 300), landmarks are at x~40-70, y~40-68
    bbox_distant = (200.0, 200.0, 300.0, 300.0)
    pose = estimate_face_pose(frontal_landmarks, bbox=bbox_distant)
    assert pose.category == "uncertain"
    assert pose.confidence == 0.0


def test_low_landmark_confidence(frontal_landmarks):
    """Verify that landmark confidence below threshold produces 'uncertain' category."""
    pose = estimate_face_pose(frontal_landmarks, confidence=0.15)
    assert pose.category == "uncertain"
    assert pose.confidence < 0.3


# ---------------------------------------------------------------------------
# Pitch and Roll evaluation
# ---------------------------------------------------------------------------

def test_pitch_up_and_down():
    """Verify head pitch up produces positive pitch and pitch down produces negative pitch."""
    # Head pitched up (nose closer to eyes in 2D)
    pts_up = project_reference_model(pitch=22.0)
    pose_up = estimate_face_pose(pts_up, method="pnp")
    assert pose_up.pitch > 10.0

    # Head pitched down (nose closer to mouth in 2D)
    pts_down = project_reference_model(pitch=-22.0)
    pose_down = estimate_face_pose(pts_down, method="pnp")
    assert pose_down.pitch < -10.0


def test_pitch_frontal_boundaries(frontal_landmarks):
    """Verify extreme pitch outside frontal limits categorizes as uncertain."""
    # Yaw is frontal (0), but pitch is extreme up
    pts_high_pitch = project_reference_model(yaw=0.0, pitch=25.0)
    pose_high = estimate_face_pose(pts_high_pitch, method="pnp")
    assert pose_high.pitch > FRONTAL_PITCH_THRESHOLD
    # Since pitch > 15, cannot be classified as "frontal"
    assert pose_high.category == "uncertain"


def test_roll_tilt():
    """Verify clockwise and counter-clockwise head tilts produce corresponding roll."""
    pts_cw = project_reference_model(roll=20.0)
    pose_cw = estimate_face_pose(pts_cw, method="pnp")
    assert pose_cw.roll > 10.0

    pts_ccw = project_reference_model(roll=-20.0)
    pose_ccw = estimate_face_pose(pts_ccw, method="pnp")
    assert pose_ccw.roll < -10.0


# ---------------------------------------------------------------------------
# Method Comparison: PnP vs Geometric ratio estimation
# ---------------------------------------------------------------------------

def test_both_methods_agreement(frontal_landmarks, left_turned_landmarks, right_turned_landmarks):
    """Verify both PnP and Geometric methods produce consistent orientation categories."""
    for pts, expected_cat in [
        (frontal_landmarks, "frontal"),
        (left_turned_landmarks, "left_angle"),
        (right_turned_landmarks, "right_angle"),
    ]:
        pose_pnp = estimate_face_pose(pts, method="pnp")
        pose_geom = estimate_face_pose(pts, method="geometric")
        pose_auto = estimate_face_pose(pts, method="auto")

        assert pose_pnp.category == expected_cat
        assert pose_geom.category == expected_cat
        assert pose_auto.category == expected_cat


# ---------------------------------------------------------------------------
# Input Format Flexibility
# ---------------------------------------------------------------------------

def test_numpy_array_input(frontal_landmarks):
    """Verify numpy 2D array of shape (5, 2) is accepted."""
    arr = np.array(frontal_landmarks, dtype=np.float32)
    pose = estimate_face_pose(arr)
    assert pose.category == "frontal"


def test_numpy_array_with_confidences(frontal_landmarks):
    """Verify numpy array of shape (5, 3) with confidences is parsed correctly."""
    arr = np.zeros((5, 3), dtype=np.float64)
    arr[:, :2] = frontal_landmarks
    arr[:, 2] = [0.95, 0.94, 0.96, 0.92, 0.93]

    pose = estimate_face_pose(arr)
    assert pose.category == "frontal"
    assert pose.confidence > 0.85


def test_flattened_array_input(frontal_landmarks):
    """Verify flat array of length 10 is parsed correctly."""
    flat = np.array(frontal_landmarks).flatten()
    assert flat.size == 10
    pose = estimate_face_pose(flat)
    assert pose.category == "frontal"


def test_dictionary_input():
    """Verify dictionary with named landmark keys is accepted."""
    d = {
        "right_eye": (40.0, 40.0),
        "left_eye": (70.0, 40.0),
        "nose": (55.0, 55.0),
        "right_mouth": (45.0, 68.0),
        "left_mouth": (65.0, 68.0),
    }
    pose = estimate_face_pose(d)
    assert pose.category == "frontal"


def test_normalized_coordinates():
    """Verify normalized [0, 1] coordinates are handled gracefully."""
    norm_pts = [
        (0.40, 0.40),
        (0.70, 0.40),
        (0.55, 0.55),
        (0.45, 0.68),
        (0.65, 0.68),
    ]
    pose = estimate_face_pose(norm_pts, image_size=(640, 480))
    assert pose.category == "frontal"


def test_face_observation_integration(frontal_landmarks):
    """Verify integration with FaceObservation instances."""
    obs = FaceObservation(
        observation_id="obs_test_1",
        track_id=1,
        timestamp=100.0,
        bbox=(30.0, 25.0, 80.0, 75.0),
        confidence=0.95,
        landmarks=frontal_landmarks,
    )
    estimator = FacePoseEstimator()
    pose = estimator.estimate_observation(obs)

    assert obs.pose is pose
    assert obs.pose.category == "frontal"
    d_obs = obs.to_dict()
    assert d_obs["pose"]["category"] == "frontal"


# ---------------------------------------------------------------------------
# Strict Categorization Boundary Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "yaw, pitch, expected",
    [
        (0.0, 0.0, "frontal"),
        (15.0, 0.0, "frontal"),
        (-15.0, 0.0, "frontal"),
        (0.0, 15.0, "frontal"),
        (0.0, -15.0, "frontal"),
        (15.1, 0.0, "right_angle"),
        (45.0, 0.0, "right_angle"),
        (-15.1, 0.0, "left_angle"),
        (-45.0, 0.0, "left_angle"),
        (45.1, 0.0, "profile"),
        (-45.1, 0.0, "profile"),
        (0.0, 15.1, "uncertain"),
        (0.0, -15.1, "uncertain"),
    ],
)
def test_categorize_orientation_boundaries(yaw, pitch, expected):
    """Verify strict mathematical boundaries of orientation categories."""
    cat = categorize_orientation(yaw=yaw, pitch=pitch, confidence=1.0, is_reliable=True)
    assert cat == expected
