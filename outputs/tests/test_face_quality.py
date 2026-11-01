"""Unit tests for the multi-factor face quality assessment engine."""
import cv2
import numpy as np
import pytest

from services.face_contracts import FaceObservation, FacePose, FaceQualityResult
from services.face_quality import (
    FaceQualityAssessor,
    QualityThresholds,
    QualityWeights,
    assess_face_quality,
    get_quality_tier,
)


def make_synthetic_face(size: int = 128, bg_lum: int = 135) -> np.ndarray:
    """Generate a realistic synthetic face crop with eyes, nose, and mouth."""
    img = np.full((size, size, 3), bg_lum, dtype=np.uint8)
    cx, cy = size // 2, size // 2
    r_face = int(size * 0.38)
    # Face oval / skin
    cv2.circle(img, (cx, cy), r_face, (160, 190, 215), -1)

    # Eyes with pupils and sclera
    eye_r = max(2, int(size * 0.055))
    eye_y = int(size * 0.42)
    eye_dx = int(size * 0.16)
    cv2.circle(img, (cx - eye_dx, eye_y), eye_r + 1, (240, 240, 240), -1)
    cv2.circle(img, (cx + eye_dx, eye_y), eye_r + 1, (240, 240, 240), -1)
    cv2.circle(img, (cx - eye_dx, eye_y), eye_r - 1, (30, 25, 20), -1)
    cv2.circle(img, (cx + eye_dx, eye_y), eye_r - 1, (30, 25, 20), -1)

    # Eyebrows
    brow_y = eye_y - eye_r - 2
    brow_w = int(size * 0.08)
    cv2.line(img, (cx - eye_dx - brow_w, brow_y), (cx - eye_dx + brow_w, brow_y), (40, 30, 20), max(1, size // 50))
    cv2.line(img, (cx + eye_dx - brow_w, brow_y), (cx + eye_dx + brow_w, brow_y), (40, 30, 20), max(1, size // 50))

    # Nose
    nose_y = int(size * 0.56)
    cv2.line(img, (cx, eye_y + eye_r), (cx, nose_y), (110, 130, 150), max(1, size // 40))

    # Mouth (lips)
    mouth_y = int(size * 0.72)
    mouth_w = int(size * 0.15)
    cv2.line(img, (cx - mouth_w, mouth_y), (cx + mouth_w, mouth_y), (60, 50, 160), max(1, size // 40))
    return img


def test_sharp_clear_scores_substantially_higher_than_blurred():
    """Verify sharp clear face scores substantially higher than blurred face."""
    assessor = FaceQualityAssessor()
    sharp_crop = make_synthetic_face(128)

    # Apply significant Gaussian blur
    blurred_crop = cv2.GaussianBlur(sharp_crop, (15, 15), 5.0)

    res_sharp = assessor.assess(sharp_crop)
    res_blurred = assessor.assess(blurred_crop)

    # Sharp face must score in excellent or good tier (>= 80.0)
    assert res_sharp.score >= 80.0, f"Expected sharp score >= 80, got {res_sharp.score}"
    assert res_sharp.is_acceptable is True
    assert res_sharp.sharpness >= 50.0

    # Blurred face must score poorly (< 35.0) and fail acceptance
    assert res_blurred.score < 35.0, f"Expected blurred score < 35, got {res_blurred.score}"
    assert res_blurred.is_acceptable is False
    assert res_blurred.sharpness < 15.0

    # Margin between sharp and blurred must be substantial (> 45 points)
    assert (res_sharp.score - res_blurred.score) > 45.0


def test_sharp_clear_scores_substantially_higher_than_downscaled():
    """Verify high resolution face scores substantially higher than tiny/downscaled face."""
    assessor = FaceQualityAssessor()
    sharp_crop = make_synthetic_face(128)
    downscaled_crop = cv2.resize(sharp_crop, (20, 20))

    res_sharp = assessor.assess(sharp_crop)
    res_down = assessor.assess(downscaled_crop)

    assert res_sharp.score >= 80.0
    assert res_down.score < 35.0
    assert res_down.resolution_score < 20.0
    assert res_down.size_score < 20.0
    assert res_down.is_acceptable is False
    assert (res_sharp.score - res_down.score) > 45.0


def test_well_exposed_scores_higher_than_pitch_black_and_washed_white():
    """Verify well-exposed face scores higher than pitch-black or washed-out white image."""
    assessor = FaceQualityAssessor()
    normal_face = make_synthetic_face(128, bg_lum=135)
    pitch_black = np.zeros((128, 128, 3), dtype=np.uint8)
    washed_white = np.full((128, 128, 3), 255, dtype=np.uint8)

    res_normal = assessor.assess(normal_face)
    res_black = assessor.assess(pitch_black)
    res_white = assessor.assess(washed_white)

    # Normal lighting score should be high
    assert res_normal.lighting_score >= 80.0
    assert res_normal.score >= 80.0
    assert res_normal.is_acceptable is True

    # Pitch black must yield 0 lighting and score
    assert res_black.lighting_score == 0.0
    assert res_black.score == 0.0
    assert res_black.is_acceptable is False

    # Washed out white must yield 0 lighting and score
    assert res_white.lighting_score == 0.0
    assert res_white.score == 0.0
    assert res_white.is_acceptable is False

    # Substantial difference
    assert res_normal.score > res_black.score + 70.0
    assert res_normal.score > res_white.score + 70.0


def test_underexposed_and_overexposed_penalties():
    """Verify penalties for underexposed (< 40) and overexposed (> 220) luminance."""
    assessor = FaceQualityAssessor()
    normal_face = make_synthetic_face(128, bg_lum=135)

    # Severely underexposed face (mean luminance < 25)
    underexposed = (normal_face * 0.15).astype(np.uint8)
    # Severely overexposed face (mean luminance > 230)
    overexposed = np.clip(normal_face.astype(np.float32) + 115, 0, 255).astype(np.uint8)

    res_under = assessor.assess(underexposed)
    res_over = assessor.assess(overexposed)

    assert res_under.lighting_score < 35.0
    assert res_under.score < 40.0
    assert res_under.is_acceptable is False

    assert res_over.lighting_score < 35.0
    assert res_over.score < 40.0
    assert res_over.is_acceptable is False


def test_small_face_rejected():
    """Verify small faces (< 24px and < 32px) are rejected or given low score."""
    assessor = FaceQualityAssessor()

    for size in (12, 16, 20, 23):
        tiny_crop = cv2.resize(make_synthetic_face(128), (size, size))
        res = assessor.assess(tiny_crop)
        # Requirement: 0-35 Poor / Unusable, and rejected
        assert res.score < 35.0, f"Size {size} scored {res.score}, expected < 35"
        assert res.is_acceptable is False, f"Size {size} should not be acceptable"
        assert res.resolution_score < 18.0

    # 40x40 is acceptable size (> 32) but poor/fair resolution
    fair_crop = cv2.resize(make_synthetic_face(128), (40, 40))
    res_fair = assessor.assess(fair_crop)
    assert 35.0 <= res_fair.resolution_score <= 45.0
    assert 35.0 <= res_fair.score <= 65.0  # Fair tier


def test_properly_populated_face_quality_result():
    """Verify FaceQualityResult instance is properly populated with contract fields and metrics."""
    assessor = FaceQualityAssessor()
    crop = make_synthetic_face(128)
    pose = FacePose(yaw=5.0, pitch=-2.0, roll=0.0, category="frontal")
    landmarks = [(40.0, 45.0), (88.0, 45.0), (64.0, 70.0), (46.0, 95.0), (82.0, 95.0)]

    result = assessor.assess(crop, pose=pose, landmarks=landmarks, landmark_confidence=0.98)

    # Check contract fields
    assert isinstance(result, FaceQualityResult)
    assert 0.0 <= result.score <= 100.0
    assert result.sharpness > 0.0
    assert 0.0 <= result.motion_blur <= 1.0
    assert 0.0 <= result.resolution_score <= 100.0
    assert 0.0 <= result.lighting_score <= 100.0
    assert 0.0 <= result.pose_score <= 100.0
    assert 0.0 <= result.size_score <= 100.0
    assert 0.0 <= result.stability_score <= 100.0
    assert isinstance(result.is_acceptable, bool)
    assert isinstance(result.metrics, dict)

    # Check metrics breakdown
    m = result.metrics
    expected_metrics = [
        "laplacian_var", "tenengrad", "sharpness_score", "clarity_score",
        "directional_ratio", "spectral_hf_ratio", "mean_luminance",
        "contrast_std", "exposure_factor", "face_width", "face_height",
        "face_area", "yaw", "pitch", "roll", "tier_code"
    ]
    for key in expected_metrics:
        assert key in m, f"Metric '{key}' missing from result.metrics"
        assert isinstance(m[key], (int, float)), f"Metric '{key}' must be numeric"

    # Verify serialization
    d = result.to_dict()
    assert d["score"] == round(result.score, 1)
    assert d["is_acceptable"] == result.is_acceptable
    assert isinstance(d["metrics"], dict)


def test_motion_blur_detection():
    """Verify motion blur estimation detects directional smear and penalizes quality."""
    assessor = FaceQualityAssessor()
    sharp_crop = make_synthetic_face(128)

    # Create horizontal motion blur kernel
    k_size = 15
    k_h = np.zeros((k_size, k_size), dtype=np.float32)
    k_h[k_size // 2, :] = 1.0 / k_size
    horiz_blur = cv2.filter2D(sharp_crop, -1, k_h)

    # Create diagonal motion blur kernel
    k_diag = np.diag(np.ones(k_size, dtype=np.float32)) / float(k_size)
    diag_blur = cv2.filter2D(sharp_crop, -1, k_diag)

    res_sharp = assessor.assess(sharp_crop)
    res_horiz = assessor.assess(horiz_blur)
    res_diag = assessor.assess(diag_blur)

    # Sharp face has near zero motion blur
    assert res_sharp.motion_blur < 0.15

    # Motion blurred faces have high blur score
    assert res_horiz.motion_blur >= 0.65
    assert res_diag.motion_blur >= 0.65

    # Directional ratio is low in motion blurred images (strong anisotropy)
    assert res_horiz.metrics["directional_ratio"] < 0.30
    assert res_diag.metrics["directional_ratio"] < 0.35

    # Motion blur should cause failure of acceptance and low composite score
    assert res_horiz.is_acceptable is False
    assert res_diag.is_acceptable is False
    assert res_horiz.score < 35.0
    assert res_diag.score < 35.0


def test_pose_usability_scoring():
    """Verify pose scoring rewards frontal and moderate angle, penalizes extreme profile and down-pitch."""
    assessor = FaceQualityAssessor()
    crop = make_synthetic_face(128)

    # 1. Frontal (yaw 0, pitch 0)
    res_frontal = assessor.assess(crop, pose=FacePose(yaw=0.0, pitch=0.0, roll=0.0))
    assert res_frontal.pose_score >= 98.0
    assert res_frontal.score >= 85.0

    # 2. Moderate angle (yaw 25, pitch 0) -> reward factor ~ 0.85-0.90
    res_angle = assessor.assess(crop, pose=FacePose(yaw=25.0, pitch=0.0, roll=0.0))
    assert 85.0 <= res_angle.pose_score <= 95.0
    assert res_angle.is_acceptable is True

    # 3. Extreme profile (yaw 70, pitch 0) -> heavy penalty
    res_profile = assessor.assess(crop, pose=FacePose(yaw=70.0, pitch=0.0, roll=0.0))
    assert res_profile.pose_score < 45.0
    assert res_profile.score < res_frontal.score - 20.0

    # 4. Down-pitch (pitch -35, yaw 0) -> penalty for looking down
    res_downpitch = assessor.assess(crop, pose=FacePose(yaw=0.0, pitch=-35.0, roll=0.0))
    assert res_downpitch.pose_score < 65.0
    assert res_downpitch.score < res_frontal.score - 5.0

    # 5. Extreme down-pitch (pitch -50)
    res_extreme_down = assessor.assess(crop, pose=FacePose(yaw=0.0, pitch=-50.0, roll=0.0))
    assert res_extreme_down.pose_score < 40.0


def test_landmark_pose_estimation():
    """Verify pose estimation from 5-point facial landmarks."""
    assessor = FaceQualityAssessor()
    crop = make_synthetic_face(128)

    # Frontal landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
    landmarks_frontal = [(40.0, 45.0), (88.0, 45.0), (64.0, 70.0), (46.0, 95.0), (82.0, 95.0)]
    res_f = assessor.assess(crop, landmarks=landmarks_frontal)
    assert abs(res_f.metrics["yaw"]) <= 5.0
    assert res_f.pose_score >= 95.0

    # Left-angle landmarks: nose shifted left towards left eye (in image coords)
    landmarks_angle = [(40.0, 45.0), (88.0, 45.0), (52.0, 70.0), (44.0, 95.0), (78.0, 95.0)]
    res_a = assessor.assess(crop, landmarks=landmarks_angle)
    assert res_a.metrics["yaw"] < -15.0


def test_landmark_stability_scoring():
    """Verify landmark detection confidence and geometric consistency."""
    assessor = FaceQualityAssessor()
    crop = make_synthetic_face(128)

    # Valid landmarks + confidence 0.95
    valid_landmarks = [(40.0, 45.0), (88.0, 45.0), (64.0, 70.0), (46.0, 95.0), (82.0, 95.0)]
    res_valid = assessor.assess(crop, landmarks=valid_landmarks, landmark_confidence=0.95)
    assert res_valid.stability_score >= 90.0

    # Low landmark confidence
    res_low_conf = assessor.assess(crop, landmarks=valid_landmarks, landmark_confidence=0.40)
    assert res_low_conf.stability_score <= 40.0

    # Inverted geometry (mouth above eyes)
    inverted_landmarks = [(40.0, 95.0), (88.0, 95.0), (64.0, 70.0), (46.0, 45.0), (82.0, 45.0)]
    res_inverted = assessor.assess(crop, landmarks=inverted_landmarks)
    assert res_inverted.stability_score < 60.0


def test_quality_tiers_mapping():
    """Verify calibrated composite score ranges match defined quality tiers."""
    assert get_quality_tier(0.0) == "poor"
    assert get_quality_tier(34.9) == "poor"
    assert get_quality_tier(35.0) == "fair"
    assert get_quality_tier(59.9) == "fair"
    assert get_quality_tier(60.0) == "good"
    assert get_quality_tier(79.9) == "good"
    assert get_quality_tier(80.0) == "excellent"
    assert get_quality_tier(100.0) == "excellent"


def test_observation_integration():
    """Verify FaceObservation quality evaluation and attachment."""
    assessor = FaceQualityAssessor()
    crop = make_synthetic_face(128)
    obs = FaceObservation(
        observation_id="obs_test_1",
        track_id=42,
        timestamp=123.45,
        bbox=(10.0, 10.0, 138.0, 138.0),
        confidence=0.92,
        face_crop=crop,
        pose=FacePose(yaw=0.0, pitch=0.0, roll=0.0),
    )

    q = assessor.assess_observation(obs)
    assert obs.quality is not None
    assert obs.quality == q
    assert q.score >= 80.0
    assert q.is_acceptable is True

    # Verify serialization of observation containing quality
    d = obs.to_dict()
    assert d["observation_id"] == "obs_test_1"
    assert d["quality"]["score"] == round(q.score, 1)


def test_custom_thresholds():
    """Verify custom quality thresholds can enforce stricter acceptance criteria."""
    strict_thresholds = QualityThresholds(
        min_score=90.0,
        min_sharpness=500.0,
        min_width=100,
        min_height=100,
    )
    strict_assessor = FaceQualityAssessor(thresholds=strict_thresholds)
    lenient_assessor = FaceQualityAssessor()

    # Create a 64x64 face crop (meets default 32x32, fails strict 100x100)
    mid_crop = cv2.resize(make_synthetic_face(128), (64, 64))

    res_strict = strict_assessor.assess(mid_crop)
    res_lenient = lenient_assessor.assess(mid_crop)

    assert res_strict.is_acceptable is False
    assert res_lenient.is_acceptable is True


def test_edge_cases():
    """Verify engine handles edge cases (None, empty, grayscale, BGRA, float) without error."""
    assessor = FaceQualityAssessor()

    # None crop
    res_none = assessor.assess(None)
    assert res_none.score == 0.0
    assert res_none.is_acceptable is False

    # Empty ndarray
    res_empty = assessor.assess(np.array([]))
    assert res_empty.score == 0.0
    assert res_empty.is_acceptable is False

    # 1-channel grayscale crop
    gray_crop = cv2.cvtColor(make_synthetic_face(128), cv2.COLOR_BGR2GRAY)
    res_gray = assessor.assess(gray_crop)
    assert res_gray.score >= 80.0

    # 4-channel BGRA crop
    bgra_crop = cv2.cvtColor(make_synthetic_face(128), cv2.COLOR_BGR2BGRA)
    res_bgra = assessor.assess(bgra_crop)
    assert res_bgra.score >= 80.0

    # Float normalized crop [0.0, 1.0]
    float_crop = make_synthetic_face(128).astype(np.float32) / 255.0
    res_float = assessor.assess(float_crop)
    assert res_float.score >= 80.0

    # Extreme aspect ratio (thin strip)
    strip_crop = np.full((10, 128, 3), 128, dtype=np.uint8)
    res_strip = assessor.assess(strip_crop)
    assert res_strip.score < 35.0
    assert res_strip.is_acceptable is False


def test_convenience_function():
    """Verify assess_face_quality top-level function works identically to assessor instance."""
    crop = make_synthetic_face(128)
    res = assess_face_quality(crop)
    assert isinstance(res, FaceQualityResult)
    assert res.score >= 80.0
    assert res.is_acceptable is True
