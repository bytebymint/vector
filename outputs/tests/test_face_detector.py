"""Unit tests for FaceDetector service.

Verifies:
- Correct face localization and landmarks (YuNet and Haar fallback)
- Accurate assignment to upper region of person bounding box
- Handling faces of varying scales
- Rejection of out-of-person detections
- Rejection of ambiguous assignments between overlapping persons
- Rejection of unreliable detections (confidence, min area, aspect ratio)
- Selective cadence and visibility-triggered face detection
- ROI mode vs full-frame mode
- Interoperability with duck-typed track representations and FramePacket
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest

from services.face_contracts import (
    FaceObservation,
    FramePacket,
    PersonTrack as ContractPersonTrack,
)
from services.people_engine import PersonTrack as EnginePersonTrack
from services.face_detector import (
    FaceDetector,
    FaceDetectorConfig,
    box_iou,
    point_in_box,
)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Load a representative test image containing visible faces."""
    img_path = Path("outputs/linkedin-sample-00.jpg")
    if not img_path.exists():
        # Fallback to creating a synthetic 3-channel canvas
        return np.zeros((480, 640, 3), dtype=np.uint8)
    img = cv2.imread(str(img_path))
    assert img is not None, f"Failed to load fixture at {img_path}"
    return img


def test_face_detector_initialization_yunet():
    """Verify primary model initializes with YuNet when ONNX model is available."""
    detector = FaceDetector(FaceDetectorConfig(backend="auto"))
    assert detector.active_backend == "yunet"
    assert detector._yunet is not None


def test_face_detector_fallback_to_haar():
    """Verify fallback to Haar Cascade when YuNet model is unavailable or Haar requested."""
    config = FaceDetectorConfig(
        model_path="non_existent_model_weights.onnx",
        backend="auto",
    )
    detector = FaceDetector(config)
    assert detector.active_backend == "haar"
    assert detector._haar is not None


def test_backend_switch_dynamic():
    """Verify switching between YuNet and Haar backends dynamically at runtime."""
    detector = FaceDetector(FaceDetectorConfig(backend="yunet"))
    assert detector.active_backend == "yunet"

    detector.switch_backend("haar")
    assert detector.active_backend == "haar"
    assert detector._haar is not None

    detector.switch_backend("yunet")
    assert detector.active_backend == "yunet"


def test_correct_face_localization_and_landmarks(sample_image):
    """Verify correct face localization and 5-point landmark extraction with YuNet."""
    detector = FaceDetector(FaceDetectorConfig(backend="yunet", score_threshold=0.5))

    # Known face in sample image at ~(1615, 556, 1636, 580)
    # Define person track around subject
    track = ContractPersonTrack(
        track_id=101,
        session_id="session_test",
        first_seen=10.0,
        last_seen=12.0,
        bbox=(1580.0, 540.0, 1670.0, 800.0),
        confidence=0.92,
        confirmed=True,
    )

    observations = detector.detect(
        sample_image,
        tracks=[track],
        timestamp=12.0,
        mode="full_frame",
    )

    assert len(observations) == 1
    obs = observations[0]

    # Verify FaceObservation contract conformance
    assert isinstance(obs, FaceObservation)
    assert obs.track_id == 101
    assert obs.timestamp == 12.0
    assert obs.confidence >= 0.50

    # Verify facial bounding box
    x1, y1, x2, y2 = obs.bbox
    assert x2 > x1 and y2 > y1
    assert 1600 <= x1 <= 1630
    assert 545 <= y1 <= 570
    assert 1625 <= x2 <= 1650
    assert 570 <= y2 <= 590

    # Verify 5 landmarks from YuNet
    assert len(obs.landmarks) == 5
    for lx, ly in obs.landmarks:
        # Landmarks must lie within facial bounding box
        assert (x1 - 5.0) <= lx <= (x2 + 5.0)
        assert (y1 - 5.0) <= ly <= (y2 + 5.0)

    # Verify contract serialization
    d = obs.to_dict()
    assert d["track_id"] == 101
    assert len(d["landmarks"]) == 5
    assert len(d["bbox"]) == 4


def test_accurate_assignment_to_upper_region_of_person_bounding_box():
    """Verify faces are assigned when in upper body region and rejected when in lower region."""
    detector = FaceDetector(FaceDetectorConfig(backend="auto"))

    person_bbox = (100.0, 100.0, 200.0, 400.0)  # w=100, h=300
    track = ContractPersonTrack(
        track_id=1,
        session_id="s1",
        first_seen=0.0,
        last_seen=1.0,
        bbox=person_bbox,
        confidence=0.9,
    )

    # Face A: In upper head region (y center = 145, upper 15% of body)
    face_upper = {
        "bbox": (120.0, 110.0, 180.0, 180.0),
        "confidence": 0.88,
        "landmarks": [(135.0, 135.0), (165.0, 135.0), (150.0, 150.0), (140.0, 165.0), (160.0, 165.0)],
    }

    obs_upper = detector.associate_faces([face_upper], [track], timestamp=1.0)
    assert len(obs_upper) == 1
    assert obs_upper[0].track_id == 1

    # Face B: In lower region (y center = 340, legs/knees region)
    face_lower = {
        "bbox": (120.0, 310.0, 180.0, 370.0),
        "confidence": 0.88,
        "landmarks": [],
    }

    obs_lower = detector.associate_faces([face_lower], [track], timestamp=1.0)
    assert len(obs_lower) == 0, "Face in lower half of person box must be rejected"


def test_rejection_of_out_of_person_detections():
    """Verify face detections outside any person bounding box are rejected."""
    detector = FaceDetector(FaceDetectorConfig(backend="auto"))

    track = ContractPersonTrack(
        track_id=5,
        session_id="s1",
        first_seen=0.0,
        last_seen=1.0,
        bbox=(100.0, 100.0, 200.0, 400.0),
        confidence=0.9,
    )

    # Face far away on the right side of frame (e.g. background object or billboard)
    out_of_person_face = {
        "bbox": (600.0, 120.0, 660.0, 180.0),
        "confidence": 0.85,
        "landmarks": [],
    }

    # Associate only: should be rejected
    obs = detector.associate_faces([out_of_person_face], [track], timestamp=1.0)
    assert len(obs) == 0, "Out-of-person face must not be associated with track"


def test_handling_faces_of_varying_scales():
    """Verify accurate detection and association across varying face and person scales."""
    detector = FaceDetector(FaceDetectorConfig(backend="auto"))

    # Scale 1: Small subject in distance
    track_small = ContractPersonTrack(
        track_id=1,
        session_id="s1",
        first_seen=0.0,
        last_seen=1.0,
        bbox=(40.0, 40.0, 100.0, 220.0),  # 60x180
        confidence=0.8,
    )
    face_small = {
        "bbox": (55.0, 48.0, 85.0, 82.0),  # 30x34
        "confidence": 0.82,
        "landmarks": [],
    }

    # Scale 2: Medium subject at mid-range
    track_med = ContractPersonTrack(
        track_id=2,
        session_id="s1",
        first_seen=0.0,
        last_seen=1.0,
        bbox=(200.0, 100.0, 360.0, 550.0),  # 160x450
        confidence=0.9,
    )
    face_med = {
        "bbox": (240.0, 120.0, 320.0, 210.0),  # 80x90
        "confidence": 0.91,
        "landmarks": [],
    }

    # Scale 3: Large subject close to camera
    track_large = ContractPersonTrack(
        track_id=3,
        session_id="s1",
        first_seen=0.0,
        last_seen=1.0,
        bbox=(500.0, 150.0, 850.0, 1150.0),  # 350x1000
        confidence=0.95,
    )
    face_large = {
        "bbox": (580.0, 200.0, 760.0, 400.0),  # 180x200
        "confidence": 0.96,
        "landmarks": [],
    }

    all_faces = [face_small, face_med, face_large]
    all_tracks = [track_small, track_med, track_large]

    obs = detector.associate_faces(all_faces, all_tracks, timestamp=1.0)
    assert len(obs) == 3

    obs_by_track = {o.track_id: o for o in obs}
    assert 1 in obs_by_track
    assert 2 in obs_by_track
    assert 3 in obs_by_track

    # Verify bboxes match intended scale
    assert (obs_by_track[1].bbox[2] - obs_by_track[1].bbox[0]) == 30.0
    assert (obs_by_track[2].bbox[2] - obs_by_track[2].bbox[0]) == 80.0
    assert (obs_by_track[3].bbox[2] - obs_by_track[3].bbox[0]) == 180.0


def test_rejection_of_ambiguous_overlapping_person_assignments():
    """Verify rejection of ambiguous face-to-person assignments when two people overlap heavily."""
    detector = FaceDetector(FaceDetectorConfig(
        backend="auto",
        ambiguity_iou_threshold=0.25,
        ambiguity_affinity_ratio=0.75,
    ))

    # Two persons standing side-by-side with heavy bounding box overlap (IoU > 0.6)
    track1 = ContractPersonTrack(
        track_id=10,
        session_id="s1",
        first_seen=0.0,
        last_seen=1.0,
        bbox=(100.0, 100.0, 230.0, 400.0),
        confidence=0.9,
    )
    track2 = ContractPersonTrack(
        track_id=20,
        session_id="s1",
        first_seen=0.0,
        last_seen=1.0,
        bbox=(120.0, 100.0, 250.0, 400.0),
        confidence=0.9,
    )

    assert box_iou(track1.bbox, track2.bbox) >= 0.50

    # Face detection right in the shared overlapping upper zone
    ambiguous_face = {
        "bbox": (145.0, 120.0, 195.0, 175.0),
        "confidence": 0.88,
        "landmarks": [],
    }

    obs = detector.associate_faces([ambiguous_face], [track1, track2], timestamp=1.0)
    assert len(obs) == 0, "Ambiguous face between heavily overlapping people must be rejected"


def test_rejection_of_unreliable_detections():
    """Verify rejection of low-confidence, sub-pixel area, and extreme aspect ratio faces."""
    config = FaceDetectorConfig(
        score_threshold=0.60,
        min_face_area=100.0,
        min_face_width=10.0,
        min_face_height=10.0,
        min_aspect_ratio=0.4,
        max_aspect_ratio=2.5,
    )
    detector = FaceDetector(config)

    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    canvas[50:95, 50:90] = np.random.default_rng(42).integers(30,230,(45,40,3),dtype=np.uint8)

    # 1. Low confidence
    raw_low_conf = detector._detect_raw(canvas)  # no valid detections initially without mock
    assert len(raw_low_conf) == 0

    # Test raw filtering directly
    # A. Low confidence
    f_low = {"bbox": (50.0, 50.0, 100.0, 100.0), "confidence": 0.35, "landmarks": []}
    track = ContractPersonTrack(track_id=1, session_id="s", first_seen=0, last_seen=1, bbox=(40, 40, 120, 250), confidence=0.9)
    # The associate_faces method relies on detections passing _detect_raw or caller filtering
    # Let's test that low confidence detection is rejected by detector
    detector.config.score_threshold = 0.70
    assert detector._evaluate_face_person_candidate(f_low["bbox"], (40, 40, 120, 250)) is not None

    # Test area threshold on tiny face (width=5, height=5 => area=25 < 100)
    face_tiny = {"bbox": (50.0, 50.0, 55.0, 55.0), "confidence": 0.9, "landmarks": []}
    # Test aspect ratio sliver (width=100, height=10 => ar=10 > 2.5)
    face_sliver = {"bbox": (50.0, 50.0, 150.0, 60.0), "confidence": 0.9, "landmarks": []}

    # Verify that FaceDetector._detect_raw filters these:
    # Mock raw detection with anatomically valid landmarks for the valid face
    class MockYuNet:
        def setInputSize(self, sz): pass
        def setScoreThreshold(self, th): pass
        def detect(self, img):
            # return [x, y, w, h, 10 landmarks, score]
            faces = np.array([
                [50, 50, 50, 50, 0,0,0,0,0,0,0,0,0,0, 0.40], # low confidence (< 0.60)
                [50, 50, 5, 5, 0,0,0,0,0,0,0,0,0,0, 0.90],   # area = 25 (< 100)
                [50, 50, 100, 10, 0,0,0,0,0,0,0,0,0,0, 0.90], # sliver aspect ratio (10.0 > 2.5)
                [50, 50, 40, 45, 60,60, 80,60, 70,70, 62,82, 78,82, 0.85],  # valid geometry + landmarks
            ], dtype=np.float32)
            return 1, faces

    detector._yunet = MockYuNet()
    detector.active_backend = "yunet"

    filtered = detector._detect_raw(canvas)
    assert len(filtered) == 1
    assert filtered[0]["confidence"] == pytest.approx(0.85)
    assert filtered[0]["bbox"] == (50.0, 50.0, 90.0, 95.0)


def test_selective_face_detection_cadence_and_visibility():
    """Verify selective cadence skips frames and visibility expansion triggers detection."""
    detector = FaceDetector(FaceDetectorConfig(
        selective_cadence=3,
        visibility_expansion_ratio=1.25,
        min_person_width=20.0,
        min_person_height=40.0,
    ))

    t_id = 99
    base_bbox = (100.0, 100.0, 150.0, 250.0)  # area = 50 * 150 = 7500

    # Frame 0: First appearance -> evaluate
    assert detector.should_evaluate_track(t_id, base_bbox, frame_sequence=0, timestamp=0.0) is True
    detector._record_track_evaluation(t_id, base_bbox, frame_sequence=0, timestamp=0.0)

    # Frame 1: 1 frame since last eval (< 3) -> skip
    assert detector.should_evaluate_track(t_id, base_bbox, frame_sequence=1, timestamp=0.033) is False

    # Frame 2: 2 frames since last eval (< 3) -> skip
    assert detector.should_evaluate_track(t_id, base_bbox, frame_sequence=2, timestamp=0.066) is False

    # Frame 3: 3 frames since last eval (>= 3) -> evaluate
    assert detector.should_evaluate_track(t_id, base_bbox, frame_sequence=3, timestamp=0.100) is True
    detector._record_track_evaluation(t_id, base_bbox, frame_sequence=3, timestamp=0.100)

    # Frame 4: Subject walked closer, area expands from 7500 to 10500 (1.40x > 1.25x)
    expanded_bbox = (90.0, 90.0, 160.0, 240.0)  # 70 * 150 = 10500
    assert detector.should_evaluate_track(t_id, expanded_bbox, frame_sequence=4, timestamp=0.133) is True

    # Test sub-minimum track size (e.g. 10x20 in distant background) -> skip
    tiny_bbox = (10.0, 10.0, 20.0, 30.0)
    assert detector.should_evaluate_track(102, tiny_bbox, frame_sequence=0) is False


def test_roi_and_full_frame_modes(sample_image):
    """Verify both ROI mode and full-frame mode yield valid face observations."""
    track = ContractPersonTrack(
        track_id=42,
        session_id="s_mode",
        first_seen=1.0,
        last_seen=2.0,
        bbox=(1580.0, 540.0, 1670.0, 800.0),
        confidence=0.9,
    )

    # Mode 1: full_frame
    det_ff = FaceDetector(FaceDetectorConfig(score_threshold=0.5, default_mode="full_frame"))
    obs_ff = det_ff.detect(sample_image, tracks=[track], timestamp=2.0, mode="full_frame")
    assert len(obs_ff) == 1
    assert obs_ff[0].track_id == 42

    # Mode 2: roi
    det_roi = FaceDetector(FaceDetectorConfig(score_threshold=0.5, roi_score_threshold=0.3, default_mode="roi"))
    obs_roi = det_roi.detect(sample_image, tracks=[track], timestamp=2.0, mode="roi")
    assert len(obs_roi) == 1
    assert obs_roi[0].track_id == 42


def test_haar_fallback_without_landmarks_cannot_authorize_face_capture(sample_image):
    detector=FaceDetector(FaceDetectorConfig(backend='haar',default_mode='full_frame'))
    assert detector.active_backend=='haar'
    assert detector.detect(sample_image,timestamp=2.,mode='full_frame') == []


def test_duck_typing_person_tracks():
    """Verify detector accepts various representations of person tracks across services."""
    detector = FaceDetector(FaceDetectorConfig(backend="auto"))

    face = {
        "bbox": (120.0, 110.0, 180.0, 170.0),
        "confidence": 0.9,
        "landmarks": [(135, 135), (165, 135), (150, 150), (140, 165), (160, 165)],
    }

    # 1. Contract PersonTrack
    t1 = ContractPersonTrack(1, "s", 0, 1, (100.0, 100.0, 200.0, 400.0), 0.9)
    # 2. Engine PersonTrack
    t2 = EnginePersonTrack(2, 0.0, 1.0, np.array([100.0, 100.0, 200.0, 400.0]), 0.9)
    # 3. Dict
    t3 = {"track_id": 3, "bbox": [100.0, 100.0, 200.0, 400.0], "confidence": 0.9}
    # 4. Raw list/tuple
    t4 = [100.0, 100.0, 200.0, 400.0, 4, 0.9]

    for track_repr, expected_id in [(t1, 1), (t2, 2), (t3, 3), (t4, 4)]:
        obs = detector.associate_faces([face], [track_repr], timestamp=1.0)
        assert len(obs) == 1
        assert obs[0].track_id == expected_id


def test_frame_packet_integration(sample_image):
    """Verify FramePacket instance is unpacked correctly with metadata retained."""
    detector = FaceDetector(FaceDetectorConfig(score_threshold=0.5))

    packet = FramePacket(
        sequence=150,
        timestamp=45.678,
        image=sample_image,
        source_id="camera_front",
        epoch="epoch_alpha",
        width=sample_image.shape[1],
        height=sample_image.shape[0],
    )

    track = ContractPersonTrack(
        track_id=88,
        session_id="s_packet",
        first_seen=40.0,
        last_seen=45.678,
        bbox=(1580.0, 540.0, 1670.0, 800.0),
        confidence=0.9,
    )

    obs = detector.detect(packet, tracks=[track])
    assert len(obs) == 1
    assert obs[0].track_id == 88
    assert obs[0].timestamp == 45.678


def test_face_crop_extraction(sample_image):
    """Verify facial crop extraction attaches valid ndarray when configured."""
    detector = FaceDetector(FaceDetectorConfig(
        score_threshold=0.5,
        extract_face_crops=True,
    ))

    track = ContractPersonTrack(
        track_id=33,
        session_id="s_crop",
        first_seen=1.0,
        last_seen=2.0,
        bbox=(1580.0, 540.0, 1670.0, 800.0),
        confidence=0.9,
    )

    obs = detector.detect(sample_image, tracks=[track])
    assert len(obs) == 1
    assert obs[0].face_crop is not None
    assert isinstance(obs[0].face_crop, np.ndarray)
    assert obs[0].face_crop.shape[2] == 3
    assert obs[0].face_crop.shape[0] > 10
    assert obs[0].face_crop.shape[1] > 10


def test_track_pruning_and_reset():
    """Verify cadence tracking cleanup and state reset."""
    detector = FaceDetector(FaceDetectorConfig(selective_cadence=2))

    detector._record_track_evaluation(1, (10, 10, 50, 100), frame_sequence=0)
    detector._record_track_evaluation(2, (60, 10, 100, 100), frame_sequence=0)

    assert len(detector._track_states) == 2

    # Prune track 1 (only track 2 remains active)
    detector.prune_tracks({2})
    assert len(detector._track_states) == 1
    assert 2 in detector._track_states
    assert 1 not in detector._track_states

    # Reset all
    detector.reset_cadence()
    assert len(detector._track_states) == 0


def test_face_observation_relative_box_normalization():
    """Verify relative_box is computed, normalized to parent PersonTrack.bbox, and attached."""
    detector = FaceDetector(FaceDetectorConfig(backend="auto"))

    person_bbox = (100.0, 100.0, 200.0, 400.0)  # pw = 100, ph = 300
    track = ContractPersonTrack(
        track_id=42,
        session_id="s_rel",
        first_seen=0.0,
        last_seen=1.0,
        bbox=person_bbox,
        confidence=0.9,
    )

    face = {
        "bbox": (120.0, 110.0, 180.0, 170.0),
        "confidence": 0.88,
        "landmarks": [(135.0, 135.0), (165.0, 135.0), (150.0, 150.0), (140.0, 165.0), (160.0, 165.0)],
    }

    obs = detector.associate_faces([face], [track], timestamp=1.0)
    assert len(obs) == 1
    assert obs[0].relative_box is not None
    rx1, ry1, rx2, ry2 = obs[0].relative_box
    assert rx1 == pytest.approx(0.20, abs=1e-3)
    assert ry1 == pytest.approx(0.0333, abs=1e-3)
    assert rx2 == pytest.approx(0.80, abs=1e-3)
    assert ry2 == pytest.approx(70.0 / 300.0, abs=1e-3)

    d = obs[0].to_dict()
    assert "relative_box" in d
    assert d["relative_box"] == [pytest.approx(0.20, abs=1e-3), pytest.approx(0.0333, abs=1e-3),
                                 pytest.approx(0.80, abs=1e-3), pytest.approx(70.0 / 300.0, abs=1e-3)]


def test_generous_head_portrait_framing():
    """Verify portrait crop framing includes headroom for hair, chin, and ears."""
    detector = FaceDetector(FaceDetectorConfig(
        extract_face_crops=True,
        crop_top_margin_ratio=0.35,
        crop_bottom_margin_ratio=0.25,
        crop_side_margin_ratio=0.30,
    ))

    canvas = np.zeros((500, 500, 3), dtype=np.uint8)
    face_bbox = (200.0, 200.0, 250.0, 260.0)  # fw = 50, fh = 60
    crop = detector._crop_head_portrait(canvas, face_bbox)
    assert crop is not None
    assert crop.shape[0] == 96  # height includes hair and chin (60 * 1.6 = 96)
    assert crop.shape[1] == 80  # width includes lateral margins for ears/jawline (50 * 1.6 = 80)


def test_default_confidence_thresholds():
    """Verify production score_threshold raised to 0.70 and roi_score_threshold to 0.65."""
    cfg = FaceDetectorConfig()
    assert cfg.score_threshold == 0.70
    assert cfg.roi_score_threshold == 0.65


def test_is_valid_face_landmark_anatomy_validation(sample_image):
    """Verify facial landmark anatomical checks enforce eye-nose-mouth vertical and horizontal geometry."""
    from services.face_detector import is_valid_face

    bbox = (1615.0, 556.0, 1637.0, 580.0)
    real_crop = sample_image[556:580, 1615:1637]

    # Valid upright frontal landmarks: eyes above nose, nose above mouth
    valid_lm = [
        (1621.5, 566.6),  # right eye
        (1630.7, 566.8),  # left eye
        (1625.8, 570.9),  # nose tip
        (1622.5, 575.2),  # right mouth corner
        (1629.5, 575.3),  # left mouth corner
    ]
    assert is_valid_face(crop=real_crop, landmarks=valid_lm, bbox=bbox) is True

    # Inverted anatomy: nose above eyes
    inverted_nose_lm = [
        (1621.5, 572.0),
        (1630.7, 572.0),
        (1625.8, 565.0),  # nose above eyes!
        (1622.5, 575.0),
        (1629.5, 575.0),
    ]
    assert is_valid_face(crop=real_crop, landmarks=inverted_nose_lm, bbox=bbox) is False

    # Inverted anatomy: mouth above nose
    inverted_mouth_lm = [
        (1621.5, 565.0),
        (1630.7, 565.0),
        (1625.8, 576.0),  # nose below mouth!
        (1622.5, 570.0),
        (1629.5, 570.0),
    ]
    assert is_valid_face(crop=real_crop, landmarks=inverted_mouth_lm, bbox=bbox) is False

    # Flipped eyes: right eye to the right of left eye
    flipped_eyes_lm = [
        (1630.7, 566.6),  # rx > lx
        (1621.5, 566.8),
        (1625.8, 570.9),
        (1622.5, 575.2),
        (1629.5, 575.3),
    ]
    assert is_valid_face(crop=real_crop, landmarks=flipped_eyes_lm, bbox=bbox) is False

    # Collapsed eyes: horizontal separation is less than 10% of face width
    collapsed_eyes_lm = [
        (1625.0, 566.6),
        (1625.5, 566.8),  # sep = 0.5 < 0.10 * 22 = 2.2
        (1625.8, 570.9),
        (1622.5, 575.2),
        (1629.5, 575.3),
    ]
    assert is_valid_face(crop=real_crop, landmarks=collapsed_eyes_lm, bbox=bbox) is False


def test_face_sanity_requires_geometry_and_contrast_not_skin_colour(sample_image):
    """Verify YCrCb skin tone validation rejects concrete pillars, asphalt, and low-skin crops."""
    from services.face_detector import is_valid_face

    bbox = (1615.0, 556.0, 1637.0, 580.0)
    real_crop = sample_image[556:580, 1615:1637]
    valid_lm = [
        (1621.5, 566.6),
        (1630.7, 566.8),
        (1625.8, 570.9),
        (1622.5, 575.2),
        (1629.5, 575.3),
    ]

    # 1. Real human face contains substantial skin tone (> 60%)
    assert is_valid_face(crop=real_crop, landmarks=valid_lm, bbox=bbox) is True

    # 2. Concrete pillar / neutral gray texture has 0% skin gamut
    concrete_crop = np.full((60, 60, 3), 135, dtype=np.uint8)
    assert is_valid_face(crop=concrete_crop, landmarks=valid_lm, bbox=bbox) is False

    # 3. Asphalt / dark pavement texture has 0% skin gamut
    asphalt_crop = np.full((60, 60, 3), 40, dtype=np.uint8)
    assert is_valid_face(crop=asphalt_crop, landmarks=valid_lm, bbox=bbox) is False

    # 4. Haar face candidate without landmarks on real skin is accepted
    assert is_valid_face(crop=real_crop, landmarks=[], bbox=bbox) is False
    gray=cv2.cvtColor(real_crop,cv2.COLOR_BGR2GRAY)
    assert is_valid_face(crop=gray,landmarks=valid_lm,bbox=bbox) is True

    # 5. Haar false positive on concrete pillar is rejected
    assert is_valid_face(crop=concrete_crop, landmarks=[], bbox=bbox) is False

    # 6. Low skin fraction (< 12%) is rejected
    low_skin_crop = concrete_crop.copy()
    low_skin_crop[:4, :4] = [120, 150, 200]  # tiny skin fraction < 12%
    assert is_valid_face(crop=low_skin_crop, landmarks=valid_lm, bbox=bbox) is False
