"""Unit tests for PersonDetector module.

Verifies:
- Detection output returns List[PersonDetection] from services.face_contracts
- Strict filtering of non-person classes and low confidence scores
- Crowded scene handling with 20-30+ people per scene
- Correct bounding box format (x1, y1, x2, y2) and boundary clipping
- Batched detection across multiple frames
- Adaptive inference resolution based on latency budget
- CPU/GPU device selection and automatic fallback
- Coordination with outputs/services/detector.py patterns without modifying detector.py
- Official YOLO model weights integration (yolo26n.pt, yolo26s.pt)
"""
from pathlib import Path
import numpy as np
import pytest
import torch

from services.face_contracts import PersonDetection
from services.person_detector import (
    PersonDetector,
    PersonDetectorConfig,
    validate_person_detector_config,
    select_device,
    MockYOLO,
    MockBoxes,
    MockResult,
    PERSON_CLASS_ID,
)


def create_mock_result(
    boxes_data: np.ndarray,
    orig_shape: tuple = (720, 1280)
) -> MockResult:
    """Helper to create a MockResult with specified [x1, y1, x2, y2, conf, cls] array."""
    data = np.asarray(boxes_data, dtype=np.float32)
    return MockResult(data, orig_shape)


class MockYOLOModel:
    """Mock YOLO model for controllable unit tests."""
    names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(self, predictions=None):
        self.predictions = predictions or []
        self.call_history = []

    def predict(self, source, **kwargs):
        self.call_history.append((source, kwargs))
        images = source if isinstance(source, (list, tuple)) else [source]
        results = []
        for i, img in enumerate(images):
            h, w = (img.shape[:2]) if hasattr(img, "shape") else (720, 1280)
            if i < len(self.predictions):
                boxes_data = self.predictions[i]
            elif self.predictions:
                boxes_data = self.predictions[0]
            else:
                boxes_data = np.empty((0, 6), dtype=np.float32)
            results.append(MockResult(boxes_data, (h, w)))
        return results

    def __call__(self, source, **kwargs):
        return self.predict(source, **kwargs)


# ============================================================================
# 1. Contract & Configuration Tests
# ============================================================================

def test_person_detection_dataclass_contract():
    """Verify PersonDetection fields and dictionary serialization."""
    det = PersonDetection(
        bbox=(100.5, 150.25, 300.75, 450.5),
        confidence=0.8876,
        class_id=0,
        detection_id="det_test_1"
    )
    assert det.class_id == 0
    assert det.confidence == 0.8876
    assert det.bbox == (100.5, 150.25, 300.75, 450.5)
    assert det.detection_id == "det_test_1"

    d = det.to_dict()
    assert d["bbox"] == [100.5, 150.25, 300.75, 450.5]
    assert d["confidence"] == 0.8876
    assert d["class_id"] == 0
    assert d["detection_id"] == "det_test_1"


def test_config_validation_valid():
    """Verify default and customized configurations pass validation."""
    cfg = PersonDetectorConfig()
    validated = validate_person_detector_config(cfg)
    assert validated.confidence_threshold == 0.10
    assert validated.iou_threshold == 0.65
    assert validated.resolution == 640
    assert validated.adaptive is True
    assert validated.max_det == 1000

    custom = validate_person_detector_config({
        "confidence_threshold": 0.25,
        "iou_threshold": 0.50,
        "resolution": 960,
        "adaptive": False,
        "backend": "pytorch",
    })
    assert custom.confidence_threshold == 0.25
    assert custom.iou_threshold == 0.50
    assert custom.resolution == 960
    assert custom.adaptive is False


def test_config_validation_invalid():
    """Verify invalid configurations trigger informative ValueError/TypeError."""
    with pytest.raises(TypeError):
        validate_person_detector_config("not-a-dict")

    with pytest.raises(ValueError, match="confidence_threshold"):
        validate_person_detector_config({"confidence_threshold": 0.001})

    with pytest.raises(ValueError, match="confidence_threshold"):
        validate_person_detector_config({"confidence_threshold": 1.05})

    with pytest.raises(ValueError, match="iou_threshold"):
        validate_person_detector_config({"iou_threshold": 0.05})

    with pytest.raises(ValueError, match="resolution"):
        validate_person_detector_config({"resolution": 128})

    with pytest.raises(ValueError, match="backend"):
        validate_person_detector_config({"backend": "invalid_backend"})


# ============================================================================
# 2. Device Selection & Fallback Tests
# ============================================================================

def test_device_selection_auto_and_cpu():
    """Verify automatic and explicit CPU selection."""
    dev, warn = select_device("cpu")
    assert dev == "cpu"
    assert warn is None

    # Preferred CUDA on non-CUDA system gracefully falls back to CPU
    if not torch.cuda.is_available():
        dev, warn = select_device("cuda:0")
        assert dev == "cpu"
        assert warn is not None
        assert "unavailable" in warn.lower()


# ============================================================================
# 3. Detection Output & Filtering Tests
# ============================================================================

def test_detection_output_contract_type():
    """Verify detector.detect returns a List[PersonDetection] with exact contract properties."""
    detector = PersonDetector(auto_load=False)
    raw_data = np.array([
        [100, 100, 200, 300, 0.85, 0],
        [250, 150, 350, 400, 0.72, 0],
    ], dtype=np.float32)

    detector.set_model(MockYOLOModel(predictions=[raw_data]))
    detector.device = "cpu"

    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = detector.detect(image)

    assert isinstance(detections, list)
    assert len(detections) == 2
    for det in detections:
        assert isinstance(det, PersonDetection)
        assert det.class_id == PERSON_CLASS_ID
        assert len(det.bbox) == 4
        assert det.bbox[2] > det.bbox[0]
        assert det.bbox[3] > det.bbox[1]
        assert det.confidence > 0.0
        assert det.detection_id is not None


def test_filter_non_person_classes():
    """Verify non-person classes (cars, bikes, buses, dogs) are strictly filtered out."""
    detector = PersonDetector(auto_load=False)
    # Mixed detections:
    # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 16: dog
    raw_data = np.array([
        [50, 50, 120, 250, 0.90, 0],   # Person (Keep)
        [150, 100, 250, 200, 0.88, 2],  # Car (Drop)
        [300, 150, 380, 280, 0.75, 1],  # Bicycle (Drop)
        [400, 120, 480, 260, 0.82, 3],  # Motorcycle (Drop)
        [500, 100, 700, 300, 0.95, 5],  # Bus (Drop)
        [750, 100, 950, 350, 0.91, 7],  # Truck (Drop)
        [200, 200, 260, 280, 0.70, 16], # Dog (Drop)
        [800, 200, 880, 400, 0.80, 0],  # Person (Keep)
    ], dtype=np.float32)

    detector.set_model(MockYOLOModel(predictions=[raw_data]))
    detector.device = "cpu"

    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = detector.detect(image)

    assert len(detections) == 2
    assert all(det.class_id == 0 for det in detections)
    # First kept person
    assert detections[0].bbox == (50.0, 50.0, 120.0, 250.0)
    assert abs(detections[0].confidence - 0.90) < 1e-4
    # Second kept person
    assert detections[1].bbox == (800.0, 200.0, 880.0, 400.0)
    assert abs(detections[1].confidence - 0.80) < 1e-4


def test_filter_low_confidence_scores():
    """Verify confidence threshold filters low scores and respects parameter override."""
    detector = PersonDetector(config={"confidence_threshold": 0.15}, auto_load=False)
    raw_data = np.array([
        [10, 10, 50, 120, 0.95, 0],  # Keep (>= 0.15)
        [60, 10, 100, 120, 0.50, 0],  # Keep (>= 0.15)
        [110, 10, 150, 120, 0.16, 0], # Keep (>= 0.15)
        [160, 10, 200, 120, 0.15, 0], # Keep (>= 0.15)
        [210, 10, 250, 120, 0.14, 0], # Drop (< 0.15)
        [260, 10, 300, 120, 0.08, 0], # Drop (< 0.15)
        [310, 10, 350, 120, 0.02, 0], # Drop (< 0.15)
    ], dtype=np.float32)

    detector.set_model(MockYOLOModel(predictions=[raw_data]))
    detector.device = "cpu"

    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Default 0.15 threshold
    detections = detector.detect(image)
    assert len(detections) == 4
    assert all(d.confidence >= 0.15 for d in detections)

    # Override threshold at call time to 0.40
    detections_high = detector.detect(image, confidence=0.40)
    assert len(detections_high) == 2
    assert all(d.confidence >= 0.40 for d in detections_high)


# ============================================================================
# 4. Crowded Scene Handling (20-30+ People)
# ============================================================================

def test_handles_crowded_synthetic_frame_with_20_plus_people():
    """Verify accurate detection handling of crowded frames with 25-30+ individuals."""
    num_people = 32
    width, height = 1920, 1080

    crowd_boxes = []
    # Generate 32 distinct and partially overlapping people across the frame
    for i in range(num_people):
        # 4 rows of 8 people
        row = i // 8
        col = i % 8
        x1 = 50 + col * 220 + (i % 3) * 15
        y1 = 60 + row * 230 + (i % 2) * 10
        x2 = x1 + 100 + (i % 4) * 10
        y2 = y1 + 200 + (i % 3) * 10
        conf = 0.20 + (i % 10) * 0.07  # Confidences between 0.20 and 0.83
        cls_id = 0
        crowd_boxes.append([x1, y1, x2, y2, conf, cls_id])

    raw_data = np.array(crowd_boxes, dtype=np.float32)

    detector = PersonDetector(
        config={"confidence_threshold": 0.15, "iou_threshold": 0.65, "max_det": 1000},
        auto_load=False
    )
    detector.set_model(MockYOLOModel(predictions=[raw_data]))
    detector.device = "cpu"

    synthetic_frame = np.zeros((height, width, 3), dtype=np.uint8)
    detections = detector.detect(synthetic_frame)

    assert len(detections) == num_people
    assert len(detections) >= 20
    assert all(isinstance(d, PersonDetection) for d in detections)
    assert all(d.class_id == 0 for d in detections)

    # Check bounding box integrity
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        assert 0 <= x1 < x2 <= width
        assert 0 <= y1 < y2 <= height
        assert (x2 - x1) >= 50
        assert (y2 - y1) >= 100


# ============================================================================
# 5. Bounding Box Formatting & Coordinate Clipping
# ============================================================================

def test_bounding_box_format_and_clipping():
    """Verify bounding box coordinates are clipped to frame boundary and invalid boxes discarded."""
    width, height = 1280, 720
    raw_data = np.array([
        # Box partially off the left & top edges
        [-30.0, -20.0, 150.0, 300.0, 0.80, 0],
        # Box partially off the right & bottom edges
        [1100.0, 600.0, 1350.0, 800.0, 0.75, 0],
        # Completely out of bounds (should be dropped)
        [-100.0, -100.0, -10.0, -10.0, 0.90, 0],
        # Inverted / Degenerate box (x2 <= x1)
        [200.0, 100.0, 150.0, 300.0, 0.85, 0],
        # Zero area box (y2 == y1)
        [300.0, 200.0, 400.0, 200.0, 0.85, 0],
        # Perfectly valid normal box
        [400.0, 200.0, 500.0, 450.0, 0.92, 0],
    ], dtype=np.float32)

    detector = PersonDetector(auto_load=False)
    detector.set_model(MockYOLOModel(predictions=[raw_data]))
    detector.device = "cpu"

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    detections = detector.detect(frame)

    # Only 3 valid boxes should survive
    assert len(detections) == 3

    # First box clipped to (0, 0, 150, 300)
    assert detections[0].bbox == (0.0, 0.0, 150.0, 300.0)

    # Second box clipped to (1100, 600, 1280, 720)
    assert detections[1].bbox == (1100.0, 600.0, 1280.0, 720.0)

    # Third box unchanged
    assert detections[2].bbox == (400.0, 200.0, 500.0, 450.0)


# ============================================================================
# 6. Batched Detection Tests
# ============================================================================

def test_batched_detection():
    """Verify batched detection processes multiple frames and returns List[List[PersonDetection]]."""
    frame1_boxes = np.array([
        [10, 10, 50, 100, 0.80, 0],
        [60, 20, 110, 120, 0.85, 0],
    ], dtype=np.float32)

    frame2_boxes = np.array([
        [100, 100, 200, 300, 0.90, 0],
        [250, 100, 350, 300, 0.88, 0],
        [400, 100, 500, 300, 0.75, 0],
    ], dtype=np.float32)

    frame3_boxes = np.empty((0, 6), dtype=np.float32)

    detector = PersonDetector(auto_load=False)
    detector.set_model(MockYOLOModel(predictions=[frame1_boxes, frame2_boxes, frame3_boxes]))
    detector.device = "cpu"

    f1 = np.zeros((480, 640, 3), dtype=np.uint8)
    f2 = np.zeros((720, 1280, 3), dtype=np.uint8)
    f3 = np.zeros((480, 640, 3), dtype=np.uint8)

    batch_results = detector.detect_batch([f1, f2, f3])
    assert len(batch_results) == 3
    assert len(batch_results[0]) == 2
    assert len(batch_results[1]) == 3
    assert len(batch_results[2]) == 0

    # Empty batch returns empty list
    assert detector.detect_batch([]) == []


# ============================================================================
# 7. Adaptive Resolution Scaling Tests
# ============================================================================

def test_adaptive_resolution_latency_scaling():
    """Verify adaptive inference resolution adjusts to maintain frame budget."""
    detector = PersonDetector(
        config={
            "resolution": 640,
            "min_resolution": 480,
            "max_resolution": 960,
            "target_ms": 100.0,
            "adaptive": True,
        },
        auto_load=False
    )
    assert detector.resolution == 640

    # Simulate heavy latency > target_ms * 1.2 (120ms) for 10 frames
    for _ in range(10):
        detector.adapt(180.0)

    # Resolution should have decreased by 64 (640 - 64 = 576)
    assert detector.resolution < 640
    assert detector.resolution % 32 == 0

    # Simulate very light latency < target_ms * 0.65 (65ms) for 10 frames
    for _ in range(10):
        detector.adapt(30.0)

    # Resolution should scale back up towards configured resolution
    assert detector.resolution % 32 == 0

    # When adaptive is False, resolution remains steady
    non_adaptive = PersonDetector(
        config={"resolution": 640, "adaptive": False},
        auto_load=False
    )
    for _ in range(20):
        non_adaptive.adapt(200.0)
    assert non_adaptive.resolution == 640


# ============================================================================
# 8. Coordination with detector.py Patterns
# ============================================================================

def test_coordination_with_detector_boxes():
    """Verify PersonDetector can directly consume detector.people_boxes from outputs/services/detector.py."""
    data = np.array([
        [100, 150, 200, 350, 0.85, 0],
        [300, 200, 400, 450, 0.72, 0],
        [50, 50, 80, 120, 0.10, 0],    # Low confidence
        [500, 100, 600, 200, 0.90, 2],  # Car (non-person class)
    ], dtype=np.float32)

    mock_boxes = MockBoxes(data, (720, 1280))

    # from_detector_boxes with 0.15 threshold
    detections = PersonDetector.from_detector_boxes(
        mock_boxes,
        image_shape=(720, 1280),
        confidence_threshold=0.15
    )

    assert len(detections) == 2
    assert all(isinstance(d, PersonDetection) for d in detections)
    assert detections[0].bbox == (100.0, 150.0, 200.0, 350.0)
    assert abs(detections[0].confidence - 0.85) < 1e-4
    assert detections[1].bbox == (300.0, 200.0, 400.0, 450.0)

    # from_detector wrapping an object with people_boxes attribute
    class MockDetector:
        people_boxes = mock_boxes

    det_from_service = PersonDetector.from_detector(
        MockDetector(),
        image_shape=(720, 1280),
        confidence_threshold=0.15
    )
    assert len(det_from_service) == 2


# ============================================================================
# 9. Fallback & Mock Verification
# ============================================================================

def test_backend_fallback_warning():
    """Verify fallback to PyTorch when unsupported backend runtime is missing."""
    detector = PersonDetector(
        config={"backend": "onnx", "fallback_to_mock": True},
        auto_load=False
    )
    detector.load()
    assert detector.backend == "pytorch"
    assert detector.warning is not None
    assert "onnx" in detector.warning.lower()


def test_mock_yolo_direct_usage():
    """Verify MockYOLO stand-in class returns appropriate results."""
    mock = MockYOLO()
    assert mock.names[0] == "person"
    results = mock.predict([np.zeros((100, 100, 3), dtype=np.uint8)])
    assert len(results) == 1
    assert len(results[0].boxes) == 0

    # Inject mock boxes
    custom_boxes = np.array([[10, 10, 50, 50, 0.9, 0]], dtype=np.float32)
    mock.set_boxes(custom_boxes)
    results2 = mock.predict(np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(results2) == 1
    assert len(results2[0].boxes) == 1


# ============================================================================
# 10. Official Model Weights Smoke Tests (If present)
# ============================================================================

@pytest.mark.parametrize("model_name", ["yolo26n", "yolo26s"])
def test_official_weights_smoke_test(model_name):
    """Run smoke test on official weights if present in outputs/models."""
    models_dir = Path(__file__).resolve().parents[1] / "models"
    weights_path = models_dir / f"{model_name}.pt"
    if not weights_path.exists():
        pytest.skip(f"Weights file {weights_path} not found")

    detector = PersonDetector(
        models_dir=models_dir,
        config={
            "model_name": model_name,
            "confidence_threshold": 0.15,
            "resolution": 640,
            "adaptive": False,
        }
    )

    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(test_frame)

    assert isinstance(detections, list)
    assert detector.model is not None
    assert "batch_total_ms" in detector.timings


# ============================================================================
# 11. UHD / 4K Stream High-Resolution & Low-Threshold Tests
# ============================================================================

def test_default_confidence_threshold_is_0_10():
    """Verify default confidence threshold is 0.10 to capture distant individuals."""
    cfg = PersonDetectorConfig()
    assert cfg.confidence_threshold == 0.10
    detector = PersonDetector(auto_load=False)
    assert detector.config.confidence_threshold == 0.10


def test_uhd_4k_frame_adaptive_resolution_and_floor():
    """Verify UHD/4K frames (width >= 1920 or height >= 1080) use at least 960 and adapt up to 1280."""
    detector = PersonDetector(
        config={"resolution": 640, "adaptive": True, "target_ms": 100.0},
        auto_load=False
    )
    mock_model = MockYOLOModel()
    detector.set_model(mock_model)
    detector.device = "cpu"

    # Input is 4K frame (3840x2160)
    uhd_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    detector.detect(uhd_frame)

    assert detector.is_uhd is True
    # Initial UHD resolution should be elevated to at least 960
    assert detector.resolution >= 960

    # Under light latency (< target_ms * 0.65), adapt should scale resolution up to 1280
    for _ in range(60):
        detector.adapt(20.0)

    assert detector.resolution == 1280

    # Under heavy latency (> target_ms * 1.2), adapt should NOT scale down below 960 on UHD
    for _ in range(60):
        detector.adapt(250.0)

    assert detector.resolution >= 960


def test_detector_py_uhd_imgsz_and_person_confidence_08():
    """Verify detector.py ensures imgsz >= 960 on UHD and person_confidence defaults to 0.08."""
    from services.detector import Detector, validate_settings, DEFAULTS
    detector = Detector(Path(__file__).resolve().parents[1] / "models")

    called_kwargs = []
    class MockDetectorModel:
        names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        def predict(self, image, **kwargs):
            called_kwargs.append(kwargs)
            raw_data = np.array([
                [100, 100, 150, 200, 0.09, 0],  # Person with 0.09 conf (should be kept by 0.08 default)
                [300, 300, 350, 400, 0.07, 0],  # Person with 0.07 conf (should be dropped)
            ], dtype=np.float32)
            mock_res = type('Res', (), {'boxes': MockBoxes(raw_data, (2160, 3840)), 'speed': {}})()
            return [mock_res]

    detector.model = MockDetectorModel()
    settings = validate_settings({**DEFAULTS, 'resolution': 640, 'adaptive': True})

    # 4K frame
    uhd_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    detector.predict(uhd_frame, settings, {'items': []})

    # imgsz passed to model.predict must be at least 960
    assert len(called_kwargs) == 1
    assert called_kwargs[0]['imgsz'] >= 960
    assert detector.is_uhd is True

    # Person with 0.09 confidence must be kept
    assert len(detector.people_boxes) == 1
    assert abs(float(detector.people_boxes.conf[0]) - 0.09) < 1e-4

    # Adapt must not reduce resolution below 960 on UHD
    detector.resolution = 960
    for _ in range(40):
        detector.adapt(300.0, settings)
    assert detector.resolution >= 960
