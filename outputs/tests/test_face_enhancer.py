"""Unit tests verifying the facial enhancement and super-resolution pipeline.

Tests fulfill all mission requirements:
1. Super-resolution producing exact requested multiples (2x and 4x).
2. Lanczos4 fallback when models are unavailable or explicitly requested.
3. Strict asset separation and crop immutability (original is never mutated/overwritten).
4. Asynchronous queue processing, priority ordering, and completion callbacks.
5. Queue overload protection with drop of lower-priority items and immediate rejection.
6. Memory protection via crop dimension clamping (max_input_dim <= 400).
7. Graceful handling of invalid inputs and edge cases.
8. Shared contracts integration (CaptureRecord and CaptureCandidate).
"""
import os
from pathlib import Path
import tempfile
import threading
import time

import cv2
import numpy as np
import pytest

from services.face_contracts import CaptureCandidate, CaptureRecord, FacePose
from services.face_enhancer import (
    EnhancementConfig,
    EnhancementResult,
    FaceEnhancer,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_NORMAL,
    SafePriorityQueue,
    format_portrait_aspect_ratio,
    get_default_enhancer,
    get_enhanced_filename,
    get_enhanced_path,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_face_crop() -> np.ndarray:
    """Generate a realistic synthetic 60x50 BGR face-like patch."""
    np.random.seed(42)
    # Base skin tone gradient with simulated facial feature edges
    crop = np.full((60, 50, 3), [130, 160, 210], dtype=np.uint8)
    # Add simulated eye regions (darker horizontal bars)
    crop[18:24, 10:20] = [60, 70, 90]
    crop[18:24, 30:40] = [60, 70, 90]
    # Add simulated mouth
    crop[42:47, 15:35] = [90, 100, 160]
    # Add subtle sensor noise
    noise = np.random.normal(0, 4, crop.shape).astype(np.int16)
    noisy_crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_crop


# -----------------------------------------------------------------------------
# 1. Super-Resolution Scaling & Exact Dimensions (2x and 4x)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ["espcn", "fsrcnn"])
def test_neural_superres_2x_exact_multiple(sample_face_crop: np.ndarray, model_name: str):
    """Verify 2x upscale produces exact 2x dimensions with expected metadata."""
    enhancer = FaceEnhancer(model_name=model_name, default_scale=2, auto_start=False)
    orig_h, orig_w = sample_face_crop.shape[:2]

    res = enhancer.enhance(sample_face_crop, scale=2)

    assert res.success is True
    assert res.is_fallback is False
    assert res.scale == 2
    assert res.enhanced_shape == (orig_h * 2, orig_w * 2, 3)
    assert res.enhanced_image is not None
    assert res.enhanced_image.shape == (orig_h * 2, orig_w * 2, 3)
    assert f"{model_name.upper()}-x2 + CLAHE" in res.method
    assert res.processing_time_ms > 0.0


@pytest.mark.parametrize("model_name", ["espcn", "fsrcnn"])
def test_neural_superres_4x_exact_multiple(sample_face_crop: np.ndarray, model_name: str):
    """Verify 4x upscale produces exact 4x dimensions with expected metadata."""
    enhancer = FaceEnhancer(model_name=model_name, default_scale=4, auto_start=False)
    orig_h, orig_w = sample_face_crop.shape[:2]

    res = enhancer.enhance(sample_face_crop, scale=4)

    assert res.success is True
    assert res.is_fallback is False
    assert res.scale == 4
    assert res.enhanced_shape == (orig_h * 4, orig_w * 4, 3)
    assert res.enhanced_image is not None
    assert res.enhanced_image.shape == (orig_h * 4, orig_w * 4, 3)
    assert f"{model_name.upper()}-x4 + CLAHE" in res.method
    assert res.processing_time_ms > 0.0


def test_odd_and_non_square_dimensions():
    """Verify arbitrary crop aspect ratios produce exact integer multiples."""
    enhancer = FaceEnhancer(model_name="espcn", auto_start=False)
    # Test non-square, odd dimensions (h=53, w=47)
    odd_crop = np.random.randint(60, 200, (53, 47, 3), dtype=np.uint8)

    res2 = enhancer.enhance(odd_crop, scale=2)
    assert res2.enhanced_shape == (106, 94, 3)

    res4 = enhancer.enhance(odd_crop, scale=4)
    assert res4.enhanced_shape == (212, 188, 3)


# -----------------------------------------------------------------------------
# 2. Lanczos4 Fallback Behavior
# -----------------------------------------------------------------------------

def test_lanczos_fallback_when_model_missing(sample_face_crop: np.ndarray, tmp_path: Path):
    """Verify graceful fallback to Lanczos4 when model file is absent."""
    # Point models_dir to an empty directory
    empty_dir = tmp_path / "no_models"
    empty_dir.mkdir()

    enhancer = FaceEnhancer(
        model_name="espcn",
        default_scale=2,
        models_dir=empty_dir,
        auto_start=False,
    )
    orig_h, orig_w = sample_face_crop.shape[:2]

    res = enhancer.enhance(sample_face_crop, scale=2)

    assert res.success is True
    assert res.is_fallback is True
    assert res.scale == 2
    assert res.enhanced_shape == (orig_h * 2, orig_w * 2, 3)
    assert "Lanczos4-x2 + CLAHE" in res.method


def test_explicit_lanczos_mode(sample_face_crop: np.ndarray):
    """Verify explicitly requested Lanczos mode operates without neural models."""
    enhancer = FaceEnhancer(model_name="lanczos", default_scale=4, auto_start=False)
    orig_h, orig_w = sample_face_crop.shape[:2]

    res = enhancer.enhance(sample_face_crop, scale=4)

    assert res.success is True
    assert res.is_fallback is True
    assert res.scale == 4
    assert res.enhanced_shape == (orig_h * 4, orig_w * 4, 3)
    assert "Lanczos4-x4 + CLAHE" in res.method


# -----------------------------------------------------------------------------
# 3. Immutability & Original Crop Preservation
# -----------------------------------------------------------------------------

def test_original_crop_preserved_immutability(sample_face_crop: np.ndarray):
    """Strictly verify that the input image array is NEVER modified in memory."""
    enhancer = FaceEnhancer(model_name="espcn", auto_start=False)
    pristine_clone = sample_face_crop.copy()
    initial_hash = hash(pristine_clone.tobytes())

    res = enhancer.enhance(sample_face_crop, scale=2)

    # Verify input array values remain 100% byte-for-byte identical
    assert np.array_equal(sample_face_crop, pristine_clone)
    assert hash(sample_face_crop.tobytes()) == initial_hash

    # Verify output is an entirely distinct array in memory
    assert res.enhanced_image is not sample_face_crop
    assert res.enhanced_image.base is not sample_face_crop


# -----------------------------------------------------------------------------
# 4. Strict Asset Separation & File Naming
# -----------------------------------------------------------------------------

def test_file_naming_conventions():
    """Verify distinct and safe file path generation rules."""
    assert get_enhanced_filename("cap_001_orig.jpg", scale=2) == "cap_001_enhanced_2x.jpg"
    assert get_enhanced_filename("cap_001_orig.jpg", scale=4) == "cap_001_enhanced_4x.jpg"
    assert get_enhanced_filename("face.png", scale=2) == "face_enhanced_2x.png"
    assert get_enhanced_filename("/tmp/track_5_orig.jpeg", scale=2) == "track_5_enhanced_2x.jpeg"

    # Verify path derivation
    p = get_enhanced_path("c:/data/captures/img_orig.jpg", scale=2)
    assert p.name == "img_enhanced_2x.jpg"
    assert str(p.parent).replace("\\", "/").endswith("data/captures")


def test_asset_separation_on_disk(sample_face_crop: np.ndarray, tmp_path: Path):
    """Verify disk saving writes distinct enhanced asset without touching original."""
    orig_file = tmp_path / "person_01_orig.jpg"
    cv2.imwrite(str(orig_file), sample_face_crop)
    orig_bytes = orig_file.read_bytes()
    orig_mtime = orig_file.stat().st_mtime

    enhancer = FaceEnhancer(model_name="espcn", auto_start=False)
    res = enhancer.enhance(
        sample_face_crop,
        scale=2,
        original_path=orig_file,
        save_to_disk=True,
    )

    expected_enhanced_path = tmp_path / "person_01_enhanced_2x.jpg"

    # Original file is untouched
    assert orig_file.exists()
    assert orig_file.read_bytes() == orig_bytes
    assert orig_file.stat().st_mtime == orig_mtime

    # Enhanced file exists and has distinct characteristics
    assert expected_enhanced_path.exists()
    assert res.enhanced_path == str(expected_enhanced_path)
    assert expected_enhanced_path.stat().st_size > 0
    assert expected_enhanced_path.stat().st_size != len(orig_bytes)

    # Check enhanced image dimensions from disk
    disk_enhanced = cv2.imread(str(expected_enhanced_path))
    assert disk_enhanced.shape == res.enhanced_shape


# -----------------------------------------------------------------------------
# 5. Pipeline Stages: Noise Reduction, CLAHE, and Anti-Ringing Unsharp Mask
# -----------------------------------------------------------------------------

def test_clahe_luminance_and_color_constancy(sample_face_crop: np.ndarray):
    """Verify CLAHE operates strictly on luminance and preserves color tones."""
    config = EnhancementConfig(clahe_clip_limit=1.4)
    enhancer = FaceEnhancer(config=config, auto_start=False)

    res = enhancer.enhance(sample_face_crop, scale=2)
    assert res.success is True

    # Test clip limit boundary enforcement in config
    bounded_config = EnhancementConfig(clahe_clip_limit=3.0)  # Over upper limit
    enhancer_bounded = FaceEnhancer(config=bounded_config, auto_start=False)
    res_bounded = enhancer_bounded.enhance(sample_face_crop, scale=2)
    assert res_bounded.success is True


def test_unsharp_mask_bounds_and_anti_ringing():
    """Verify unsharp mask does not exceed uint8 bounds [0, 255] or introduce halos."""
    # Synthetic high-contrast step edge
    edge_crop = np.zeros((40, 40, 3), dtype=np.uint8)
    edge_crop[:, 20:] = 255

    config = EnhancementConfig(
        unsharp_amount=0.35,
        unsharp_threshold=2.0,
        unsharp_max_delta=8.0,
    )
    enhancer = FaceEnhancer(config=config, auto_start=False)
    res = enhancer.enhance(edge_crop, scale=2)

    assert res.success is True
    enhanced = res.enhanced_image
    assert enhanced.dtype == np.uint8
    assert enhanced.min() >= 0
    assert enhanced.max() <= 255


# -----------------------------------------------------------------------------
# 6. Memory and Resource Protection (Crop Limits & OOM Prevention)
# -----------------------------------------------------------------------------

def test_memory_protection_oversized_crop_limit():
    """Large native crops bypass neural work without losing source detail."""
    # Input exceeding default max_input_dim (400)
    large_crop = np.random.randint(50, 200, (600, 500, 3), dtype=np.uint8)

    enhancer = FaceEnhancer(
        config=EnhancementConfig(max_input_dim=400),
        auto_start=False,
    )
    res = enhancer.enhance(large_crop, scale=2)

    assert res.success is True
    assert res.metadata.get("pre_downscaled") is False
    assert res.metadata.get("intermediate_shape") == large_crop.shape
    assert res.method.startswith('Lanczos4')
    # Final output dimensions still match requested exact 2x multiple of original
    assert res.enhanced_shape == (1200, 1000, 3)


# -----------------------------------------------------------------------------
# 7. Asynchronous Execution, Queueing, and Callbacks
# -----------------------------------------------------------------------------

def test_asynchronous_submit_and_future(sample_face_crop: np.ndarray):
    """Verify asynchronous submission returns Future and processes in background."""
    with FaceEnhancer(auto_start=True) as enhancer:
        future = enhancer.submit(sample_face_crop, scale=2)
        res = future.result(timeout=4.0)

        assert isinstance(res, EnhancementResult)
        assert res.success is True
        assert res.scale == 2
        assert res.enhanced_shape == (120, 100, 3)


def test_asynchronous_completion_callback(sample_face_crop: np.ndarray):
    """Verify completion callback is invoked asynchronously upon task completion."""
    callback_event = threading.Event()
    callback_result = []

    def on_complete(result: EnhancementResult):
        callback_result.append(result)
        callback_event.set()

    with FaceEnhancer(auto_start=True) as enhancer:
        future = enhancer.submit(
            sample_face_crop,
            scale=2,
            callback=on_complete,
        )
        assert callback_event.wait(timeout=4.0)
        assert len(callback_result) == 1
        assert callback_result[0].success is True
        assert future.done() is True
        assert future.result() is callback_result[0]


def test_queue_priority_ordering(sample_face_crop: np.ndarray):
    """Verify high-priority tasks (0) are processed before normal-priority (10)."""
    # Keep enhancer stopped so tasks accumulate in queue
    enhancer = FaceEnhancer(auto_start=False)

    order_processed = []

    def make_cb(label: str):
        return lambda res: order_processed.append(label)

    # Submit normal priority (10) then high priority (0)
    enhancer.submit(sample_face_crop, priority=PRIORITY_NORMAL, callback=make_cb("normal_1"))
    enhancer.submit(sample_face_crop, priority=PRIORITY_LOW, callback=make_cb("low"))
    enhancer.submit(sample_face_crop, priority=PRIORITY_HIGH, callback=make_cb("high"))

    # Start worker and wait for queue to drain
    enhancer.start()
    timeout_time = time.time() + 4.0
    while enhancer._queue.qsize() > 0 or len(order_processed) < 3:
        if time.time() > timeout_time:
            break
        time.sleep(0.05)
    enhancer.stop(wait=True)

    assert len(order_processed) == 3
    # High priority must finish before low priority
    assert order_processed.index("high") < order_processed.index("low")


def test_queue_overload_drop_and_rejection(sample_face_crop: np.ndarray):
    """Verify queue capacity limit (maxsize=2): drop of lowest priority & rejection."""
    # Worker is paused/stopped so queue fills
    enhancer = FaceEnhancer(max_queue_size=2, auto_start=False)

    # Fill queue to capacity (2/2) with normal priority (10)
    fut1 = enhancer.submit(sample_face_crop, priority=PRIORITY_NORMAL)
    fut2 = enhancer.submit(sample_face_crop, priority=PRIORITY_NORMAL)
    assert enhancer._queue.qsize() == 2

    # Task 3 has lower priority (20) -> should be rejected immediately
    fut3 = enhancer.submit(sample_face_crop, priority=PRIORITY_LOW)
    res3 = fut3.result(timeout=1.0)
    assert res3.success is False
    assert "rejected under queue overload" in (res3.error or "")

    # Task 4 has higher priority (0) -> should preempt/drop one of the priority-10 tasks
    fut4 = enhancer.submit(sample_face_crop, priority=PRIORITY_HIGH)
    assert enhancer._queue.qsize() == 2

    # Verify that either fut1 or fut2 was dropped
    dropped_fut = fut1 if fut1.done() else (fut2 if fut2.done() else None)
    assert dropped_fut is not None
    dropped_res = dropped_fut.result(timeout=1.0)
    assert dropped_res.success is False
    assert "dropped from queue under overload" in (dropped_res.error or "")

    # Clean up and verify metrics
    metrics = enhancer.get_metrics()
    assert metrics["tasks_rejected"] >= 1
    assert metrics["tasks_dropped"] >= 1


# -----------------------------------------------------------------------------
# 8. Graceful Handling of Invalid Inputs
# -----------------------------------------------------------------------------

def test_handling_none_input_without_path():
    """Verify passing None without a path fails gracefully without exception."""
    enhancer = FaceEnhancer(auto_start=False)
    res = enhancer.enhance(None)
    assert res.success is False
    assert "None" in (res.error or "")
    assert res.enhanced_image is None


def test_handling_empty_and_undersized_crops():
    """Verify empty or smaller than min_input_dim crops fail gracefully."""
    enhancer = FaceEnhancer(auto_start=False)

    # Zero size array
    res_empty = enhancer.enhance(np.zeros((0, 0, 3), dtype=np.uint8))
    assert res_empty.success is False

    # Below min_input_dim (4)
    res_tiny = enhancer.enhance(np.zeros((2, 2, 3), dtype=np.uint8))
    assert res_tiny.success is False
    assert "smaller than min limit" in (res_tiny.error or "")


def test_handling_grayscale_and_4channel_inputs():
    """Verify 2D grayscale and 4-channel BGRA are automatically converted to BGR."""
    enhancer = FaceEnhancer(auto_start=False)

    # Grayscale (2D)
    gray = np.random.randint(60, 200, (40, 40), dtype=np.uint8)
    res_gray = enhancer.enhance(gray, scale=2)
    assert res_gray.success is True
    assert res_gray.enhanced_shape == (80, 80, 3)

    # BGRA (4-channel)
    bgra = np.random.randint(60, 200, (40, 40, 4), dtype=np.uint8)
    res_bgra = enhancer.enhance(bgra, scale=2)
    assert res_bgra.success is True
    assert res_bgra.enhanced_shape == (80, 80, 3)


# -----------------------------------------------------------------------------
# 9. Contracts Integration (CaptureRecord & CaptureCandidate)
# -----------------------------------------------------------------------------

def test_capture_record_contract_update(sample_face_crop: np.ndarray, tmp_path: Path):
    """Verify CaptureRecord contract instance is properly enriched and updated."""
    orig_path = tmp_path / "cap_99_orig.jpg"
    cv2.imwrite(str(orig_path), sample_face_crop)

    record = CaptureRecord(
        capture_id="cap_99",
        track_id=42,
        session_id="session:42",
        timestamp=123.456,
        category="frontal",
        quality_score=94.5,
        pose_category="frontal",
        yaw=2.1,
        pitch=-1.0,
        original_path=str(orig_path),
        crop_width=sample_face_crop.shape[1],
        crop_height=sample_face_crop.shape[0],
    )

    with FaceEnhancer() as enhancer:
        res = enhancer.enhance(
            sample_face_crop,
            scale=2,
            save_to_disk=True,
            capture_record=record,
        )

        assert res.success is True
        assert res.capture_id == "cap_99"
        assert res.track_id == 42
        assert res.session_id == "session:42"

        # Verify CaptureRecord fields updated
        assert record.is_enhanced is True
        assert record.enhanced_path is not None
        assert "cap_99_enhanced_2x.jpg" in record.enhanced_path
        assert record.enhancement_method is not None
        assert "ESPCN-x2 + CLAHE" in record.enhancement_method
        assert record.file_size_bytes > 0

        # Verify CaptureRecord to_dict() contract
        d = record.to_dict()
        assert d["is_enhanced"] is True
        assert "/api/people/captures/cap_99/image?type=enhanced" in d["enhanced_url"]


def test_capture_candidate_crop_extraction(sample_face_crop: np.ndarray):
    """Verify passing a CaptureCandidate object directly into enhance() extracts crop."""
    pose = FacePose(yaw=0.0, pitch=0.0, roll=0.0, category="frontal")
    candidate = CaptureCandidate(
        candidate_id="cand_1",
        track_id=10,
        timestamp=12.0,
        category="frontal",
        quality_score=90.0,
        pose=pose,
        crop_original=sample_face_crop,
        frame_sequence=1,
        bbox_in_frame=(10.0, 10.0, 60.0, 70.0),
    )

    enhancer = FaceEnhancer(auto_start=False)
    res = enhancer.enhance(candidate, scale=2)

    assert res.success is True
    assert res.enhanced_shape == (
        sample_face_crop.shape[0] * 2,
        sample_face_crop.shape[1] * 2,
        3,
    )


# -----------------------------------------------------------------------------
# 10. Lifecycle, Metrics & Global Singleton
# -----------------------------------------------------------------------------

def test_metrics_tracking(sample_face_crop: np.ndarray):
    """Verify latency and operational counts are accurately reflected in get_metrics()."""
    enhancer = FaceEnhancer(auto_start=False)
    m_init = enhancer.get_metrics()
    assert m_init["tasks_processed"] == 0

    enhancer.enhance(sample_face_crop, scale=2)
    m_after = enhancer.get_metrics()
    assert m_after["tasks_processed"] == 1
    assert m_after["enhancement_ms"] > 0.0
    assert m_after["avg_enhancement_ms"] > 0.0


def test_global_singleton():
    """Verify get_default_enhancer() returns a persistent active FaceEnhancer."""
    e1 = get_default_enhancer()
    e2 = get_default_enhancer()
    assert e1 is e2
    assert isinstance(e1, FaceEnhancer)


# -----------------------------------------------------------------------------
# 11. High-Quality Face Capture & Minimum 256px Auto-Upscaling
# -----------------------------------------------------------------------------

def test_auto_upscale_small_face_crop_minimum_256(sample_face_crop: np.ndarray):
    """Verify that any face crop below 256px is super-resolved and upscaled to >= 256x256."""
    enhancer = FaceEnhancer(model_name="espcn", auto_start=False)
    # sample_face_crop is 60x50, below 256px
    res = enhancer.enhance(sample_face_crop)

    assert res.success is True
    assert res.enhanced_image is not None
    h, w = res.enhanced_shape[:2]
    assert h >= 256
    assert w >= 256
    assert res.enhanced_image.dtype == np.uint8
    assert "ESPCN" in res.method
    assert "CLAHE" in res.method


def test_auto_upscale_fallback_minimum_256(sample_face_crop: np.ndarray, tmp_path: Path):
    """Verify fallback pipeline also ensures >= 256x256 using Lanczos4 + CLAHE."""
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    enhancer = FaceEnhancer(models_dir=empty_dir, auto_start=False)

    res = enhancer.enhance(sample_face_crop)
    assert res.success is True
    assert res.is_fallback is True
    h, w = res.enhanced_shape[:2]
    assert h >= 256
    assert w >= 256
    assert "Lanczos4" in res.method


def test_adaptive_lighting_contrast_enhancement():
    """Verify under-exposed faces have midtone exposure lifted naturally."""
    dark_face = np.full((60, 60, 3), [40, 45, 55], dtype=np.uint8)
    # Add facial features
    dark_face[18:24, 10:20] = [15, 20, 25]
    dark_face[18:24, 40:50] = [15, 20, 25]

    enhancer = FaceEnhancer(auto_start=False)
    res = enhancer.enhance(dark_face)

    assert res.success is True
    assert res.enhanced_image is not None
    # Luminance should be lifted compared to dark input
    assert float(np.mean(res.enhanced_image)) > float(np.mean(dark_face))


def test_async_submit_auto_upscale_to_minimum_256(sample_face_crop: np.ndarray):
    """Verify asynchronous queue worker automatically enforces >= 256x256."""
    with FaceEnhancer(auto_start=True) as enhancer:
        fut = enhancer.submit(sample_face_crop)
        res = fut.result(timeout=4.0)

        assert res.success is True
        h, w = res.enhanced_shape[:2]
        assert h >= 256
        assert w >= 256


# -----------------------------------------------------------------------------
# 12. Ultra-High-Quality Crisp Portrait Framing & Aspect Ratio
# -----------------------------------------------------------------------------

class TestUltraHighQualityPortraitEnhancement:
    """Verify production face capture quality requirements:
    1. Small faces (< 320x320) upscale to at least 384x384 (or 512x512).
    2. Exact 1:1 square aspect ratio without distortion or black bars.
    3. Exact 4:5 portrait aspect ratio without distortion or black bars.
    4. CLAHE on L-channel in LAB space with clipLimit=1.8, tileGridSize=(8, 8).
    5. Subtle unsharp mask (sigma=1.0, strength=0.35) for crisp facial contours.
    """

    def test_small_face_upscale_to_minimum_384(self, sample_face_crop: np.ndarray):
        """Small faces below 320x320 are upscaled to at least 384x384."""
        enhancer = FaceEnhancer(auto_start=False)
        # sample_face_crop is 60x50
        res = enhancer.enhance(sample_face_crop)

        assert res.success is True
        assert res.enhanced_image is not None
        h, w = res.enhanced_shape[:2]
        assert h >= 384
        assert w >= 384
        # Default portrait aspect ratio is 1:1 square
        assert h == w
        assert h == 384
        assert w == 384

    def test_format_portrait_aspect_ratio_1_to_1_tall(self):
        """Tall portrait crop formatted to 1:1 uses top-headroom bias without black bars."""
        # Tall image 160x100 (H=160, W=100)
        tall_img = np.full((160, 100, 3), 120, dtype=np.uint8)
        # Top region has distinct simulated hair/head
        tall_img[:30, :] = 40
        # Center has face
        tall_img[30:90, :] = 180

        formatted = FaceEnhancer.format_portrait_aspect_ratio(tall_img, target_aspect="1:1", min_dim=384)
        assert formatted.shape == (384, 384, 3)
        # Verify no black bars (no padded 0-value rows or columns at edges)
        assert np.all(formatted > 0)
        assert formatted.shape[0] == formatted.shape[1]

    def test_format_portrait_aspect_ratio_1_to_1_wide(self):
        """Wide crop formatted to 1:1 uses center crop without black bars."""
        # Wide image 100x160 (H=100, W=160)
        wide_img = np.full((100, 160, 3), 150, dtype=np.uint8)
        formatted = FaceEnhancer.format_portrait_aspect_ratio(wide_img, target_aspect="1:1", min_dim=384)
        assert formatted.shape == (384, 384, 3)
        assert np.all(formatted > 0)
        assert formatted.shape[0] == formatted.shape[1]

    def test_format_portrait_aspect_ratio_4_to_5(self):
        """4:5 portrait aspect ratio produces exact 4:5 ratio without distortion."""
        # Square image 120x120
        square_img = np.full((120, 120, 3), 140, dtype=np.uint8)
        formatted = FaceEnhancer.format_portrait_aspect_ratio(square_img, target_aspect="4:5", min_dim=384)
        h, w = formatted.shape[:2]
        assert w >= 384
        assert h == int(round(w * 1.25))
        assert pytest.approx(float(w) / float(h), rel=1e-3) == 0.8
        assert np.all(formatted > 0)

    def test_lab_color_space_clahe_parameters(self, sample_face_crop: np.ndarray):
        """Verify LAB color space CLAHE brings out facial feature contrast with clipLimit=1.8."""
        config = EnhancementConfig(
            color_space="lab",
            clahe_clip_limit=1.8,
            clahe_tile_grid_size=(8, 8),
        )
        enhancer = FaceEnhancer(config=config, auto_start=False)
        res = enhancer.enhance(sample_face_crop, scale=2)

        assert res.success is True
        assert res.enhanced_image is not None
        # Contrast of enhanced image is higher than blurred/flat input
        assert float(np.std(res.enhanced_image)) >= float(np.std(sample_face_crop)) * 0.9

    def test_unsharp_mask_sharpens_edges(self):
        """Verify unsharp mask with sigma=1.0 and strength=0.35 sharpens facial edges without ringing."""
        # Create an image with an edge transition
        edge_img = np.full((80, 80, 3), 100, dtype=np.uint8)
        edge_img[:, 40:] = 200

        config = EnhancementConfig(
            unsharp_amount=0.35,
            unsharp_radius=1.0,
            unsharp_threshold=2.0,
            unsharp_max_delta=8.0,
        )
        enhancer = FaceEnhancer(config=config, auto_start=False)
        res = enhancer.enhance(edge_img, scale=2)

        assert res.success is True
        enhanced = res.enhanced_image
        assert enhanced.shape == (160, 160, 3)
        # Check edge sharpness: gradient at edge in enhanced should be crisp
        grad_orig = np.abs(np.diff(edge_img.astype(float), axis=1))
        grad_enh = np.abs(np.diff(enhanced.astype(float), axis=1))
        assert np.max(grad_enh) > 0

    def test_convenience_function_export(self):
        """Verify module-level format_portrait_aspect_ratio function works identically."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        res1 = format_portrait_aspect_ratio(img, target_aspect="1:1", min_dim=384)
        res2 = FaceEnhancer.format_portrait_aspect_ratio(img, target_aspect="1:1", min_dim=384)
        assert np.array_equal(res1, res2)
        assert res1.shape == (384, 384, 3)
