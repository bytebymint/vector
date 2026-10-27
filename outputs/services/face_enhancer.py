"""Asynchronous Facial Enhancement and Super-Resolution Pipeline.

Supports 2x and 4x super-resolution using OpenCV DnnSuperRes (ESPCN, FSRCNN)
blended with Lanczos4 (0.85 * SR + 0.15 * Lanczos4), gentle edge-preserving
bilateral luma/chroma denoising, adaptive highlight-protected shadow lift,
subtle CIE-LAB L-channel CLAHE, clamped halo-free luminance micro-sharpening,
and strict asset separation.
"""
from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
import heapq
import itertools
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from services.face_contracts import CaptureRecord
except ImportError:
    try:
        from .face_contracts import CaptureRecord
    except ImportError:
        CaptureRecord = Any  # type: ignore

logger = logging.getLogger("face_enhancer")

# Standard priority levels for facial enhancement tasks
PRIORITY_HIGH: int = 0      # Best frontal captures, interactive user requests
PRIORITY_NORMAL: int = 10   # Standard quality captures
PRIORITY_LOW: int = 20      # Continuous / non-best captures, batch work


def get_enhanced_path(
    original_path: Union[str, Path],
    scale: int = 2,
    suffix: Optional[str] = None
) -> Path:
    """Derive distinct enhanced file path without mutating or overwriting the original.

    Transforms e.g. 'cap_1_orig.jpg' -> 'cap_1_enhanced_2x.jpg'
    or 'face.png' -> 'face_enhanced_4x.png'.
    """
    orig = Path(original_path)
    ext = orig.suffix or ".jpg"
    tag = f"_enhanced_{scale}x" if suffix is None else suffix

    stem = orig.stem
    if stem.endswith("_orig"):
        new_stem = stem[:-5] + tag
    else:
        new_stem = stem + tag

    enhanced = orig.with_name(f"{new_stem}{ext}")
    # Safeguard: never collide with original path
    if enhanced == orig:
        enhanced = orig.with_name(f"{new_stem}_enh{ext}")
    return enhanced


def get_enhanced_filename(original_path: Union[str, Path], scale: int = 2) -> str:
    """Return the base filename for the enhanced asset."""
    return get_enhanced_path(original_path, scale=scale).name


@dataclass
class EnhancementConfig:
    """Configuration options for studio-grade facial enhancement and super-resolution."""
    model_name: str = "espcn"                  # "espcn", "fsrcnn", or "lanczos"
    scale: int = 2                             # 2 or 4
    min_output_dim: int = 384                  # Minimum output dimension for enhanced faces
    adaptive_exposure: bool = True             # Highlight-protected shadow lift for dark/backlit faces
    clahe_clip_limit: float = 1.5              # Subtle L-channel clip limit avoiding harsh local contrast
    clahe_tile_grid_size: Tuple[int, int] = (6, 6)
    clahe_blend_weight: float = 0.55           # Smooth L-channel CLAHE blend weight (0.55)
    color_space: str = "lab"                   # "lab" (recommended) or "ycrcb"
    portrait_aspect_ratio: str = "1:1"         # "1:1", "4:5", or "original"
    bilateral_d: int = 5                       # Compact neighborhood preserving eyelashes & skin texture
    bilateral_sigma_color: float = 10.0        # Gentle edge-preserving sigmaColor=10
    bilateral_sigma_space: float = 2.5         # Gentle edge-preserving sigmaSpace=2.5
    unsharp_amount: float = 0.38               # Natural luminance micro-contrast strength
    unsharp_radius: float = 0.95               # Fine micro-detail Gaussian sigma
    unsharp_threshold: float = 2.0             # Soft-coring deadband to suppress skin-grain ringing
    unsharp_max_delta: float = 7.0             # Clamping limit (7.0) to eliminate edge halos
    max_input_dim: int = 400                   # Crop limit before neural upscale to prevent OOM
    min_input_dim: int = 4                     # Minimum valid crop size
    blend_conventional: float = 0.15           # Hybrid SR blend: 0.85 * SR + 0.15 * Lanczos4
    models_dir: Optional[Union[str, Path]] = None
    save_to_disk: bool = False
    jpeg_quality: int = 95


@dataclass
class EnhancementResult:
    """Persisted and memory result of a facial enhancement execution."""
    enhanced_image: Optional[np.ndarray] = None
    original_shape: Optional[Tuple[int, int, int]] = None
    enhanced_shape: Optional[Tuple[int, int, int]] = None
    scale: int = 2
    method: str = ""
    processing_time_ms: float = 0.0
    original_path: Optional[str] = None
    enhanced_path: Optional[str] = None
    success: bool = True
    is_fallback: bool = False
    error: Optional[str] = None
    capture_id: Optional[str] = None
    track_id: Optional[int] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "is_fallback": self.is_fallback,
            "scale": self.scale,
            "method": self.method,
            "processing_time_ms": round(float(self.processing_time_ms), 2),
            "original_shape": list(self.original_shape) if self.original_shape else None,
            "enhanced_shape": list(self.enhanced_shape) if self.enhanced_shape else None,
            "original_path": str(self.original_path) if self.original_path else None,
            "enhanced_path": str(self.enhanced_path) if self.enhanced_path else None,
            "capture_id": self.capture_id,
            "track_id": self.track_id,
            "session_id": self.session_id,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class EnhancementTask:
    """Internal task descriptor for queued asynchronous enhancement."""
    priority: int
    image: Optional[np.ndarray]
    scale: Optional[int]
    model_name: Optional[str]
    capture_record: Optional[Any]
    original_path: Optional[str]
    save_to_disk: bool
    callback: Optional[Callable[[EnhancementResult], None]]
    future: Future
    submit_time: float = field(default_factory=time.time)
    min_dimension: Optional[int] = None
    aspect_ratio: Optional[str] = None


class SafePriorityQueue:
    """Thread-safe priority queue with capacity limiting and overload drop/rejection.

    Under overload (queue full):
    - Rejects tasks that have lower or equal priority than the lowest priority item in queue.
    - If incoming task has higher priority, drops the lowest priority pending item to make room.
    """
    def __init__(self, maxsize: int = 20):
        self.maxsize = max(1, int(maxsize))
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._heap: List[Tuple[int, int, EnhancementTask]] = []
        self._seq = itertools.count()

    def qsize(self) -> int:
        with self._lock:
            return len(self._heap)

    def empty(self) -> bool:
        with self._lock:
            return len(self._heap) == 0

    def put(
        self,
        task: EnhancementTask
    ) -> Tuple[str, Optional[EnhancementTask]]:
        """Add task to priority queue with overload handling.

        Returns:
            ("enqueued", None) if placed into queue.
            ("dropped", dropped_task) if placed by preempting a lower-priority task.
            ("rejected", task) if queue was full and task was rejected.
        """
        with self._lock:
            priority = task.priority
            seq = next(self._seq)

            if len(self._heap) >= self.maxsize:
                # Find the lowest priority item (highest priority number)
                max_p = max(item[0] for item in self._heap)
                if priority < max_p:
                    # Incoming task is more urgent; evict the lowest priority pending task
                    drop_idx = max(
                        range(len(self._heap)),
                        key=lambda i: (self._heap[i][0], self._heap[i][1])
                    )
                    dropped_entry = self._heap.pop(drop_idx)
                    heapq.heapify(self._heap)

                    # Push incoming task
                    heapq.heappush(self._heap, (priority, seq, task))
                    self._not_empty.notify()
                    return "dropped", dropped_entry[2]
                else:
                    # Queue is full of tasks with equal or higher priority
                    return "rejected", task
            else:
                heapq.heappush(self._heap, (priority, seq, task))
                self._not_empty.notify()
                return "enqueued", None

    def get(self, timeout: Optional[float] = None) -> Optional[EnhancementTask]:
        """Fetch highest-priority task (lowest priority number), blocking up to timeout."""
        with self._not_empty:
            end_time = None if timeout is None else time.time() + timeout
            while len(self._heap) == 0:
                if timeout is None:
                    self._not_empty.wait()
                else:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        return None
                    self._not_empty.wait(remaining)
            return heapq.heappop(self._heap)[2]

    def clear(self) -> List[EnhancementTask]:
        """Clear all pending tasks and return them."""
        with self._lock:
            tasks = [item[2] for item in self._heap]
            self._heap.clear()
            return tasks


class FaceEnhancer:
    """Asynchronous, studio-grade portrait enhancement and super-resolution engine.

    Features:
    - 2x and 4x super-resolution via OpenCV DnnSuperRes (ESPCN, FSRCNN) blended
      with Lanczos4 (0.85 * SR + 0.15 * Lanczos4), with pure Lanczos4 fallback.
    - Gentle edge-preserving bilateral luma & chroma denoising (sigmaColor=10, sigmaSpace=2.5).
    - Highlight-protected shadow lift and smooth CIE-LAB L-channel CLAHE (clipLimit=1.5, 6x6).
    - Clamped, halo-free luminance micro-sharpening (unsharp_max_delta=7.0).
    - Immutability: original capture is never mutated or overwritten.
    - Asynchronous worker execution with bounded priority queue and overload protection.
    """

    def __init__(
        self,
        config: Optional[EnhancementConfig] = None,
        model_name: str = "espcn",
        default_scale: int = 2,
        models_dir: Optional[Union[str, Path]] = None,
        max_queue_size: int = 20,
        num_workers: int = 1,
        auto_start: bool = True,
    ):
        self.config = config or EnhancementConfig()
        if model_name:
            self.config.model_name = model_name
        if default_scale:
            self.config.scale = default_scale
        if models_dir:
            self.config.models_dir = Path(models_dir)
        elif self.config.models_dir is None:
            # Default to outputs/models relative to this file
            self.config.models_dir = Path(__file__).resolve().parent.parent / "models"

        self.max_queue_size = max_queue_size
        self.num_workers = max(1, int(num_workers))
        self._queue = SafePriorityQueue(maxsize=self.max_queue_size)
        self._task_queue = self._queue

        # Super-resolution model cache and inference lock
        self._models: Dict[Tuple[str, int], Any] = {}
        self._model_lock = threading.Lock()

        # Worker threads & lifecycle
        self._workers: List[threading.Thread] = []
        self._running = False
        self._stop_event = threading.Event()

        # Operational metrics
        self._metrics_lock = threading.Lock()
        self._tasks_processed = 0
        self._tasks_dropped = 0
        self._tasks_rejected = 0
        self._total_processing_ms = 0.0
        self._last_processing_ms = 0.0

        if auto_start:
            self.start()

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start background worker thread(s)."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._workers = []
        for i in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"FaceEnhancer-Worker-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)
        logger.info("FaceEnhancer started with %d worker(s).", self.num_workers)

    def stop(self, wait: bool = True, timeout: float = 5.0) -> None:
        """Stop background worker thread(s) gracefully."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()

        # Wake up any waiting threads
        with self._queue._not_empty:
            self._queue._not_empty.notify_all()

        if wait:
            for t in self._workers:
                t.join(timeout=timeout)
        self._workers.clear()

        # Resolve any remaining queued tasks so caller futures/callbacks don't hang
        remaining = self._queue.clear()
        for task in remaining:
            if not task.future.done():
                res = EnhancementResult(
                    enhanced_image=task.image.copy() if task.image is not None else None,
                    scale=task.scale,
                    method="Cancelled",
                    original_path=task.original_path,
                    success=False,
                    is_fallback=True,
                    error="FaceEnhancer stopped before task could be executed",
                )
                task.future.set_result(res)
                if task.callback:
                    try:
                        task.callback(res)
                    except Exception as e:
                        logger.error("Callback error on shutdown: %s", e)
        logger.info("FaceEnhancer stopped.")

    def __enter__(self) -> FaceEnhancer:
        if not self._running:
            self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop(wait=True)

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational metrics for monitoring and contracts."""
        with self._metrics_lock:
            avg_ms = (
                self._total_processing_ms / self._tasks_processed
                if self._tasks_processed > 0 else 0.0
            )
            return {
                "pending_enhancements": self._queue.qsize(),
                "tasks_processed": self._tasks_processed,
                "tasks_dropped": self._tasks_dropped,
                "tasks_rejected": self._tasks_rejected,
                "enhancement_ms": round(float(self._last_processing_ms), 2),
                "avg_enhancement_ms": round(float(avg_ms), 2),
                "is_running": self._running,
            }

    # -------------------------------------------------------------------------
    # Core Synchronous Enhancement Pipeline
    # -------------------------------------------------------------------------

    @staticmethod
    def format_portrait_aspect_ratio(
        image: np.ndarray,
        target_aspect: str = "1:1",
        min_dim: int = 384,
    ) -> np.ndarray:
        """Format an image to exact portrait aspect ratio without distortion or black bars.

        Args:
            image: BGR image (H, W, 3).
            target_aspect: "1:1" (square) or "4:5" (portrait). If "original", aspect ratio is retained.
            min_dim: Minimum output dimension (e.g. 384).

        Returns:
            Properly framed BGR image without distortion or black bars.
        """
        if image is None or image.size == 0:
            return image

        h, w = image.shape[:2]

        if target_aspect == "1:1":
            target_ratio = 1.0
        elif target_aspect == "4:5":
            target_ratio = 4.0 / 5.0
        else:
            if min_dim > 0 and (w < min_dim or h < min_dim):
                scale = max(float(min_dim) / w, float(min_dim) / h)
                nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
                return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
            return image

        current_ratio = float(w) / float(h)
        if current_ratio > target_ratio:
            # Wider than target -> center crop horizontally
            new_w = max(1, int(round(h * target_ratio)))
            start_x = max(0, (w - new_w) // 2)
            cropped = image[:, start_x:start_x + new_w]
        elif current_ratio < target_ratio:
            # Taller than target -> crop vertically with top headroom bias (35% top, 65% bottom)
            new_h = max(1, int(round(w / target_ratio)))
            diff_h = h - new_h
            if diff_h > 0:
                start_y = max(0, int(round(diff_h * 0.35)))
                if start_y + new_h > h:
                    start_y = h - new_h
                cropped = image[start_y:start_y + new_h, :]
            else:
                cropped = image
        else:
            cropped = image

        crop_h, crop_w = cropped.shape[:2]
        if min_dim > 0 and (crop_w < min_dim or crop_h < min_dim):
            if target_aspect == "1:1":
                target_w = max(min_dim, crop_w)
                target_h = target_w
            elif target_aspect == "4:5":
                target_w = max(min_dim, crop_w)
                target_h = int(round(target_w * 1.25))
            else:
                scale = max(float(min_dim) / crop_w, float(min_dim) / crop_h)
                target_w = max(1, int(round(crop_w * scale)))
                target_h = max(1, int(round(crop_h * scale)))
            cropped = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        return cropped

    def enhance(
        self,
        image: Optional[np.ndarray],
        scale: Optional[int] = None,
        model_name: Optional[str] = None,
        original_path: Optional[Union[str, Path]] = None,
        save_to_disk: Optional[bool] = None,
        capture_record: Optional[Any] = None,
        min_dimension: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
    ) -> EnhancementResult:
        """Synchronously execute the full studio-grade facial enhancement pipeline.

        Guarantees:
        - NEVER mutates or overwrites the input image.
        - Produces output of exact requested multiple (2x or 4x).
        - Protects memory against OOM on oversized inputs.
        - Recovers gracefully with Lanczos4 fallback or original input.
        """
        start_time = time.perf_counter()
        target_scale = scale or self.config.scale
        algo = (model_name or self.config.model_name).lower()
        should_save = self.config.save_to_disk if save_to_disk is None else save_to_disk

        orig_path_str = str(original_path) if original_path is not None else None
        capture_id = getattr(capture_record, "capture_id", None)
        track_id = getattr(capture_record, "track_id", None)
        session_id = getattr(capture_record, "session_id", None)

        if orig_path_str is None and capture_record is not None:
            orig_path_str = getattr(capture_record, "original_path", None)

        # Handle crop extraction if candidate or record passed as image
        if hasattr(image, "crop_original"):
            image = getattr(image, "crop_original")
        elif image is None and capture_record is not None and hasattr(capture_record, "crop_original"):
            image = getattr(capture_record, "crop_original")

        # Handle image loading if None but path is provided
        if image is None:
            if orig_path_str and os.path.exists(orig_path_str):
                try:
                    image = cv2.imread(orig_path_str)
                except Exception as e:
                    return EnhancementResult(
                        success=False,
                        error=f"Failed to read image from path '{orig_path_str}': {e}",
                        scale=target_scale,
                        original_path=orig_path_str,
                        capture_id=capture_id,
                        track_id=track_id,
                        session_id=session_id,
                    )
            else:
                return EnhancementResult(
                    success=False,
                    error="Input image is None and valid file path was not provided",
                    scale=target_scale,
                    original_path=orig_path_str,
                    capture_id=capture_id,
                    track_id=track_id,
                    session_id=session_id,
                )

        # Validate input structure
        if not isinstance(image, np.ndarray) or image.size == 0 or len(image.shape) < 2:
            return EnhancementResult(
                success=False,
                error="Invalid image array provided (empty or wrong dimensions)",
                scale=target_scale,
                original_path=orig_path_str,
                capture_id=capture_id,
                track_id=track_id,
                session_id=session_id,
            )

        orig_h, orig_w = image.shape[:2]
        orig_shape = (orig_h, orig_w, 3 if len(image.shape) == 2 else image.shape[2])

        # IMMUTABILITY GUARANTEE: strictly work on an isolated deep copy
        clean_input = np.copy(image)

        # Format conversions to 3-channel BGR uint8
        if len(clean_input.shape) == 2:
            clean_input = cv2.cvtColor(clean_input, cv2.COLOR_GRAY2BGR)
        elif clean_input.shape[2] == 4:
            clean_input = cv2.cvtColor(clean_input, cv2.COLOR_BGRA2BGR)
        elif clean_input.shape[2] != 3:
            return EnhancementResult(
                success=False,
                error=f"Unsupported channel count: {clean_input.shape[2]}",
                original_shape=orig_shape,
                scale=target_scale,
                original_path=orig_path_str,
            )

        if clean_input.dtype != np.uint8:
            clean_input = np.clip(clean_input, 0, 255).astype(np.uint8)

        # Check minimum input bounds
        h, w = clean_input.shape[:2]
        if h < self.config.min_input_dim or w < self.config.min_input_dim:
            return EnhancementResult(
                success=False,
                error=f"Crop dimensions ({w}x{h}) smaller than min limit {self.config.min_input_dim}",
                original_shape=orig_shape,
                scale=target_scale,
                original_path=orig_path_str,
            )

        # Determine aspect ratio handling
        if aspect_ratio is not None:
            target_aspect = aspect_ratio
        elif scale is None:
            target_aspect = getattr(self.config, "portrait_aspect_ratio", "1:1")
        else:
            target_aspect = "original"

        # Apply aspect ratio framing if 1:1 or 4:5 requested without distortion or black bars
        if target_aspect in ("1:1", "4:5"):
            curr_h, curr_w = clean_input.shape[:2]
            target_ratio = 1.0 if target_aspect == "1:1" else 0.8
            curr_ratio = float(curr_w) / float(curr_h)
            if curr_ratio > target_ratio:
                new_w = max(1, int(round(curr_h * target_ratio)))
                start_x = max(0, (curr_w - new_w) // 2)
                clean_input = clean_input[:, start_x:start_x + new_w]
            elif curr_ratio < target_ratio:
                new_h = max(1, int(round(curr_w / target_ratio)))
                diff_h = curr_h - new_h
                if diff_h > 0:
                    start_y = max(0, int(round(diff_h * 0.35)))
                    if start_y + new_h > curr_h:
                        start_y = curr_h - new_h
                    clean_input = clean_input[start_y:start_y + new_h, :]

        h, w = clean_input.shape[:2]

        # Bound neural work without discarding source detail and then enlarging
        # the discarded pixels again. Large crops use conventional processing.
        pre_downscaled = False
        if max(h, w) > self.config.max_input_dim:
            algo = "lanczos"

        min_required_dim = min_dimension if min_dimension is not None else self.config.min_output_dim
        is_auto_scaled = False

        if scale is not None:
            target_scale = int(scale)
            expected_target_w = w * target_scale
            expected_target_h = h * target_scale
            if min_dimension is not None and min_required_dim > 0:
                min_side = min(expected_target_h, expected_target_w)
                if min_side < min_required_dim:
                    scale_factor = float(min_required_dim) / min_side
                    expected_target_w = max(1, int(round(expected_target_w * scale_factor)))
                    expected_target_h = max(1, int(round(expected_target_h * scale_factor)))
        else:
            if min_required_dim > 0 and (w < min_required_dim or h < min_required_dim):
                is_auto_scaled = True
                # Select optimal neural model scale factor:
                # 4x super-resolution for small crops <= 128px, 2x for > 128px
                if max(w, h) <= 128:
                    target_scale = 4
                else:
                    target_scale = 2

                if target_aspect == "1:1":
                    expected_target_w = max(min_required_dim, max(w * target_scale, h * target_scale))
                    expected_target_h = expected_target_w
                elif target_aspect == "4:5":
                    expected_target_w = max(min_required_dim, w * target_scale)
                    expected_target_h = int(round(expected_target_w * 1.25))
                else:
                    raw_w = w * target_scale
                    raw_h = h * target_scale
                    min_side = min(raw_w, raw_h)
                    if min_side < min_required_dim:
                        scale_factor = float(min_required_dim) / min_side
                        expected_target_w = max(min_required_dim, int(round(raw_w * scale_factor)))
                        expected_target_h = max(min_required_dim, int(round(raw_h * scale_factor)))
                    else:
                        expected_target_w = raw_w
                        expected_target_h = raw_h
            else:
                target_scale = self.config.scale
                if target_aspect == "1:1":
                    expected_target_w = max(w, h)
                    expected_target_h = expected_target_w
                elif target_aspect == "4:5":
                    expected_target_w = w
                    expected_target_h = int(round(w * 1.25))
                else:
                    expected_target_w = w * target_scale
                    expected_target_h = h * target_scale

        if expected_target_w * expected_target_h > 16_000_000:
            return EnhancementResult(
                enhanced_image=clean_input,
                original_shape=orig_shape,
                enhanced_shape=clean_input.shape,
                scale=1,
                method="Original (preview size limit)",
                success=False,
                is_fallback=True,
                error="Requested preview exceeds 16 megapixels",
            )

        try:
            # -----------------------------------------------------------------
            # Step 1 & 2: Gentle Bilateral Luma/Chroma Denoising, Shadow Lift & L-CLAHE
            # -----------------------------------------------------------------
            pre_sr = self._apply_filtering_and_clahe(clean_input)

            # -----------------------------------------------------------------
            # Step 3: Hybrid Super-Resolution Upscaling (0.85 * SR + 0.15 * Lanczos4)
            # -----------------------------------------------------------------
            upscaled, method_name, is_fallback = self._upscale(
                pre_sr, target_scale, algo
            )

            # Ensure exact target dimensions match requested multiple of original crop
            if (upscaled.shape[1], upscaled.shape[0]) != (expected_target_w, expected_target_h):
                upscaled = cv2.resize(
                    upscaled,
                    (expected_target_w, expected_target_h),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            # -----------------------------------------------------------------
            # Step 4: Clamped Halo-Free Luminance Micro-Sharpening
            # -----------------------------------------------------------------
            enhanced = self._apply_unsharp_mask(upscaled)

        except Exception as e:
            logger.exception("Enhancement pipeline error: %s; falling back to original", e)
            fallback_img = cv2.resize(
                image.copy(),
                (expected_target_w, expected_target_h),
                interpolation=cv2.INTER_LANCZOS4,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return EnhancementResult(
                enhanced_image=fallback_img,
                original_shape=orig_shape,
                enhanced_shape=fallback_img.shape,
                scale=target_scale,
                method=f"Fallback-Lanczos4-x{target_scale} (Error Recovery)",
                processing_time_ms=elapsed_ms,
                original_path=orig_path_str,
                success=False,
                is_fallback=True,
                error=str(e),
                capture_id=capture_id,
                track_id=track_id,
                session_id=session_id,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Build full method identifier string
        if is_auto_scaled and not is_fallback:
            if (expected_target_w, expected_target_h) != (w * target_scale, h * target_scale):
                full_method = f"{method_name} + Lanczos4-SR + CLAHE"
            else:
                full_method = f"{method_name} + CLAHE"
        else:
            full_method = f"{method_name} + CLAHE"

        saved_path_str = None
        # Handle disk separation & saving
        if should_save and orig_path_str:
            saved_path = get_enhanced_path(orig_path_str, scale=target_scale)
            try:
                saved_path.parent.mkdir(parents=True, exist_ok=True)
                # Verify distinct file path before writing
                if str(saved_path) == orig_path_str:
                    saved_path = saved_path.with_name(f"{saved_path.stem}_enhanced{saved_path.suffix}")
                cv2.imwrite(
                    str(saved_path),
                    enhanced,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality],
                )
                saved_path_str = str(saved_path)
            except Exception as e:
                logger.error("Failed to write enhanced image to %s: %s", saved_path, e)

        # Update CaptureRecord contract fields if provided
        if capture_record is not None:
            try:
                if saved_path_str:
                    capture_record.enhanced_path = saved_path_str
                    if os.path.exists(saved_path_str):
                        capture_record.file_size_bytes = os.path.getsize(saved_path_str)
                capture_record.is_enhanced = True
                capture_record.enhancement_method = full_method
            except Exception as e:
                logger.warning("Could not update CaptureRecord: %s", e)

        metadata = {
            "pre_downscaled": pre_downscaled,
            "intermediate_shape": clean_input.shape,
            "target_scale": target_scale,
            "model_requested": algo,
        }

        result = EnhancementResult(
            enhanced_image=enhanced,
            original_shape=orig_shape,
            enhanced_shape=enhanced.shape,
            scale=target_scale,
            method=full_method,
            processing_time_ms=elapsed_ms,
            original_path=orig_path_str,
            enhanced_path=saved_path_str,
            success=True,
            is_fallback=is_fallback,
            capture_id=capture_id,
            track_id=track_id,
            session_id=session_id,
            metadata=metadata,
        )

        with self._metrics_lock:
            self._tasks_processed += 1
            self._total_processing_ms += elapsed_ms
            self._last_processing_ms = elapsed_ms

        return result

    # -------------------------------------------------------------------------
    # Pipeline Processing Steps
    # -------------------------------------------------------------------------

    def _lift_shadows_preserve_highlights(self, luma: np.ndarray) -> np.ndarray:
        """Apply smooth highlight-protected shadow lift to dark or backlit portraits."""
        mean_l = float(np.mean(luma))
        if mean_l <= 3.0:
            return luma

        p15 = float(np.percentile(luma, 15))
        if mean_l >= 118.0 and p15 >= 58.0:
            return luma

        eff_dark = min(mean_l, p15 + 34.0)
        lift_strength = float(np.clip((118.0 - eff_dark) / 95.0, 0.0, 0.85))
        if lift_strength <= 0.02:
            return luma

        gamma_exp = float(np.clip(mean_l / 138.0, 0.68, 0.92))
        x = np.linspace(0.0, 1.0, 256, dtype=np.float64)
        gamma_curve = np.power(x, gamma_exp)
        # Protect bright highlights from blowing out while opening dark shadows
        highlight_rolloff = 1.0 - 0.65 * np.power(x, 2.5)
        y_lut = x + lift_strength * (gamma_curve - x) * highlight_rolloff
        lut = np.clip(np.round(y_lut * 255.0), 0, 255).astype(np.uint8)
        return cv2.LUT(luma, lut)

    def _enhance_luma_channel(self, luma: np.ndarray) -> np.ndarray:
        """Denoise luminance, lift shadows if needed, and blend subtle CLAHE smoothly."""
        h, w = luma.shape[:2]
        clean_l = cv2.bilateralFilter(
            luma,
            d=self.config.bilateral_d,
            sigmaColor=float(self.config.bilateral_sigma_color),
            sigmaSpace=float(self.config.bilateral_sigma_space),
        )

        if getattr(self.config, "adaptive_exposure", True):
            clean_l = self._lift_shadows_preserve_highlights(clean_l)

        # Evaluate post-lift histogram balance; bypass CLAHE when natural tonal
        # range is already studio-balanced so skin tones never look over-processed.
        mean_l = float(np.mean(clean_l))
        std_l = float(np.std(clean_l))
        p10, p90 = np.percentile(clean_l, [10, 90])
        span_l = float(p90 - p10)
        dark_frac = float(np.mean(clean_l < 25))
        bright_frac = float(np.mean(clean_l > 240))

        already_balanced = (
            105.0 <= mean_l <= 165.0
            and 38.0 <= std_l <= 68.0
            and span_l >= 95.0
            and dark_frac < 0.08
            and bright_frac < 0.05
        )
        if already_balanced:
            return clean_l

        clip = max(1.0, min(3.0, float(self.config.clahe_clip_limit)))
        grid_x = max(2, min(int(self.config.clahe_tile_grid_size[0]), max(2, w // 4)))
        grid_y = max(2, min(int(self.config.clahe_tile_grid_size[1]), max(2, h // 4)))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid_x, grid_y))
        clahe_l = clahe.apply(clean_l)

        w_clahe = float(np.clip(getattr(self.config, "clahe_blend_weight", 0.55), 0.0, 1.0))
        return cv2.addWeighted(clean_l, 1.0 - w_clahe, clahe_l, w_clahe, 0.0)

    def _apply_filtering_and_clahe(self, image: np.ndarray) -> np.ndarray:
        """Gentle edge-preserving bilateral luma/chroma denoising, shadow lift & subtle CLAHE.

        Uses compact bilateral filtering (sigmaColor=10, sigmaSpace=2.5) on both
        luminance and chrominance to remove compression/sensor blockiness without
        plastic skin smoothing or color bleed across lips/eyes, followed by smooth
        0.55-weighted CIE-LAB L-channel CLAHE (clipLimit=1.5, tileGridSize=(6, 6)).
        """
        color_space = getattr(self.config, "color_space", "lab").lower()
        chroma_sigma_c = max(6.0, float(self.config.bilateral_sigma_color))
        chroma_sigma_s = float(self.config.bilateral_sigma_space)

        if color_space == "ycrcb":
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            contrast_y = self._enhance_luma_channel(y)
            cr_clean = cv2.bilateralFilter(
                cr, d=self.config.bilateral_d, sigmaColor=chroma_sigma_c, sigmaSpace=chroma_sigma_s
            )
            cb_clean = cv2.bilateralFilter(
                cb, d=self.config.bilateral_d, sigmaColor=chroma_sigma_c, sigmaSpace=chroma_sigma_s
            )
            return cv2.cvtColor(cv2.merge((contrast_y, cr_clean, cb_clean)), cv2.COLOR_YCrCb2BGR)

        # Default: CIE-LAB perceptual lightness + chroma channels
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        contrast_l = self._enhance_luma_channel(l_chan)
        a_clean = cv2.bilateralFilter(
            a_chan, d=self.config.bilateral_d, sigmaColor=chroma_sigma_c, sigmaSpace=chroma_sigma_s
        )
        b_clean = cv2.bilateralFilter(
            b_chan, d=self.config.bilateral_d, sigmaColor=chroma_sigma_c, sigmaSpace=chroma_sigma_s
        )
        return cv2.cvtColor(cv2.merge((contrast_l, a_clean, b_clean)), cv2.COLOR_LAB2BGR)

    def _upscale(
        self,
        image: np.ndarray,
        scale: int,
        algo: str
    ) -> Tuple[np.ndarray, str, bool]:
        """Perform hybrid neural super-resolution upscale (0.85 * SR + 0.15 * Lanczos4) or fallback."""
        h, w = image.shape[:2]
        target_size = (w * scale, h * scale)

        # If lanczos is explicitly requested or scale is not 2 or 4
        if algo == "lanczos" or scale not in (2, 4):
            upscaled = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
            return upscaled, f"Lanczos4-x{scale}", True

        # Attempt neural DnnSuperRes
        sr_model = self._get_or_load_model(algo, scale)
        if sr_model is not None:
            try:
                with self._model_lock:
                    neural = sr_model.upsample(image)

                # Blend DNN SR with Lanczos4 (0.85 * sr + 0.15 * lanczos) to retain
                # natural micro-texture and eliminate synthetic neural ringing
                blend_w = float(np.clip(self.config.blend_conventional, 0.0, 1.0))
                if blend_w > 0.0:
                    lanczos = cv2.resize(
                        image,
                        (neural.shape[1], neural.shape[0]),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                    neural = cv2.addWeighted(
                        neural,
                        1.0 - blend_w,
                        lanczos,
                        blend_w,
                        0.0,
                    )
                return neural, f"{algo.upper()}-x{scale}", False
            except Exception as e:
                logger.warning(
                    "DnnSuperRes upsample failed for %s x%d: %s; falling back to Lanczos4",
                    algo, scale, e
                )

        # Fallback to high-grade Lanczos4
        upscaled = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
        return upscaled, f"Lanczos4-x{scale}", True

    def _get_or_load_model(self, algo: str, scale: int) -> Optional[Any]:
        """Retrieve cached OpenCV DnnSuperRes model or load from disk."""
        key = (algo, scale)
        if key in self._models:
            return self._models[key]

        if not hasattr(cv2, "dnn_superres"):
            logger.warning("cv2.dnn_superres is not available in OpenCV build")
            return None

        models_dir = Path(self.config.models_dir) if self.config.models_dir else None
        if models_dir is None or not models_dir.exists():
            logger.warning("Super-resolution models directory not found: %s", models_dir)
            return None

        # Look for model file e.g. ESPCN_x2.pb, FSRCNN_x4.pb
        candidates = [
            models_dir / f"{algo.upper()}_x{scale}.pb",
            models_dir / f"{algo.lower()}_x{scale}.pb",
            models_dir / f"{algo}_x{scale}.pb",
        ]
        model_path = next((p for p in candidates if p.exists()), None)
        if model_path is None:
            logger.warning("Model file not found for %s x%d in %s", algo, scale, models_dir)
            return None

        try:
            with self._model_lock:
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                sr.readModel(str(model_path))
                sr.setModel(algo.lower(), scale)
                sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self._models[key] = sr
                logger.info("Loaded super-resolution model: %s (%s x%d)", model_path.name, algo, scale)
                return sr
        except Exception as e:
            logger.error("Failed to load DnnSuperRes model from %s: %s", model_path, e)
            return None

    def _apply_unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        """Apply clamped, halo-free luminance micro-sharpening in CIE-LAB space.

        Sharpens strictly the perceptual lightness (L) channel with soft-coring
        deadband and +-unsharp_max_delta (7.0) clamping so eyelashes, eyebrows,
        and iris detail become razor-sharp without color fringing or edge halos.
        """
        if self.config.unsharp_amount <= 0.0:
            return image

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        l_f = l_chan.astype(np.float32)
        blurred_l = cv2.GaussianBlur(l_f, (0, 0), sigmaX=float(self.config.unsharp_radius))
        detail = l_f - blurred_l

        # Soft-threshold coring: ignore flat-skin sensor grain smoothly without step contours
        thresh = float(self.config.unsharp_threshold)
        if thresh > 0.0:
            detail = np.sign(detail) * np.maximum(0.0, np.abs(detail) - thresh)

        # Clamping: strictly bound edge delta (default 7.0) to eliminate bright/dark halos
        max_delta = float(self.config.unsharp_max_delta)
        detail = np.clip(detail, -max_delta, max_delta)

        sharp_l = np.clip(
            np.round(l_f + float(self.config.unsharp_amount) * detail),
            0.0,
            255.0,
        ).astype(np.uint8)

        return cv2.cvtColor(cv2.merge((sharp_l, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

    # -------------------------------------------------------------------------
    # Asynchronous Execution & Queue Management
    # -------------------------------------------------------------------------

    def submit(
        self,
        image: Optional[np.ndarray] = None,
        priority: int = PRIORITY_NORMAL,
        scale: Optional[int] = None,
        model_name: Optional[str] = None,
        capture_record: Optional[Any] = None,
        original_path: Optional[Union[str, Path]] = None,
        save_to_disk: Optional[bool] = None,
        callback: Optional[Callable[[EnhancementResult], None]] = None,
        min_dimension: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
    ) -> Future:
        """Asynchronously submit an enhancement task to the background queue.

        Args:
            image: Original face crop (numpy array).
            priority: Priority integer (lower = higher priority, e.g. 0=high, 10=normal, 20=low).
            scale: Target upscale multiple (2 or 4), or None for automatic super-resolution.
            model_name: "espcn", "fsrcnn", or "lanczos".
            capture_record: Optional CaptureRecord contract instance.
            original_path: Optional original asset path.
            save_to_disk: Whether to persist enhanced asset to disk.
            callback: Completion callback receiving EnhancementResult.
            min_dimension: Minimum output dimension (e.g. 256 or 384).
            aspect_ratio: Target portrait aspect ratio ("1:1", "4:5", or "original").

        Returns:
            concurrent.futures.Future resolving to EnhancementResult.
        """
        fut: Future = Future()
        target_scale = scale
        should_save = self.config.save_to_disk if save_to_disk is None else save_to_disk
        orig_path_str = str(original_path) if original_path is not None else None

        if orig_path_str is None and capture_record is not None:
            orig_path_str = getattr(capture_record, "original_path", None)

        # Handle crop extraction if candidate or record passed as image
        if hasattr(image, "crop_original"):
            image = getattr(image, "crop_original")
        elif image is None and capture_record is not None and hasattr(capture_record, "crop_original"):
            image = getattr(capture_record, "crop_original")

        task = EnhancementTask(
            priority=priority,
            image=image,
            scale=target_scale,
            model_name=model_name,
            capture_record=capture_record,
            original_path=orig_path_str,
            save_to_disk=should_save,
            callback=callback,
            future=fut,
            min_dimension=min_dimension,
            aspect_ratio=aspect_ratio,
        )

        status, displaced = self._queue.put(task)

        if status == "rejected":
            # Queue was full of higher-or-equal priority tasks; reject immediately
            with self._metrics_lock:
                self._tasks_rejected += 1
            err_msg = f"Task rejected under queue overload (capacity: {self.max_queue_size})"
            res = EnhancementResult(
                enhanced_image=image.copy() if image is not None else None,
                scale=target_scale,
                method="None",
                original_path=orig_path_str,
                success=False,
                is_fallback=True,
                error=err_msg,
            )
            fut.set_result(res)
            if callback:
                try:
                    callback(res)
                except Exception as e:
                    logger.error("Callback error on rejection: %s", e)
            return fut

        elif status == "dropped" and displaced is not None:
            # Low-priority task was evicted to make room for higher-priority task
            with self._metrics_lock:
                self._tasks_dropped += 1
            err_msg = "Task dropped from queue under overload to prioritize higher-priority task"
            res = EnhancementResult(
                enhanced_image=displaced.image.copy() if displaced.image is not None else None,
                scale=displaced.scale,
                method="None",
                original_path=displaced.original_path,
                success=False,
                is_fallback=True,
                error=err_msg,
            )
            displaced.future.set_result(res)
            if displaced.callback:
                try:
                    displaced.callback(res)
                except Exception as e:
                    logger.error("Callback error on dropped task: %s", e)

        return fut

    def _worker_loop(self) -> None:
        """Background worker thread processing enhancement tasks."""
        while not self._stop_event.is_set():
            task = self._queue.get(timeout=0.1)
            if task is None:
                continue

            try:
                result = self.enhance(
                    image=task.image,
                    scale=task.scale,
                    model_name=task.model_name,
                    original_path=task.original_path,
                    save_to_disk=task.save_to_disk,
                    capture_record=task.capture_record,
                    min_dimension=getattr(task, "min_dimension", None),
                    aspect_ratio=getattr(task, "aspect_ratio", None),
                )
            except Exception as e:
                logger.exception("Unexpected exception in worker thread: %s", e)
                result = EnhancementResult(
                    enhanced_image=task.image.copy() if task.image is not None else None,
                    scale=task.scale,
                    method="Worker-Error",
                    original_path=task.original_path,
                    success=False,
                    is_fallback=True,
                    error=str(e),
                )

            # Deliver result to future
            if not task.future.done():
                task.future.set_result(result)

            # Trigger completion callback
            if task.callback:
                try:
                    task.callback(result)
                except Exception as cb_err:
                    logger.error("Error in enhancement completion callback: %s", cb_err)


def format_portrait_aspect_ratio(
    image: np.ndarray,
    target_aspect: str = "1:1",
    min_dim: int = 384,
) -> np.ndarray:
    """Format an image to exact portrait aspect ratio without distortion or black bars."""
    return FaceEnhancer.format_portrait_aspect_ratio(
        image=image, target_aspect=target_aspect, min_dim=min_dim
    )


# Module-level default singleton for easy application-wide integration
_default_enhancer: Optional[FaceEnhancer] = None
_default_lock = threading.Lock()


def get_default_enhancer() -> FaceEnhancer:
    """Retrieve or create the global application FaceEnhancer instance."""
    global _default_enhancer
    with _default_lock:
        if _default_enhancer is None:
            _default_enhancer = FaceEnhancer()
        return _default_enhancer


__all__ = [
    "FaceEnhancer",
    "EnhancementConfig",
    "EnhancementResult",
    "EnhancementTask",
    "SafePriorityQueue",
    "get_default_enhancer",
    "get_enhanced_path",
    "get_enhanced_filename",
    "format_portrait_aspect_ratio",
    "PRIORITY_HIGH",
    "PRIORITY_NORMAL",
    "PRIORITY_LOW",
]
