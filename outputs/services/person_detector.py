"""Reliable multi-person detector wrapping YOLO inference for crowded and varied scenes.

Part of the real-time multi-person facial capture pipeline.
Extracts localized PersonDetection instances conforming to services.face_contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import psutil
import torch

try:
    from ultralytics import YOLO
    from ultralytics.engine.results import Boxes, Results
except ImportError:  # pragma: no cover
    YOLO = None
    Boxes = None
    Results = None

try:
    from .face_contracts import PersonDetection
except ImportError:
    from services.face_contracts import PersonDetection

logger = logging.getLogger(__name__)

# COCO Person class ID
PERSON_CLASS_ID = 0

DEFAULT_MODEL_NAME = "yolo26n"
DEFAULT_CONFIDENCE_THRESHOLD = 0.10
DEFAULT_IOU_THRESHOLD = 0.65
DEFAULT_RESOLUTION = 640
DEFAULT_TARGET_MS = 120.0
DEFAULT_MIN_RESOLUTION = 480
DEFAULT_MAX_RESOLUTION = 1280
DEFAULT_MAX_DETECTIONS = 1000


@dataclass
class PersonDetectorConfig:
    """Runtime configuration for PersonDetector."""
    model_name: str = DEFAULT_MODEL_NAME
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    resolution: int = DEFAULT_RESOLUTION
    adaptive: bool = True
    target_ms: float = DEFAULT_TARGET_MS
    min_resolution: int = DEFAULT_MIN_RESOLUTION
    max_resolution: int = DEFAULT_MAX_RESOLUTION
    max_det: int = DEFAULT_MAX_DETECTIONS
    device: Optional[str] = None
    backend: str = "pytorch"
    cpu_threads: Optional[int] = None
    fallback_to_mock: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "resolution": self.resolution,
            "adaptive": self.adaptive,
            "target_ms": self.target_ms,
            "min_resolution": self.min_resolution,
            "max_resolution": self.max_resolution,
            "max_det": self.max_det,
            "device": self.device,
            "backend": self.backend,
            "cpu_threads": self.cpu_threads,
            "fallback_to_mock": self.fallback_to_mock,
        }


def validate_person_detector_config(config: Union[PersonDetectorConfig, Dict[str, Any]]) -> PersonDetectorConfig:
    """Validate and normalize detector settings."""
    if isinstance(config, PersonDetectorConfig):
        data = config.to_dict()
    elif isinstance(config, dict):
        data = {**PersonDetectorConfig().to_dict(), **config}
    else:
        raise TypeError("Configuration must be a PersonDetectorConfig or dict")

    # Model name validation
    model_name = str(data.get("model_name", DEFAULT_MODEL_NAME)).strip()
    if not model_name:
        raise ValueError("model_name cannot be empty")
    data["model_name"] = model_name

    # Confidence threshold: 0.01 to 0.99
    conf = float(data.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
    if not np.isfinite(conf) or not (0.01 <= conf <= 0.99):
        raise ValueError("confidence_threshold must be between 0.01 and 0.99")
    data["confidence_threshold"] = conf

    # IoU threshold: 0.1 to 0.99
    iou = float(data.get("iou_threshold", DEFAULT_IOU_THRESHOLD))
    if not np.isfinite(iou) or not (0.10 <= iou <= 0.99):
        raise ValueError("iou_threshold must be between 0.10 and 0.99")
    data["iou_threshold"] = iou

    # Backend
    backend = str(data.get("backend", "pytorch")).lower().strip()
    if backend not in ("pytorch", "onnx", "tensorrt"):
        raise ValueError(f"Unsupported backend '{backend}'. Choose from 'pytorch', 'onnx', 'tensorrt'")
    data["backend"] = backend

    # Resolutions: snap to multiple of 32
    res = int(data.get("resolution", DEFAULT_RESOLUTION))
    min_res = int(data.get("min_resolution", DEFAULT_MIN_RESOLUTION))
    max_res = int(data.get("max_resolution", DEFAULT_MAX_RESOLUTION))

    if not (256 <= res <= 2560):
        raise ValueError("resolution must be between 256 and 2560")
    if not (256 <= min_res <= max_res):
        raise ValueError("min_resolution must be between 256 and max_resolution")
    if not (min_res <= max_res <= 2560):
        raise ValueError("max_resolution must be between min_resolution and 2560")

    data["resolution"] = int(round(res / 32) * 32)
    data["min_resolution"] = int(round(min_res / 32) * 32)
    data["max_resolution"] = int(round(max_res / 32) * 32)

    # Target ms
    target_ms = float(data.get("target_ms", DEFAULT_TARGET_MS))
    if not np.isfinite(target_ms) or target_ms <= 0:
        raise ValueError("target_ms must be positive")
    data["target_ms"] = target_ms

    # Max detections
    max_det = int(data.get("max_det", DEFAULT_MAX_DETECTIONS))
    if max_det < 1:
        raise ValueError("max_det must be at least 1")
    data["max_det"] = max_det

    data["adaptive"] = bool(data.get("adaptive", True))
    data["fallback_to_mock"] = bool(data.get("fallback_to_mock", True))

    return PersonDetectorConfig(**data)


def select_device(preferred_device: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Determine the optimal compute device with automatic fallback.

    Returns:
        (device_str, warning_message_or_None)
    """
    if preferred_device is not None:
        device_str = str(preferred_device).lower().strip()
        if device_str.startswith("cuda"):
            if torch.cuda.is_available():
                return preferred_device, None
            return "cpu", f"CUDA device '{preferred_device}' requested but CUDA is unavailable; falling back to CPU"
        elif device_str == "cpu":
            return "cpu", None
        return preferred_device, None

    if torch.cuda.is_available():
        return "cuda:0", None
    return "cpu", None


def configure_cpu_threads(cpu_threads: Optional[int] = None) -> int:
    """Reserve balanced CPU capacity avoiding thread starvation across pipeline services."""
    physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    env_threads = os.environ.get("TORCH_THREADS") or os.environ.get("TRAFFIC_TORCH_THREADS")
    requested = cpu_threads if cpu_threads is not None else env_threads
    threads = max(
        1,
        min(
            int(requested) if requested is not None else min(2, max(1, physical - 2)),
            os.cpu_count() or 1
        )
    )
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    return threads


class MockBoxes:
    """Lightweight fallback mock for ultralytics Boxes supporting numpy arrays."""
    def __init__(self, data: np.ndarray, orig_shape: Tuple[int, int]):
        self.data = np.asarray(data, dtype=np.float32).reshape(-1, 6) if len(data) > 0 else np.empty((0, 6), dtype=np.float32)
        self.orig_shape = orig_shape

    @property
    def xyxy(self) -> np.ndarray:
        return self.data[:, :4] if len(self.data) > 0 else np.empty((0, 4), dtype=np.float32)

    @property
    def conf(self) -> np.ndarray:
        return self.data[:, 4] if len(self.data) > 0 else np.empty((0,), dtype=np.float32)

    @property
    def cls(self) -> np.ndarray:
        return self.data[:, 5] if len(self.data) > 0 else np.empty((0,), dtype=np.float32)

    def cpu(self) -> MockBoxes:
        return self

    def numpy(self) -> MockBoxes:
        return self

    def __len__(self) -> int:
        return len(self.data)


class MockResult:
    """Lightweight fallback mock for ultralytics Results object."""
    def __init__(self, boxes_data: np.ndarray, orig_shape: Tuple[int, int]):
        if Boxes is not None:
            try:
                self.boxes = Boxes(boxes_data, orig_shape)
            except Exception:
                self.boxes = MockBoxes(boxes_data, orig_shape)
        else:
            self.boxes = MockBoxes(boxes_data, orig_shape)
        self.orig_shape = orig_shape
        self.speed = {"preprocess": 0.5, "inference": 12.0, "postprocess": 0.8}


class MockYOLO:
    """Stand-in YOLO model for test environments or when model weights are not present."""
    names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(self, default_boxes: Optional[np.ndarray] = None):
        self.default_boxes = default_boxes

    def set_boxes(self, boxes: np.ndarray) -> None:
        self.default_boxes = boxes

    def predict(self, source: Any, **kwargs) -> List[MockResult]:
        images = source if isinstance(source, (list, tuple)) else [source]
        results = []
        for img in images:
            h, w = (img.shape[:2]) if hasattr(img, "shape") else (720, 1280)
            if self.default_boxes is not None:
                boxes_data = np.asarray(self.default_boxes, dtype=np.float32)
            else:
                boxes_data = np.empty((0, 6), dtype=np.float32)
            results.append(MockResult(boxes_data, (h, w)))
        return results

    def __call__(self, source: Any, **kwargs) -> List[MockResult]:
        return self.predict(source, **kwargs)


class PersonDetector:
    """High-reliability multi-person detector supporting crowded environments and adaptive resolution.

    Features:
    - Wraps YOLO (yolo26n, yolo26s) or seamless fallback mock.
    - Optimized for dense crowds (20-30+ people per scene).
    - Filters strictly for COCO class 0 (person).
    - Configurable confidence threshold (default 0.15) and NMS IoU (0.65).
    - Dynamic adaptive inference resolution (e.g. 640 or 960) responding to frame latency.
    - Automatic CPU/GPU device selection with fallback.
    - Batched detection support.
    - Produces typed List[PersonDetection] from services.face_contracts.
    - Coordinates with outputs/services/detector.py patterns without direct modification.
    """

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        config: Optional[Union[PersonDetectorConfig, Dict[str, Any]]] = None,
        cpu_threads: Optional[int] = None,
        auto_load: bool = True,
    ):
        if models_dir is not None:
            self.models_dir = Path(models_dir)
        else:
            self.models_dir = Path(__file__).resolve().parents[1] / "models"

        self.config = validate_person_detector_config(config or PersonDetectorConfig())
        self.device, self.warning = select_device(self.config.device)
        self.cpu_threads = configure_cpu_threads(cpu_threads or self.config.cpu_threads)

        self.model: Optional[Any] = None
        self.key: Optional[Tuple[str, str]] = None
        self.backend: str = self.config.backend
        self.resolution: int = self.config.resolution
        self.configured_resolution: int = self.config.resolution
        self.samples: int = 0
        self.inference_ema_ms: Optional[float] = None
        self.timings: Dict[str, float] = {}
        self._detection_counter: int = 0

        if auto_load:
            self.load()

    def load(self, config_override: Optional[Union[PersonDetectorConfig, Dict[str, Any]]] = None) -> None:
        """Load or reload model weights according to configuration."""
        if config_override is not None:
            self.config = validate_person_detector_config(config_override)

        key = (self.config.model_name, self.config.backend)
        if key == self.key and self.model is not None:
            if self.configured_resolution != self.config.resolution:
                self.configured_resolution = self.config.resolution
                self.resolution = self.config.resolution
            return

        backend = self.config.backend
        model_name = self.config.model_name
        self.warning = None

        # Build candidate weights path
        if not model_name.endswith((".pt", ".onnx", ".engine")):
            path = self.models_dir / f"{model_name}.pt"
        else:
            path = self.models_dir / model_name

        # Backend check and fallback
        if backend == "onnx":
            onnx_path = path.with_suffix(".onnx")
            if onnx_path.exists() and importlib.util.find_spec("onnxruntime") is not None:
                path = onnx_path
            else:
                self.warning = f"ONNX runtime or exported {onnx_path.name} unavailable; using PyTorch"
                backend = "pytorch"
                path = path.with_suffix(".pt")
        elif backend == "tensorrt":
            engine_path = path.with_suffix(".engine")
            if (
                engine_path.exists()
                and self.device.startswith("cuda")
                and importlib.util.find_spec("tensorrt") is not None
            ):
                path = engine_path
            else:
                self.warning = f"TensorRT runtime or exported {engine_path.name} unavailable; using PyTorch"
                backend = "pytorch"
                path = path.with_suffix(".pt")

        self.backend = backend
        self.resolution = self.config.resolution
        self.configured_resolution = self.config.resolution

        # Check model file existence or fall back to mock
        if path.exists() and YOLO is not None:
            try:
                self.model = YOLO(str(path), task="detect")
                # Warmup inference pass
                warmup_img = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
                self.model.predict(
                    warmup_img,
                    imgsz=self.resolution,
                    device=self.device,
                    quantize=16 if self.device.startswith("cuda") else 32,
                    verbose=False,
                    save=False,
                    classes=[PERSON_CLASS_ID],
                    conf=self.config.confidence_threshold,
                )
                if self.device == "cpu":
                    torch.set_num_threads(self.cpu_threads)
            except Exception as ex:
                if self.config.fallback_to_mock:
                    self.warning = f"Failed to initialize YOLO from {path}: {ex}; falling back to MockYOLO"
                    self.model = MockYOLO()
                else:
                    raise
        else:
            if self.config.fallback_to_mock:
                self.warning = f"Model file {path} not found; using MockYOLO fallback"
                self.model = MockYOLO()
            else:
                raise FileNotFoundError(f"Model file {path} does not exist and fallback_to_mock is False")

        self.samples = 0
        self.inference_ema_ms = None
        self.key = key

    def detect(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
        resolution: Optional[int] = None,
    ) -> List[PersonDetection]:
        """Run single-frame person detection.

        Args:
            image: BGR numpy ndarray (H, W, 3)
            confidence: Optional confidence threshold override
            iou: Optional IoU threshold override
            resolution: Optional inference resolution override

        Returns:
            List[PersonDetection] instances
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise ValueError("Input image must be a 3D numpy ndarray (H, W, 3)")

        results = self.detect_batch(
            [image],
            confidence=confidence,
            iou=iou,
            resolution=resolution,
        )
        return results[0] if results else []

    def detect_batch(
        self,
        images: Sequence[np.ndarray],
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
        resolution: Optional[int] = None,
    ) -> List[List[PersonDetection]]:
        """Run batched person detection across multiple frames.

        Args:
            images: List or sequence of BGR numpy ndarrays
            confidence: Optional confidence threshold override
            iou: Optional IoU threshold override
            resolution: Optional inference resolution override

        Returns:
            List of lists of PersonDetection instances (one list per frame)
        """
        if not images:
            return []

        # Ensure model is initialized
        if self.model is None:
            self.load()

        conf_thresh = float(confidence) if confidence is not None else self.config.confidence_threshold
        iou_thresh = float(iou) if iou is not None else self.config.iou_threshold

        # Check if any input frame is UHD/4K (width >= 1920 or height >= 1080)
        is_uhd = any(
            getattr(img, "shape", (0, 0))[1] >= 1920 or getattr(img, "shape", (0, 0))[0] >= 1080
            for img in images
        )
        self.is_uhd = is_uhd

        # Resolution handling
        if resolution is not None:
            imgsz = int(round(resolution / 32) * 32)
            if not self.config.adaptive:
                self.resolution = imgsz
        else:
            if is_uhd:
                # On UHD/4K frames, support adaptive resolution up to 1280 (or higher if configured), floor at 960
                max_uhd_res = max(1280, self.config.max_resolution)
                if not self.config.adaptive:
                    self.resolution = max(960, self.config.resolution)
                else:
                    self.resolution = max(960, min(max_uhd_res, max(self.resolution, 960)))
                imgsz = max(960, min(self.resolution, max_uhd_res))
            else:
                if not self.config.adaptive:
                    self.resolution = self.config.resolution
                else:
                    self.resolution = min(self.resolution, self.config.resolution)
                imgsz = self.resolution

        t0 = time.perf_counter()

        # Run model inference
        # Filter for class 0 (person) directly during YOLO NMS
        try:
            raw_results = self.model.predict(
                list(images),
                imgsz=imgsz,
                conf=min(conf_thresh, 0.05),  # Keep candidates for Python confidence filtering
                iou=iou_thresh,
                max_det=self.config.max_det,
                classes=[PERSON_CLASS_ID],
                device=self.device,
                quantize=16 if self.device.startswith("cuda") else 32,
                verbose=False,
                save=False,
            )
        except Exception as ex:
            logger.error(f"Inference failed on device {self.device}: {ex}")
            # Attempt CPU fallback if running on CUDA
            if self.device != "cpu":
                self.warning = f"CUDA inference failed ({ex}); falling back to CPU"
                self.device = "cpu"
                raw_results = self.model.predict(
                    list(images),
                    imgsz=imgsz,
                    conf=min(conf_thresh, 0.05),
                    iou=iou_thresh,
                    max_det=self.config.max_det,
                    classes=[PERSON_CLASS_ID],
                    device="cpu",
                    quantize=32,
                    verbose=False,
                    save=False,
                )
            else:
                raise

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.adapt(elapsed_ms)

        batch_detections: List[List[PersonDetection]] = []

        for idx, (img, result) in enumerate(zip(images, raw_results)):
            h, w = img.shape[:2]
            detections = self._extract_detections(result, (h, w), conf_thresh)
            batch_detections.append(detections)

        # Record speed timings from first result if available
        if raw_results and hasattr(raw_results[0], "speed"):
            self.timings = dict(getattr(raw_results[0], "speed", {}) or {})
        self.timings["batch_total_ms"] = elapsed_ms

        return batch_detections

    def _extract_detections(
        self,
        result: Any,
        image_shape: Tuple[int, int],
        conf_thresh: float
    ) -> List[PersonDetection]:
        """Parse ultralytics Result object into strict PersonDetection contracts."""
        h, w = image_shape
        boxes_obj = getattr(result, "boxes", None)
        if boxes_obj is None or len(boxes_obj) == 0:
            return []

        # Extract arrays handling both torch tensor and numpy interfaces
        if hasattr(boxes_obj, "cpu"):
            boxes_data = boxes_obj.cpu().numpy()
            xyxy_arr = boxes_data.xyxy
            conf_arr = boxes_data.conf
            cls_arr = boxes_data.cls
        elif hasattr(boxes_obj, "xyxy"):
            xyxy_arr = boxes_obj.xyxy
            conf_arr = boxes_obj.conf
            cls_arr = boxes_obj.cls
        elif isinstance(boxes_obj, np.ndarray) and boxes_obj.ndim == 2 and boxes_obj.shape[1] >= 6:
            xyxy_arr = boxes_obj[:, :4]
            conf_arr = boxes_obj[:, 4]
            cls_arr = boxes_obj[:, 5]
        else:
            return []

        # Convert to numpy arrays if needed
        xyxy_arr = np.asarray(xyxy_arr, dtype=np.float32)
        conf_arr = np.asarray(conf_arr, dtype=np.float32)
        cls_arr = np.asarray(cls_arr, dtype=np.float32)

        detections: List[PersonDetection] = []

        for box, conf, cls_id in zip(xyxy_arr, conf_arr, cls_arr):
            # Strict filter: only COCO Person class (0)
            if int(cls_id) != PERSON_CLASS_ID:
                continue

            conf_val = float(conf)
            # Strict filter: confidence threshold
            if conf_val < conf_thresh:
                continue

            # Clip coordinates to image boundary
            x1 = float(max(0.0, min(float(w), float(box[0]))))
            y1 = float(max(0.0, min(float(h), float(box[1]))))
            x2 = float(max(0.0, min(float(w), float(box[2]))))
            y2 = float(max(0.0, min(float(h), float(box[3]))))

            # Must have non-zero positive area
            if x2 <= x1 or y2 <= y1:
                continue

            self._detection_counter += 1
            det_id = f"det_{int(time.time() * 1000)}_{self._detection_counter}"

            detections.append(
                PersonDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf_val,
                    class_id=PERSON_CLASS_ID,
                    detection_id=det_id,
                )
            )

        return detections

    def adapt(self, elapsed_ms: float) -> None:
        """Dynamically adapt inference resolution to target frame budget."""
        self.samples += 1
        self.inference_ema_ms = (
            elapsed_ms
            if self.inference_ema_ms is None
            else 0.15 * elapsed_ms + 0.85 * self.inference_ema_ms
        )

        if self.config.adaptive and self.samples % 10 == 0:
            is_uhd = getattr(self, "is_uhd", False)
            min_floor = 960 if is_uhd else self.config.min_resolution
            max_ceiling = max(1280, self.config.max_resolution) if is_uhd else self.config.max_resolution

            if self.inference_ema_ms > self.config.target_ms * 1.2:
                # Reduce resolution to alleviate frame drop, keeping floor
                new_res = max(min_floor, self.resolution - 64)
                self.resolution = int(round(new_res / 32) * 32)
            elif self.inference_ema_ms < self.config.target_ms * 0.65:
                # Increase resolution to improve far-distance & crowded recall up to 1280 for UHD
                new_res = min(max_ceiling, self.resolution + 64)
                self.resolution = int(round(new_res / 32) * 32)

    def set_model(self, model: Any) -> None:
        """Inject a custom or mock model directly (useful for tests)."""
        self.model = model
        self.key = ("custom", "custom")

    @classmethod
    def from_detector_boxes(
        cls,
        boxes: Any,
        image_shape: Optional[Tuple[int, int]] = None,
        confidence_threshold: float = 0.0,
    ) -> List[PersonDetection]:
        """Convert ultralytics Boxes (such as detector.people_boxes) to List[PersonDetection].

        Coordinates with outputs/services/detector.py patterns without modifying detector.py.
        """
        if boxes is None:
            return []

        if hasattr(boxes, "cpu"):
            boxes_data = boxes.cpu().numpy()
            xyxy = boxes_data.xyxy
            conf = boxes_data.conf
            cls_arr = boxes_data.cls
        elif hasattr(boxes, "xyxy"):
            xyxy = boxes.xyxy
            conf = boxes.conf
            cls_arr = boxes.cls
        elif isinstance(boxes, np.ndarray) and boxes.ndim == 2 and boxes.shape[1] >= 6:
            xyxy = boxes[:, :4]
            conf = boxes[:, 4]
            cls_arr = boxes[:, 5]
        else:
            return []

        h, w = (1e9, 1e9)
        if image_shape is not None:
            h, w = image_shape[:2]
        elif hasattr(boxes, "orig_shape") and boxes.orig_shape is not None:
            h, w = boxes.orig_shape[:2]

        results = []
        counter = 0
        for box, c_conf, c_cls in zip(xyxy, conf, cls_arr):
            if int(c_cls) != PERSON_CLASS_ID:
                continue
            c_val = float(c_conf)
            if c_val < confidence_threshold:
                continue
            x1 = float(max(0.0, min(float(w), float(box[0]))))
            y1 = float(max(0.0, min(float(h), float(box[1]))))
            x2 = float(max(0.0, min(float(w), float(box[2]))))
            y2 = float(max(0.0, min(float(h), float(box[3]))))
            if x2 <= x1 or y2 <= y1:
                continue
            counter += 1
            results.append(
                PersonDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=c_val,
                    class_id=PERSON_CLASS_ID,
                    detection_id=f"det_ext_{counter}",
                )
            )
        return results

    @classmethod
    def from_detector(
        cls,
        detector_instance: Any,
        image_shape: Optional[Tuple[int, int]] = None,
        confidence_threshold: float = 0.0,
    ) -> List[PersonDetection]:
        """Extract PersonDetection list directly from an existing detector.Detector instance."""
        if hasattr(detector_instance, "people_boxes"):
            return cls.from_detector_boxes(
                detector_instance.people_boxes,
                image_shape=image_shape,
                confidence_threshold=confidence_threshold,
            )
        return []
