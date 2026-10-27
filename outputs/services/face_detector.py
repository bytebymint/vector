"""Real-time Facial Detection and Person Association Service.

Locates visible faces associated with existing person tracks using YuNet
(with OpenCV Haar Cascade fallback), enforces geometric upper-body association,
rejects ambiguous multi-person assignments and low-quality detections, and supports
selective inference cadences for high-throughput pipelines.
"""
from __future__ import annotations

import logging
import math
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np

try:
    from .face_contracts import (
        FaceObservation,
        PersonTrack,
        FramePacket,
        FacePose,
        FaceQualityResult,
    )
except ImportError:
    from services.face_contracts import (
        FaceObservation,
        PersonTrack,
        FramePacket,
        FacePose,
        FaceQualityResult,
    )

logger = logging.getLogger("face_detector")


def box_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Compute Intersection-over-Union (IoU) of two boxes (x1, y1, x2, y2)."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def point_in_box(
    px: float,
    py: float,
    box: Tuple[float, float, float, float],
    margin: float = 0.0,
) -> bool:
    """Check if point (px, py) is contained within box with optional margin."""
    return bool(
        (box[0] - margin) <= px <= (box[2] + margin)
        and (box[1] - margin) <= py <= (box[3] + margin)
    )


def is_valid_face(crop=None, landmarks=None, bbox=None, min_skin_fraction=None) -> bool:
    """Geometry/contrast sanity only; a trained second detector proves face content.

    No skin-colour threshold is used: illumination and skin tone are not evidence
    that a patch is (or is not) a face.
    """
    if bbox is None or len(bbox)!=4 or not np.isfinite(bbox).all():return False
    x1,y1,x2,y2=map(float,bbox)
    w,h=x2-x1,y2-y1
    if min(w,h)<=0:return False
    if crop is not None:
        if crop.size==0 or crop.ndim not in (2,3):return False
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY) if crop.ndim==3 else crop
        if float(np.std(gray))<3.:return False
    if landmarks is None or len(landmarks)!=5:return False
    pts=np.asarray(landmarks,dtype=np.float64)
    if pts.shape!=(5,2) or not np.isfinite(pts).all():return False
    if np.any(pts[:,0]<x1-.15*w) or np.any(pts[:,0]>x2+.15*w) or np.any(pts[:,1]<y1-.15*h) or np.any(pts[:,1]>y2+.15*h):return False
    eye=pts[1]-pts[0]
    distance=float(np.linalg.norm(eye))
    if eye[0]<=0 or distance<.08*w or distance>.85*w:return False
    horizontal=eye/distance
    down=np.array([-horizontal[1],horizontal[0]])
    upright=(pts-(pts[0]+pts[1])/2)@np.column_stack((horizontal,down))
    nose=upright[2,1];mouth=float(np.mean(upright[3:5,1]))
    if mouth<.12*h or nose<-.05*h or nose>mouth+.1*h:return False
    if np.linalg.norm(pts[4]-pts[3])<.04*w:return False
    return True


@dataclass
class FaceDetectorConfig:
    """Configuration options for FaceDetector model, geometry, and filtering."""

    model_path: Optional[str] = None
    backend: str = "auto"  # "auto", "yunet", "haar"
    score_threshold: float = 0.70
    roi_score_threshold: float = 0.65
    nms_threshold: float = 0.30
    top_k: int = 5000
    min_face_area: float = 64.0  # e.g. 8x8 pixels minimum
    min_face_width: float = 8.0
    min_face_height: float = 8.0
    min_aspect_ratio: float = 0.35  # width / height
    max_aspect_ratio: float = 2.80
    upper_region_ratio: float = 0.55  # Face center must be in top 55% of person height
    headroom_margin_ratio: float = 0.12  # Allow face center up to 12% headroom above person top
    horizontal_margin_ratio: float = 0.12  # Lateral tolerance around person bbox width
    max_face_to_person_area_ratio: float = 0.55  # Face cannot exceed 55% of person area
    ambiguity_iou_threshold: float = 0.25  # Person bbox IoU indicating heavy overlap
    ambiguity_affinity_ratio: float = 0.75  # 2nd best affinity / best affinity > 0.75 => ambiguous
    selective_cadence: int = 1  # Evaluate every N frames per track (1 = every frame)
    visibility_expansion_ratio: float = 1.25  # Force eval if person area increases by 25%
    min_person_width: float = 16.0  # Skip tracks smaller than this to save compute
    min_person_height: float = 32.0
    default_mode: str = "full_frame"  # "full_frame" or "roi"
    extract_face_crops: bool = True  # Populate face_crop in FaceObservation
    crop_margin_ratio: float = 0.25  # Expand crop to include hair, chin, and head outline
    crop_top_margin_ratio: float = 0.35  # Generous headroom / hair margin above top of face box
    crop_bottom_margin_ratio: float = 0.25  # Chin margin below bottom of face box
    crop_side_margin_ratio: float = 0.30  # Lateral ears / jawline margin
    max_inference_dim: int = 0  # 0 preserves full-frame detection; runtime ROI mode sets a limit
    fixed_input_size: int = 0  # Stable square DNN input avoids rebuilding the graph per person
    dnn_target: int = 0  # OpenCV DNN_TARGET_CPU; optional verified OpenCL FP16
    runtime: str = "opencv"


@dataclass
class TrackCadenceState:
    """Per-track history for selective evaluation cadence and visibility tracking."""

    last_frame_sequence: int = -1
    last_timestamp: float = -1.0
    last_area: float = 0.0
    eval_count: int = 0


class FaceDetector:
    """Primary facial detector with person track association.

    is_valid_face = staticmethod(is_valid_face)

    Integrates OpenCV YuNet ONNX with automatic fallback to Haar Cascade.
    Enforces geometric validation against person tracks, disambiguates
    overlapping detections, filters unreliable predictions, and supports
    selective inference scheduling.
    """

    def __init__(self, config: Optional[Union[FaceDetectorConfig, str, Path, Dict[str, Any]]] = None):
        if isinstance(config, (str, Path)):
            p = Path(config)
            model_file = p / "face_detection_yunet_2023mar.onnx" if p.is_dir() else p
            self.config = FaceDetectorConfig(model_path=str(model_file) if model_file.exists() else None)
        elif isinstance(config, dict):
            self.config = FaceDetectorConfig(**config)
        else:
            self.config = config or FaceDetectorConfig()
        self.active_backend: str = "none"
        self._yunet: Optional[cv2.FaceDetectorYN] = None
        self._haar: Optional[cv2.CascadeClassifier] = None
        self._track_states: Dict[int, TrackCadenceState] = {}

        self._initialize_backend()

    def _resolve_yunet_model_path(self) -> Optional[Path]:
        """Locate YuNet ONNX weights across common relative paths."""
        if self.config.model_path is not None:
            p = Path(self.config.model_path)
            try:
                resolved = p.resolve()
                if resolved.is_file():
                    return resolved
            except Exception:
                pass
            return None

        # Search default locations if model_path was not specified
        module_dir = Path(__file__).resolve().parent
        candidates = [
            module_dir.parent / "models" / "face_detection_yunet_2023mar.onnx",
            Path("outputs/models/face_detection_yunet_2023mar.onnx"),
            Path("models/face_detection_yunet_2023mar.onnx"),
            Path("face_detection_yunet_2023mar.onnx"),
        ]

        for p in candidates:
            try:
                resolved = p.resolve()
                if resolved.is_file():
                    return resolved
            except Exception:
                continue
        return None

    def _initialize_backend(self) -> None:
        """Initialize YuNet or fallback to Haar Cascade classifier."""
        desired_backend = self.config.backend.lower()

        if desired_backend in ("auto", "yunet"):
            model_path = self._resolve_yunet_model_path()
            if model_path is not None:
                try:
                    if self.config.runtime == 'onnxruntime':
                        from services.yunet_runtime import YuNetRuntime
                        self._yunet = YuNetRuntime(str(model_path), self.config.score_threshold,
                                                  self.config.nms_threshold, self.config.top_k)
                    else:
                        self._yunet = cv2.FaceDetectorYN.create(
                            str(model_path),
                            "",
                            (320, 320),
                            score_threshold=float(self.config.score_threshold),
                            nms_threshold=float(self.config.nms_threshold),
                            top_k=int(self.config.top_k),
                            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                            target_id=int(self.config.dnn_target),
                        )
                    self.active_backend = "yunet"
                    logger.info("FaceDetector initialized with YuNet: %s", model_path)
                    return
                except Exception as exc:
                    if self.config.runtime == 'onnxruntime':
                        logger.warning("ONNX Runtime unavailable; using OpenCV: %s", exc)
                        self.config.runtime = 'opencv'
                        self._initialize_backend()
                        return
                    logger.warning("Failed to initialize YuNet (%s). Attempting fallback.", exc)
                    if desired_backend == "yunet":
                        raise RuntimeError(f"Failed to initialize YuNet backend: {exc}") from exc
            else:
                if desired_backend == "yunet":
                    raise FileNotFoundError(
                        f"YuNet ONNX model not found (path={self.config.model_path})"
                    )
                logger.info("YuNet weights not found. Falling back to Haar Cascade.")

        # Fallback to Haar Cascade
        self._initialize_haar()

    def _initialize_haar(self) -> None:
        """Initialize Haar frontal face cascade classifier."""
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(haar_path):
            raise FileNotFoundError(f"OpenCV Haar cascade XML missing at {haar_path}")

        cascade = cv2.CascadeClassifier(haar_path)
        if cascade.empty():
            raise RuntimeError(f"Failed to load OpenCV Haar Cascade from {haar_path}")

        self._haar = cascade
        self.active_backend = "haar"
        logger.info("FaceDetector active backend: Haar Cascade (%s)", haar_path)

    def switch_backend(self, backend: str) -> None:
        """Switch between 'yunet' and 'haar' backends dynamically."""
        backend_lower = backend.lower()
        if backend_lower == "yunet":
            model_path = self._resolve_yunet_model_path()
            if not model_path:
                raise FileNotFoundError("YuNet ONNX model not found for manual switch")
            self._yunet = cv2.FaceDetectorYN.create(
                str(model_path),
                "",
                (320, 320),
                score_threshold=float(self.config.score_threshold),
                nms_threshold=float(self.config.nms_threshold),
                top_k=int(self.config.top_k),
            )
            self.active_backend = "yunet"
        elif backend_lower == "haar":
            self._initialize_haar()
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'yunet' or 'haar'.")

    # -------------------------------------------------------------------------
    # Track Parsing and Selective Inference Cadence
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_track(
        track: Any,
    ) -> Optional[Tuple[int, Tuple[float, float, float, float], float, bool]]:
        """Extract (track_id, (x1, y1, x2, y2), confidence, confirmed) from duck-typed tracks."""
        if track is None:
            return None

        track_id: int = 0
        bbox: Optional[Tuple[float, float, float, float]] = None
        confidence: float = 1.0
        confirmed: bool = True

        if hasattr(track, "track_id"):
            track_id = int(track.track_id)
        elif hasattr(track, "ident"):
            track_id = int(track.ident)
        elif hasattr(track, "id"):
            track_id = int(track.id)
        elif isinstance(track, dict):
            track_id = int(track.get("track_id", track.get("id", track.get("ident", 0))))

        if hasattr(track, "bbox"):
            b = track.bbox
            if hasattr(b, "tolist"):
                b = b.tolist()
            if len(b) >= 4:
                bbox = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        elif isinstance(track, dict) and "bbox" in track:
            b = track["bbox"]
            if len(b) >= 4:
                bbox = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        elif isinstance(track, (list, tuple)) and len(track) >= 4:
            bbox = (float(track[0]), float(track[1]), float(track[2]), float(track[3]))
            if len(track) >= 5:
                track_id = int(track[4])
            if len(track) >= 6:
                confidence = float(track[5])

        if hasattr(track, "confidence"):
            confidence = float(track.confidence)
        elif isinstance(track, dict) and "confidence" in track:
            confidence = float(track["confidence"])

        if hasattr(track, "confirmed"):
            confirmed = bool(track.confirmed)
        elif isinstance(track, dict) and "confirmed" in track:
            confirmed = bool(track["confirmed"])

        if bbox is None:
            return None

        # Validate box integrity
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None

        return (track_id, bbox, confidence, confirmed)

    def should_evaluate_track(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        frame_sequence: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Determine whether face detection should run on this track for current frame."""
        pw = bbox[2] - bbox[0]
        ph = bbox[3] - bbox[1]

        # Reject small or distant persons to conserve computation
        if pw < self.config.min_person_width or ph < self.config.min_person_height:
            return False

        if frame_sequence is None:
            return True

        state = self._track_states.get(track_id)
        if state is None:
            return True

        current_area = pw * ph
        # Trigger immediate evaluation if subject walked significantly closer
        if (
            state.last_area > 0.0
            and (current_area / state.last_area) >= self.config.visibility_expansion_ratio
        ):
            return True

        frames_since = frame_sequence - state.last_frame_sequence
        if frames_since >= self.config.selective_cadence:
            return True

        return False

    def _record_track_evaluation(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        frame_sequence: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Update track evaluation history state."""
        pw = bbox[2] - bbox[0]
        ph = bbox[3] - bbox[1]
        state = self._track_states.get(track_id)
        if state is None:
            state = TrackCadenceState()
            self._track_states[track_id] = state

        state.last_frame_sequence = frame_sequence if frame_sequence is not None else 0
        state.last_timestamp = timestamp if timestamp is not None else 0.0
        state.last_area = float(pw * ph)
        state.eval_count += 1

    def prune_tracks(self, active_track_ids: Set[int]) -> None:
        """Clean up cadence state for tracks that have terminated."""
        dead_ids = [tid for tid in self._track_states if tid not in active_track_ids]
        for tid in dead_ids:
            del self._track_states[tid]

    def reset_cadence(self) -> None:
        """Reset all selective cadence history."""
        self._track_states.clear()

    # -------------------------------------------------------------------------
    # Raw Detection Engine
    # -------------------------------------------------------------------------

    def _detect_raw(
        self,
        image: np.ndarray,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Execute underlying detector (YuNet or Haar) on an image / ROI crop."""
        if image is None or image.size == 0 or image.shape[0] < 5 or image.shape[1] < 5:
            return []

        h, w = image.shape[:2]
        fixed = int(self.config.fixed_input_size)
        if fixed > 0 and (h != fixed or w != fixed):
            ratio = min(1.0, fixed / float(max(h,w)))
            nh, nw = max(1,round(h*ratio)), max(1,round(w*ratio))
            small = cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA) if (nw,nh)!=(w,h) else image
            canvas = np.zeros((fixed,fixed,3),dtype=image.dtype)
            canvas[:nh,:nw] = cv2.cvtColor(small,cv2.COLOR_GRAY2BGR) if small.ndim==2 else small
            faces = self._detect_raw(canvas,score_threshold)
            mapped = []
            for face in faces:
                x1,y1,x2,y2 = face['bbox']
                if not (0 <= (x1+x2)/2 < nw and 0 <= (y1+y2)/2 < nh):
                    continue
                face['bbox'] = (max(0,x1)*w/nw,max(0,y1)*h/nh,min(nw,x2)*w/nw,min(nh,y2)*h/nh)
                face['landmarks'] = [(x*w/nw,y*h/nh) for x,y in face['landmarks']]
                mapped.append(face)
            return mapped
        limit = int(self.config.max_inference_dim)
        if limit > 0 and max(h, w) > max(64, limit):
            limit = max(64, limit)
            ratio = limit / float(max(h, w))
            nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
            target_shape = (nh, nw) if image.ndim == 2 else (nh, nw, image.shape[2])
            resize_buf = getattr(self, "_resize_buf", None)
            if resize_buf is None or resize_buf.shape != target_shape or resize_buf.dtype != image.dtype:
                resize_buf = np.empty(target_shape, dtype=image.dtype)
                self._resize_buf = resize_buf
            interp = cv2.INTER_LINEAR if max(h, w) >= 1920 else cv2.INTER_AREA
            small = cv2.resize(image, (nw, nh), dst=resize_buf, interpolation=interp)
            faces = self._detect_raw(small, score_threshold)
            sx, sy = w / small.shape[1], h / small.shape[0]
            for face in faces:
                x1, y1, x2, y2 = face['bbox']
                face['bbox'] = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
                face['landmarks'] = [(x * sx, y * sy) for x, y in face['landmarks']]
            return faces
        effective_threshold = (
            score_threshold
            if score_threshold is not None
            else self.config.score_threshold
        )
        detections: List[Dict[str, Any]] = []

        if self.active_backend == "yunet" and self._yunet is not None:
            bgr_image = (
                cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
            )
            if not bgr_image.flags["C_CONTIGUOUS"]:
                bgr_image = np.ascontiguousarray(bgr_image)

            try:
                self._yunet.setInputSize((int(w), int(h)))
                self._yunet.setScoreThreshold(float(effective_threshold))
                _, raw_faces = self._yunet.detect(bgr_image)
            except Exception as exc:
                logger.warning("YuNet execution failed on image (%dx%d): %s", w, h, exc)
                if self.config.runtime == 'onnxruntime':
                    self.config.runtime = 'opencv'
                    self._initialize_backend()
                    return self._detect_raw(image, score_threshold)
                if self.config.dnn_target != cv2.dnn.DNN_TARGET_CPU:
                    logger.warning("Retrying face inference on CPU")
                    self.config.dnn_target = cv2.dnn.DNN_TARGET_CPU
                    self._initialize_backend()
                    return self._detect_raw(image, score_threshold)
                raise RuntimeError(f"YuNet inference failed: {exc}") from exc

            if raw_faces is not None and len(raw_faces) > 0:
                for row in raw_faces:
                    fx, fy, fw, fh = (
                        float(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                    )
                    score = float(row[14])
                    landmarks = [
                        (float(row[k]), float(row[k + 1])) for k in range(4, 14, 2)
                    ]
                    detections.append({
                        "bbox": (fx, fy, fx + fw, fy + fh),
                        "confidence": score,
                        "landmarks": landmarks,
                    })
        elif self.active_backend == "haar" and self._haar is not None:
            gray_image = (
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            )
            if not gray_image.flags["C_CONTIGUOUS"]:
                gray_image = np.ascontiguousarray(gray_image)

            try:
                min_w = int(max(4.0, self.config.min_face_width))
                min_h = int(max(4.0, self.config.min_face_height))
                rects, _, weights = self._haar.detectMultiScale3(
                    gray_image,
                    scaleFactor=1.05,
                    minNeighbors=2,
                    minSize=(min_w, min_h),
                    outputRejectLevels=True,
                )
            except Exception as exc:
                logger.warning("Haar execution failed on image (%dx%d): %s", w, h, exc)
                rects, weights = (), ()

            if len(rects) > 0:
                has_weights = hasattr(weights, "__len__") and len(weights) == len(rects)
                for i, (hx, hy, hw, hh) in enumerate(rects):
                    fx, fy, fw, fh = float(hx), float(hy), float(hw), float(hh)
                    if has_weights:
                        w_val = float(weights[i])
                        # Sigmoidal mapping from Haar decision log-odds to confidence
                        score = float(1.0 / (1.0 + math.exp(-w_val / 2.0)))
                    else:
                        score = 0.85

                    detections.append({
                        "bbox": (fx, fy, fx + fw, fy + fh),
                        "confidence": score,
                        "landmarks": [],  # Haar provides no facial landmarks
                    })

        # Apply unreliable detection rejection filters
        valid_detections: List[Dict[str, Any]] = []
        for det in detections:
            bbox = det["bbox"]
            conf = det["confidence"]
            fw = bbox[2] - bbox[0]
            fh = bbox[3] - bbox[1]
            area = fw * fh

            # 1. Confidence threshold
            if conf < effective_threshold:
                continue

            # 2. Minimum pixel area and dimensions
            if (
                area < self.config.min_face_area
                or fw < self.config.min_face_width
                or fh < self.config.min_face_height
            ):
                continue

            # 3. Aspect ratio sanity (reject slivers)
            ar = fw / max(fh, 1e-4)
            if not (self.config.min_aspect_ratio <= ar <= self.config.max_aspect_ratio):
                continue

            # 4. Image bounds sanity
            if bbox[2] <= 0 or bbox[3] <= 0 or bbox[0] >= w or bbox[1] >= h:
                continue

            # 5. Landmark geometry sanity; semantic verification happens separately
            crop = None
            if image is not None and image.size > 0:
                cx1 = max(0, min(w, int(round(bbox[0]))))
                cy1 = max(0, min(h, int(round(bbox[1]))))
                cx2 = max(0, min(w, int(round(bbox[2]))))
                cy2 = max(0, min(h, int(round(bbox[3]))))
                if cy2 > cy1 and cx2 > cx1:
                    crop = image[cy1:cy2, cx1:cx2]

            if not is_valid_face(crop=crop, landmarks=det.get("landmarks"), bbox=bbox):
                continue

            valid_detections.append(det)

        return valid_detections

    # -------------------------------------------------------------------------
    # Association and Ambiguity Resolution
    # -------------------------------------------------------------------------

    def _evaluate_face_person_candidate(
        self,
        face_bbox: Tuple[float, float, float, float],
        person_bbox: Tuple[float, float, float, float],
    ) -> Optional[float]:
        """Verify face satisfies upper-body geometry for a person, returning affinity score.

        Returns None if face fails spatial containment, vertical upper-half constraint,
        or scale plausibility.
        """
        fx1, fy1, fx2, fy2 = face_bbox
        px1, py1, px2, py2 = person_bbox

        fw = fx2 - fx1
        fh = fy2 - fy1
        pw = px2 - px1
        ph = py2 - py1

        if fw <= 0.0 or fh <= 0.0 or pw <= 0.0 or ph <= 0.0:
            return None

        face_area = fw * fh
        person_area = pw * ph

        # Scale check: Face cannot exceed reasonable upper bound of total person area
        if face_area > self.config.max_face_to_person_area_ratio * person_area:
            return None
        if fw > pw * 1.30 or fh > ph * 0.70:
            return None

        fcx = (fx1 + fx2) / 2.0
        fcy = (fy1 + fy2) / 2.0

        # Horizontal alignment check
        margin_x = self.config.horizontal_margin_ratio * pw
        if not ((px1 - margin_x) <= fcx <= (px2 + margin_x)):
            return None

        # Vertical upper region check (face center must be in upper half / headroom)
        headroom = self.config.headroom_margin_ratio * ph
        y_min = py1 - headroom
        y_max = py1 + self.config.upper_region_ratio * ph
        if not (y_min <= fcy <= y_max):
            return None

        # Overlap check: Face must physically intersect the person bounding box
        inter_x1 = max(fx1, px1)
        inter_y1 = max(fy1, py1)
        inter_x2 = min(fx2, px2)
        inter_y2 = min(fy2, py2)
        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        if inter_area < 0.30 * face_area:
            return None

        # Affinity metric: Distance from expected head center (top 18% of person box)
        expected_hx = px1 + 0.5 * pw
        expected_hy = py1 + 0.18 * ph
        dx = (fcx - expected_hx) / max(pw, 1.0)
        dy = (fcy - expected_hy) / max(ph, 1.0)
        norm_dist = math.sqrt(dx * dx + (2.5 * dy) * (2.5 * dy))
        affinity = float(1.0 / (1.0 + norm_dist))
        return affinity

    def _associate_faces_with_tracks(
        self,
        raw_faces: List[Dict[str, Any]],
        parsed_tracks: List[Tuple[int, Tuple[float, float, float, float], float, bool]],
        timestamp: float,
        frame: Optional[np.ndarray] = None,
    ) -> List[FaceObservation]:
        """Associate detections to tracks with ambiguity and overlapping-person rejection."""
        if not raw_faces or not parsed_tracks:
            return []

        # Step 1: Collect valid candidates for each face
        # face_candidates[j] = [(track_id, affinity, person_bbox), ...]
        face_candidates: List[List[Tuple[int, float, Tuple[float, float, float, float]]]] = []
        for face in raw_faces:
            cands = []
            for t_id, p_bbox, _, _ in parsed_tracks:
                affinity = self._evaluate_face_person_candidate(face["bbox"], p_bbox)
                if affinity is not None:
                    cands.append((t_id, affinity, p_bbox))
            cands.sort(key=lambda c: c[1], reverse=True)
            face_candidates.append(cands)

        # Step 2: Ambiguity rejection
        # If multiple people overlap heavily and ownership is unclear, reject
        accepted_matches: List[Tuple[int, int, float]] = []  # (face_idx, track_id, affinity)

        for j, cands in enumerate(face_candidates):
            if not cands:
                continue

            face = raw_faces[j]
            fcx = (face["bbox"][0] + face["bbox"][2]) / 2.0
            fcy = (face["bbox"][1] + face["bbox"][3]) / 2.0

            if len(cands) == 1:
                t_id_1, aff_1, p_bbox_1 = cands[0]
                # Check if p_bbox_1 heavily overlaps with another track in the scene
                # that also encloses this face center
                is_ambiguous = False
                for other_id, other_bbox, _, _ in parsed_tracks:
                    if other_id == t_id_1:
                        continue
                    overlap_iou = box_iou(p_bbox_1, other_bbox)
                    if overlap_iou >= self.config.ambiguity_iou_threshold:
                        if point_in_box(fcx, fcy, other_bbox, margin=0.05 * (other_bbox[2] - other_bbox[0])):
                            # Overlapping person also contains or is adjacent to face center
                            is_ambiguous = True
                            break

                if not is_ambiguous:
                    accepted_matches.append((j, t_id_1, aff_1))
            else:
                # 2 or more candidate tracks qualify for this face
                t_id_1, aff_1, p_bbox_1 = cands[0]
                t_id_2, aff_2, p_bbox_2 = cands[1]
                overlap_iou = box_iou(p_bbox_1, p_bbox_2)
                affinity_ratio = aff_2 / max(aff_1, 1e-6)

                # If the two persons overlap heavily or their affinity scores are close
                if (
                    overlap_iou >= self.config.ambiguity_iou_threshold
                    or affinity_ratio >= self.config.ambiguity_affinity_ratio
                ):
                    # Unclear ownership between overlapping persons -> REJECT
                    continue

                # Top candidate is substantially clearer and persons are distinct
                accepted_matches.append((j, t_id_1, aff_1))

        # Step 3: Enforce 1-to-1 matching (at most 1 face per track per frame)
        track_to_best_match: Dict[int, Tuple[int, float]] = {}
        for face_idx, t_id, aff in accepted_matches:
            face_conf = raw_faces[face_idx]["confidence"]
            match_score = aff * face_conf
            if t_id not in track_to_best_match or match_score > track_to_best_match[t_id][1]:
                track_to_best_match[t_id] = (face_idx, match_score)

        # Step 4: Construct FaceObservation instances
        final_observations: List[FaceObservation] = []
        track_map = {t_id: p_bbox for t_id, p_bbox, _, _ in parsed_tracks}
        for t_id, (face_idx, _) in track_to_best_match.items():
            face = raw_faces[face_idx]
            fb = face["bbox"]

            p_bbox = track_map.get(t_id)
            relative_box = None
            if p_bbox is not None:
                pw = max(1.0, float(p_bbox[2] - p_bbox[0]))
                ph = max(1.0, float(p_bbox[3] - p_bbox[1]))
                rx1 = (float(fb[0]) - float(p_bbox[0])) / pw
                ry1 = (float(fb[1]) - float(p_bbox[1])) / ph
                rx2 = (float(fb[2]) - float(p_bbox[0])) / pw
                ry2 = (float(fb[3]) - float(p_bbox[1])) / ph
                relative_box = (float(rx1), float(ry1), float(rx2), float(ry2))

            crop_img = None
            if self.config.extract_face_crops and frame is not None:
                crop_img = self._crop_head_portrait(frame, fb)

            obs_id = f"face_{t_id}_{int(timestamp * 1000)}_{uuid.uuid4().hex[:6]}"
            obs = FaceObservation(
                observation_id=obs_id,
                track_id=t_id,
                timestamp=float(timestamp),
                bbox=(float(fb[0]), float(fb[1]), float(fb[2]), float(fb[3])),
                confidence=float(face["confidence"]),
                landmarks=[(float(p[0]), float(p[1])) for p in face["landmarks"]],
                face_crop=crop_img,
                relative_box=relative_box,
            )
            obs.relative_box = relative_box
            final_observations.append(obs)

        return final_observations

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def _crop_head_portrait(
        self,
        frame: np.ndarray,
        fb: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """Extract generous head portrait framing including hair, chin, and ears.

        Uses 0.35 above top of face box (headroom/hair), 0.25 below chin, and
        0.30 lateral margins for jawline/ears. Never produces tight claustrophobic slivers.
        """
        if frame is None or frame.size == 0:
            return None
        fh, fw = frame.shape[:2]
        fb_w = max(1.0, float(fb[2] - fb[0]))
        fb_h = max(1.0, float(fb[3] - fb[1]))

        top_margin = float(getattr(self.config, "crop_top_margin_ratio", 0.35))
        bottom_margin = float(getattr(self.config, "crop_bottom_margin_ratio", 0.25))
        side_margin = float(getattr(self.config, "crop_side_margin_ratio", 0.30))

        cy1 = max(0, min(fh, int(round(fb[1] - top_margin * fb_h))))
        cy2 = max(0, min(fh, int(round(fb[3] + bottom_margin * fb_h))))
        cx1 = max(0, min(fw, int(round(fb[0] - side_margin * fb_w))))
        cx2 = max(0, min(fw, int(round(fb[2] + side_margin * fb_w))))

        if cy2 > cy1 and cx2 > cx1:
            return frame[cy1:cy2, cx1:cx2].copy()
        return None

    def associate_faces(
        self,
        raw_faces: List[Dict[str, Any]],
        tracks: Sequence[Any],
        timestamp: float = 0.0,
        frame: Optional[np.ndarray] = None,
    ) -> List[FaceObservation]:
        """Public helper to associate raw face detections with track candidates."""
        parsed_tracks = []
        for t in tracks:
            parsed = self._parse_track(t)
            if parsed is not None:
                parsed_tracks.append(parsed)
        return self._associate_faces_with_tracks(raw_faces, parsed_tracks, timestamp, frame)

    def detect(
        self,
        frame: Union[np.ndarray, FramePacket],
        tracks: Optional[Sequence[Any]] = None,
        timestamp: Optional[float] = None,
        frame_sequence: Optional[int] = None,
        mode: Optional[str] = None,
        associate_only: bool = True,
    ) -> List[FaceObservation]:
        """Detect faces and associate them with person tracks.

        Args:
            frame: Raw BGR/grayscale image numpy array or FramePacket contract instance.
            tracks: Optional sequence of PersonTrack or dicts representing active person tracks.
            timestamp: Frame acquisition timestamp in seconds (extracted from FramePacket if provided).
            frame_sequence: Integer frame sequence number for cadence management.
            mode: Detection mode ('roi', 'full_frame', or None for auto/default).
            associate_only: If True and tracks are given, only return associated observations.
                           If False or tracks is None, unassociated faces have track_id=-1.

        Returns:
            List of FaceObservation instances conforming to services.face_contracts.
        """
        # Unpack FramePacket if supplied
        if isinstance(frame, FramePacket):
            raw_image = frame.image
            if timestamp is None:
                timestamp = frame.timestamp
            if frame_sequence is None:
                frame_sequence = frame.sequence
        else:
            raw_image = frame

        if raw_image is None or raw_image.size == 0:
            return []

        ts = float(timestamp if timestamp is not None else 0.0)
        img_h, img_w = raw_image.shape[:2]

        parsed_tracks: List[Tuple[int, Tuple[float, float, float, float], float, bool]] = []
        if tracks is not None:
            for t in tracks:
                parsed = self._parse_track(t)
                if parsed is not None:
                    parsed_tracks.append(parsed)

        active_mode = (mode or self.config.default_mode).lower()
        if not parsed_tracks and active_mode != "full_frame":
            active_mode = "full_frame"

        # ---------------------------------------------------------------------
        # Execution Path 1: ROI Mode (Upper torso crop per track)
        # ---------------------------------------------------------------------
        if active_mode == "roi" and parsed_tracks:
            candidate_raw_faces: List[Dict[str, Any]] = []

            for t_id, p_bbox, _, _ in parsed_tracks:
                if not self.should_evaluate_track(t_id, p_bbox, frame_sequence, ts):
                    continue

                pw = p_bbox[2] - p_bbox[0]
                ph = p_bbox[3] - p_bbox[1]
                margin_x = self.config.horizontal_margin_ratio * pw
                headroom = self.config.headroom_margin_ratio * ph

                roi_x1 = max(0, int(p_bbox[0] - margin_x))
                roi_y1 = max(0, int(p_bbox[1] - headroom))
                roi_x2 = min(img_w, int(p_bbox[2] + margin_x))
                roi_y2 = min(img_h, int(p_bbox[1] + self.config.upper_region_ratio * ph + headroom))

                if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
                    continue

                roi_crop = raw_image[roi_y1:roi_y2, roi_x1:roi_x2]
                roi_faces = self._detect_raw(
                    roi_crop,
                    score_threshold=self.config.roi_score_threshold,
                )
                self._record_track_evaluation(t_id, p_bbox, frame_sequence, ts)

                # Translate ROI coordinates back to full-frame space
                for f in roi_faces:
                    rb = f["bbox"]
                    full_bbox = (
                        rb[0] + roi_x1,
                        rb[1] + roi_y1,
                        rb[2] + roi_x1,
                        rb[3] + roi_y1,
                    )
                    full_landmarks = [
                        (lx + roi_x1, ly + roi_y1) for (lx, ly) in f["landmarks"]
                    ]
                    candidate_raw_faces.append({
                        "bbox": full_bbox,
                        "confidence": f["confidence"],
                        "landmarks": full_landmarks,
                    })

            # Run global track association & ambiguity resolution on all candidate faces
            return self._associate_faces_with_tracks(
                candidate_raw_faces, parsed_tracks, ts, raw_image
            )

        # ---------------------------------------------------------------------
        # Execution Path 2: Full Frame Detection
        # ---------------------------------------------------------------------
        all_raw_faces = self._detect_raw(raw_image, score_threshold=self.config.score_threshold)

        if parsed_tracks:
            # Filter tracks evaluated on this sequence
            eval_tracks = [
                pt for pt in parsed_tracks
                if self.should_evaluate_track(pt[0], pt[1], frame_sequence, ts)
            ]
            for pt in eval_tracks:
                self._record_track_evaluation(pt[0], pt[1], frame_sequence, ts)

            associated_obs = self._associate_faces_with_tracks(
                all_raw_faces, eval_tracks, ts, raw_image
            )

            if associate_only:
                return associated_obs

            # If associate_only is False, append unassociated faces with track_id = -1
            associated_bboxes = {obs.bbox for obs in associated_obs}
            extra_obs = []
            for face in all_raw_faces:
                if face["bbox"] not in associated_bboxes:
                    fb = face["bbox"]
                    crop_img = None
                    if self.config.extract_face_crops:
                        crop_img = self._crop_head_portrait(raw_image, fb)

                    extra_obs.append(
                        FaceObservation(
                            observation_id=f"face_unassigned_{uuid.uuid4().hex[:8]}",
                            track_id=-1,
                            timestamp=ts,
                            bbox=fb,
                            confidence=face["confidence"],
                            landmarks=face["landmarks"],
                            face_crop=crop_img,
                        )
                    )
            return associated_obs + extra_obs

        # No tracks provided: return unassociated detections with track_id = -1
        results: List[FaceObservation] = []
        for face in all_raw_faces:
            fb = face["bbox"]
            crop_img = None
            if self.config.extract_face_crops:
                crop_img = self._crop_head_portrait(raw_image, fb)

            results.append(
                FaceObservation(
                    observation_id=f"face_unassigned_{uuid.uuid4().hex[:8]}",
                    track_id=-1,
                    timestamp=ts,
                    bbox=fb,
                    confidence=face["confidence"],
                    landmarks=face["landmarks"],
                    face_crop=crop_img,
                )
            )
        return results

    def detect_regions_mosaic(
        self,
        image: np.ndarray,
        regions: Sequence[Tuple[float, float, float, float]],
        score_threshold: Optional[float] = None,
        cell_size: int = 256,
        min_face_pixels: int = 16,
        timestamp: float = 0.0,
        frame_sequence: int = -1,
    ) -> List[FaceObservation]:
        """Batch-detect faces across multiple image ROIs in a single grid mosaic pass."""
        if image is None or image.size == 0 or not regions:
            return []

        h, w = image.shape[:2]
        if h < 16 or w < 16:
            return []

        cell_dim = max(16, int(cell_size))
        valid_regions: List[Tuple[int, int, int, int]] = []
        for reg in regions:
            if reg is None or len(reg) < 4:
                continue
            rx1, ry1, rx2, ry2 = reg[0], reg[1], reg[2], reg[3]
            if not (
                math.isfinite(rx1)
                and math.isfinite(ry1)
                and math.isfinite(rx2)
                and math.isfinite(ry2)
            ):
                continue
            x1 = max(0, int(rx1))
            y1 = max(0, int(ry1))
            x2 = min(w, int(rx2))
            y2 = min(h, int(ry2))
            if (x2 - x1) < 16 or (y2 - y1) < 16:
                continue
            valid_regions.append((x1, y1, x2, y2))

        if not valid_regions:
            return []

        num_cells = len(valid_regions)
        cols = math.ceil(math.sqrt(num_cells))
        rows = math.ceil(num_cells / cols)
        pad = 16

        scaled_dims: List[Tuple[int, int, int, int, int, int]] = []
        col_widths = [0] * cols
        row_heights = [0] * rows
        for idx, (x1, y1, x2, y2) in enumerate(valid_regions):
            cw = x2 - x1
            ch = y2 - y1
            ratio = min(1.0, float(cell_dim) / float(max(ch, cw)))
            nw = max(1, round(cw * ratio))
            nh = max(1, round(ch * ratio))
            row = idx // cols
            col = idx % cols
            if nw > col_widths[col]:
                col_widths[col] = nw
            if nh > row_heights[row]:
                row_heights[row] = nh
            scaled_dims.append((x1, y1, cw, ch, nw, nh))

        col_offsets = [0] * cols
        for c in range(1, cols):
            col_offsets[c] = col_offsets[c - 1] + col_widths[c - 1] + pad
        row_offsets = [0] * rows
        for r in range(1, rows):
            row_offsets[r] = row_offsets[r - 1] + row_heights[r - 1] + pad

        mosaic_w = col_offsets[-1] + col_widths[-1]
        mosaic_h = row_offsets[-1] + row_heights[-1]

        mosaic_shape = (mosaic_h, mosaic_w, 3)
        mosaic = getattr(self, "_mosaic_buf", None)
        if mosaic is None or mosaic.shape != mosaic_shape:
            mosaic = np.zeros(mosaic_shape, dtype=np.uint8)
            self._mosaic_buf = mosaic
        else:
            mosaic.fill(0)

        cells: List[Tuple[int, int, int, int, int, int, int, int]] = []
        for idx, (x1, y1, cw, ch, nw, nh) in enumerate(scaled_dims):
            row = idx // cols
            col = idx % cols
            ox = col_offsets[col]
            oy = row_offsets[row]
            crop = image[y1 : y1 + ch, x1 : x1 + cw]
            if crop.ndim == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            dst_slice = mosaic[oy : oy + nh, ox : ox + nw]
            if (nw, nh) != (cw, ch):
                resized = cv2.resize(crop, (nw, nh), dst=dst_slice, interpolation=cv2.INTER_AREA)
                if resized is not dst_slice:
                    mosaic[oy : oy + nh, ox : ox + nw] = resized
            else:
                dst_slice[:] = crop
            cells.append((x1, y1, cw, ch, ox, oy, nw, nh))

        threshold = (
            float(score_threshold)
            if score_threshold is not None
            else float(self.config.roi_score_threshold)
        )
        old_max_dim = self.config.max_inference_dim
        old_fixed = self.config.fixed_input_size
        self.config.max_inference_dim = 0
        self.config.fixed_input_size = 0
        try:
            raw_detections = self._detect_raw(mosaic, threshold)
        finally:
            self.config.max_inference_dim = old_max_dim
            self.config.fixed_input_size = old_fixed

        if not raw_detections:
            return []

        ts = float(timestamp)
        min_px = float(min_face_pixels)
        results: List[FaceObservation] = []
        for raw in raw_detections:
            mx1, my1, mx2, my2 = raw["bbox"]
            cx = 0.5 * (mx1 + mx2)
            cy = 0.5 * (my1 + my2)
            if cx < 0.0 or cy < 0.0 or cx >= mosaic_w or cy >= mosaic_h:
                continue
            col = -1
            for c in range(cols):
                if col_offsets[c] <= cx < col_offsets[c] + col_widths[c]:
                    col = c
                    break
            if col < 0:
                continue
            row = -1
            for r in range(rows):
                if row_offsets[r] <= cy < row_offsets[r] + row_heights[r]:
                    row = r
                    break
            if row < 0:
                continue
            cell_idx = row * cols + col
            if cell_idx >= num_cells:
                continue

            x1, y1, cw, ch, ox, oy, nw, nh = cells[cell_idx]
            if not (ox <= cx < ox + nw and oy <= cy < oy + nh):
                continue

            lx1 = float(mx1 - ox)
            ly1 = float(my1 - oy)
            lx2 = float(mx2 - ox)
            ly2 = float(my2 - oy)
            local_bbox = (lx1, ly1, lx2, ly2)
            raw_lms = raw.get("landmarks") or []
            local_landmarks = [(float(lx - ox), float(ly - oy)) for lx, ly in raw_lms]

            fa = max(0, int(round(lx1)))
            fb = max(0, int(round(ly1)))
            fc = min(nw, int(round(lx2)))
            fd = min(nh, int(round(ly2)))
            cell_face_crop = (
                mosaic[oy + fb : oy + fd, ox + fa : ox + fc]
                if (fd > fb and fc > fa)
                else None
            )
            if not is_valid_face(crop=cell_face_crop, landmarks=local_landmarks, bbox=local_bbox):
                continue

            sx = float(cw) / float(nw)
            sy = float(ch) / float(nh)
            nat_x1 = float(x1 + max(0.0, lx1) * sx)
            nat_y1 = float(y1 + max(0.0, ly1) * sy)
            nat_x2 = float(x1 + min(float(nw), lx2) * sx)
            nat_y2 = float(y1 + min(float(nh), ly2) * sy)
            bw = nat_x2 - nat_x1
            bh = nat_y2 - nat_y1
            if min(bw, bh) < min_px:
                continue

            nat_landmarks = [
                (float(x1 + lx * sx), float(y1 + ly * sy)) for lx, ly in local_landmarks
            ]
            crop_img = (
                self._crop_head_portrait(image, (nat_x1, nat_y1, nat_x2, nat_y2))
                if self.config.extract_face_crops
                else None
            )
            results.append(
                FaceObservation(
                    observation_id=uuid.uuid4().hex,
                    track_id=-1,
                    timestamp=ts,
                    bbox=(nat_x1, nat_y1, nat_x2, nat_y2),
                    confidence=float(raw["confidence"]),
                    landmarks=nat_landmarks,
                    face_crop=crop_img,
                    frame_sequence=frame_sequence,
                )
            )

        return results
