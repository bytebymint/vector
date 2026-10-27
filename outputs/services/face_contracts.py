"""Shared interface contracts and data structures for the Multi-Person Facial Capture System.

All subagents and pipeline modules communicate using these strict data schemas.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class FramePacket:
    """Acquired camera frame preserved with original resolution and timing metadata."""
    sequence: int
    timestamp: float
    image: np.ndarray
    source_id: str
    epoch: str
    width: int
    height: int
    pts: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "source_id": self.source_id,
            "epoch": str(self.epoch),
            "width": self.width,
            "height": self.height,
            "pts": self.pts,
        }


@dataclass
class PersonDetection:
    """Localized person detection from detection model."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    confidence: float
    class_id: int = 0  # 0 for COCO person
    detection_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": [round(float(c), 2) for c in self.bbox],
            "confidence": round(float(self.confidence), 4),
            "class_id": self.class_id,
            "detection_id": self.detection_id,
        }


@dataclass
class PersonTrack:
    """Anonymous multi-object session track for an observed individual."""
    track_id: int
    session_id: str
    first_seen: float
    last_seen: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    hits: int = 1
    state: str = "active"  # "active", "lost", "ended"
    confirmed: bool = False
    speed_px_s: float = 0.0
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)  # (t, cx, cy)
    locked_face: Optional[Dict[str, Any]] = None
    face_bbox: Optional[Tuple[float, float, float, float]] = None
    face_confidence: float = 0.0
    face_landmarks: List[Tuple[float, float]] = field(default_factory=list)
    face_relative_box: Optional[Tuple[float, float, float, float]] = None
    face_last_seen: Optional[float] = None
    velocity: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        v = self.velocity
        if v is None:
            self.velocity = (0.0, 0.0)
        elif not isinstance(v, tuple) or len(v) != 2:
            self.velocity = (float(v[0]), float(v[1])) if len(v) >= 2 else (0.0, 0.0)
        if not isinstance(self.bbox, tuple):
            b = self.bbox
            self.bbox = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        if self.face_bbox is not None and not isinstance(self.face_bbox, tuple):
            fb = self.face_bbox
            self.face_bbox = (float(fb[0]), float(fb[1]), float(fb[2]), float(fb[3]))
        if self.face_relative_box is not None and not isinstance(self.face_relative_box, tuple):
            rb = self.face_relative_box
            self.face_relative_box = (float(rb[0]), float(rb[1]), float(rb[2]), float(rb[3]))

    @property
    def duration(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    def update_locked_face_from_body(self, image_bounds: Optional[Tuple[int, int]] = None) -> None:
        """Update absolute face coordinates from current body bounding box and relative offset.

        Remains active as long as track state is not 'deleted' or 'ended', unless the
        bounding box has completely exited the image bounds.
        """
        rel = self.face_relative_box
        if rel is None:
            return

        if self.state in ("deleted", "ended"):
            self.locked_face = None
            self.face_bbox = None
            return

        px1, py1, px2, py2 = self.bbox

        # Check if person has completely left image bounds (w, h)
        if image_bounds is not None:
            w, h = image_bounds
            if px2 <= 0 or py2 <= 0 or px1 >= w or py1 >= h:
                self.locked_face = None
                self.face_bbox = None
                return

        pw = px2 - px1
        if pw < 1.0:
            pw = 1.0
        ph = py2 - py1
        if ph < 1.0:
            ph = 1.0
        r0, r1, r2, r3 = rel
        fb0 = px1 + r0 * pw
        fb1 = py1 + r1 * ph
        fb2 = px1 + r2 * pw
        fb3 = py1 + r3 * ph
        self.face_bbox = (fb0, fb1, fb2, fb3)

        # Fast in-place update of relative landmarks and locked_face dict
        lf = self.locked_face
        if lf is not None:
            b = lf["bbox"]
            b[0] = round(fb0, 2)
            b[1] = round(fb1, 2)
            b[2] = round(fb2, 2)
            b[3] = round(fb3, 2)
            rel_lm = lf.get("relative_landmarks")
            if rel_lm:
                self.face_landmarks = [
                    (px1 + rlx * pw, py1 + rly * ph) for rlx, rly in rel_lm
                ]
                lm_out = lf.get("landmarks")
                if isinstance(lm_out, list) and len(lm_out) == len(self.face_landmarks):
                    for idx, (lx, ly) in enumerate(self.face_landmarks):
                        pt = lm_out[idx]
                        pt[0] = round(lx, 2)
                        pt[1] = round(ly, 2)
                else:
                    lf["landmarks"] = [
                        [round(lx, 2), round(ly, 2)] for lx, ly in self.face_landmarks
                    ]
        else:
            self.locked_face = {
                "bbox": [round(fb0, 2), round(fb1, 2), round(fb2, 2), round(fb3, 2)],
                "confidence": round(self.face_confidence, 4),
                "landmarks": [[round(p[0], 2), round(p[1], 2)] for p in self.face_landmarks] if self.face_landmarks else [],
                "relative_box": [round(r0, 4), round(r1, 4), round(r2, 4), round(r3, 4)],
                "last_seen": self.face_last_seen,
                "is_locked": True,
                "relative_landmarks": [],
            }

    def lock_face(
        self,
        face_bbox: Tuple[float, float, float, float],
        confidence: float = 1.0,
        landmarks: Optional[List[Tuple[float, float]]] = None,
        timestamp: Optional[float] = None,
        alpha: float = 0.3,
    ) -> None:
        """Lock or update the face relative position within this person's bounding box."""
        px1, py1, px2, py2 = self.bbox
        pw = px2 - px1
        if pw < 1.0:
            pw = 1.0
        ph = py2 - py1
        if ph < 1.0:
            ph = 1.0
        fx1, fy1, fx2, fy2 = face_bbox
        inv_pw = 1.0 / pw
        inv_ph = 1.0 / ph

        nr0 = (fx1 - px1) * inv_pw
        nr1 = (fy1 - py1) * inv_ph
        nr2 = (fx2 - px1) * inv_pw
        nr3 = (fy2 - py1) * inv_ph

        old_rel = self.face_relative_box
        if old_rel is not None:
            beta = 1.0 - alpha
            self.face_relative_box = (
                beta * old_rel[0] + alpha * nr0,
                beta * old_rel[1] + alpha * nr1,
                beta * old_rel[2] + alpha * nr2,
                beta * old_rel[3] + alpha * nr3,
            )
        else:
            self.face_relative_box = (nr0, nr1, nr2, nr3)

        self.face_confidence = float(confidence)
        if timestamp is not None:
            self.face_last_seen = float(timestamp)
        elif self.face_last_seen is None:
            self.face_last_seen = self.last_seen

        lf = self.locked_face
        rel_lm = []
        if landmarks:
            old_rel_lm = lf.get("relative_landmarks") if lf else None
            if old_rel_lm and len(old_rel_lm) == len(landmarks):
                beta = 1.0 - alpha
                rel_lm = [
                    (
                        beta * old_rel_lm[i][0] + alpha * ((lx - px1) * inv_pw),
                        beta * old_rel_lm[i][1] + alpha * ((ly - py1) * inv_ph),
                    )
                    for i, (lx, ly) in enumerate(landmarks)
                ]
            else:
                rel_lm = [((lx - px1) * inv_pw, (ly - py1) * inv_ph) for lx, ly in landmarks]
            self.face_landmarks = list(landmarks)

        fb = (float(fx1), float(fy1), float(fx2), float(fy2))
        self.face_bbox = fb
        r0, r1, r2, r3 = self.face_relative_box

        if lf is not None:
            b = lf["bbox"]
            b[0] = round(fb[0], 2)
            b[1] = round(fb[1], 2)
            b[2] = round(fb[2], 2)
            b[3] = round(fb[3], 2)
            lf["confidence"] = round(self.face_confidence, 4)
            rb = lf["relative_box"]
            rb[0] = round(r0, 4)
            rb[1] = round(r1, 4)
            rb[2] = round(r2, 4)
            rb[3] = round(r3, 4)
            lf["last_seen"] = self.face_last_seen
            lf["relative_landmarks"] = rel_lm
            if self.face_landmarks:
                lm_out = lf.get("landmarks")
                if isinstance(lm_out, list) and len(lm_out) == len(self.face_landmarks):
                    for i, (lx, ly) in enumerate(self.face_landmarks):
                        pt = lm_out[i]
                        pt[0] = round(lx, 2)
                        pt[1] = round(ly, 2)
                else:
                    lf["landmarks"] = [[round(p[0], 2), round(p[1], 2)] for p in self.face_landmarks]
            else:
                lf["landmarks"] = []
        else:
            self.locked_face = {
                "bbox": [round(fb[0], 2), round(fb[1], 2), round(fb[2], 2), round(fb[3], 2)],
                "confidence": round(self.face_confidence, 4),
                "landmarks": [[round(p[0], 2), round(p[1], 2)] for p in self.face_landmarks] if self.face_landmarks else [],
                "relative_box": [round(r0, 4), round(r1, 4), round(r2, 4), round(r3, 4)],
                "last_seen": self.face_last_seen,
                "is_locked": True,
                "relative_landmarks": rel_lm,
            }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "session_id": self.session_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration": round(self.duration, 2),
            "bbox": [round(float(c), 2) for c in self.bbox],
            "confidence": round(float(self.confidence), 4),
            "hits": self.hits,
            "state": self.state,
            "confirmed": self.confirmed,
            "speed_px_s": round(float(self.speed_px_s), 2),
            "velocity": [round(float(self.velocity[0]), 2), round(float(self.velocity[1]), 2)],
            "locked_face": self.locked_face,
            "face_bbox": [round(float(c), 2) for c in self.face_bbox] if self.face_bbox else None,
            "face_confidence": round(float(self.face_confidence), 4) if self.face_bbox else 0.0,
            "face_landmarks": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in self.face_landmarks] if self.face_landmarks else [],
            "face_relative_box": [round(float(c), 4) for c in self.face_relative_box] if self.face_relative_box else None,
            "face_last_seen": self.face_last_seen,
        }


@dataclass
class FacePose:
    """Estimated facial geometry and 3D head orientation."""
    yaw: float  # Degrees: negative = left, positive = right
    pitch: float  # Degrees: negative = down, positive = up
    roll: float  # Degrees: tilt
    category: str = "frontal"  # "frontal", "left_angle", "right_angle", "profile", "uncertain"
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "yaw": round(float(self.yaw), 1),
            "pitch": round(float(self.pitch), 1),
            "roll": round(float(self.roll), 1),
            "category": self.category,
            "confidence": round(float(self.confidence), 3),
        }


@dataclass
class FaceQualityResult:
    """Intelligent multi-factor face quality assessment score (0-100)."""
    score: float  # 0.0 to 100.0
    sharpness: float  # Laplacian variance or Tenengrad
    motion_blur: float  # 0 to 1, higher = more blur
    resolution_score: float  # 0 to 100 based on face area
    lighting_score: float  # 0 to 100 based on exposure and contrast
    pose_score: float  # 0 to 100 based on frontal/angular usability
    size_score: float  # 0 to 100 based on bounding box dimensions
    stability_score: float = 100.0
    is_acceptable: bool = True
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(float(self.score), 1),
            "sharpness": round(float(self.sharpness), 2),
            "motion_blur": round(float(self.motion_blur), 3),
            "resolution_score": round(float(self.resolution_score), 1),
            "lighting_score": round(float(self.lighting_score), 1),
            "pose_score": round(float(self.pose_score), 1),
            "size_score": round(float(self.size_score), 1),
            "stability_score": round(float(self.stability_score), 1),
            "is_acceptable": self.is_acceptable,
            "metrics": {k: round(float(v), 3) for k, v in self.metrics.items()},
        }


@dataclass
class FaceObservation:
    """Localized facial observation associated with an active person track."""
    observation_id: str
    track_id: int
    timestamp: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in frame coordinates
    confidence: float
    landmarks: List[Tuple[float, float]] = field(default_factory=list)  # (x, y) 5-point coords
    pose: Optional[FacePose] = None
    quality: Optional[FaceQualityResult] = None
    face_crop: Optional[np.ndarray] = None  # Original resolution crop if extracted
    relative_box: Optional[Tuple[float, float, float, float]] = None  # (rx1, ry1, rx2, ry2) normalized to parent body bbox
    frame_sequence: Optional[int] = None
    verification_score: float = 0.0
    verification_method: Optional[str] = None
    velocity: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        v = self.velocity
        if v is None:
            self.velocity = (0.0, 0.0)
        elif not isinstance(v, tuple) or len(v) != 2:
            self.velocity = (float(v[0]), float(v[1])) if len(v) >= 2 else (0.0, 0.0)
        if not isinstance(self.bbox, tuple):
            b = self.bbox
            self.bbox = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        if self.relative_box is not None and not isinstance(self.relative_box, tuple):
            rb = self.relative_box
            self.relative_box = (float(rb[0]), float(rb[1]), float(rb[2]), float(rb[3]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "bbox": [round(float(c), 2) for c in self.bbox],
            "confidence": round(float(self.confidence), 4),
            "landmarks": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in self.landmarks],
            "pose": self.pose.to_dict() if self.pose else None,
            "quality": self.quality.to_dict() if self.quality else None,
            "relative_box": [round(float(c), 4) for c in self.relative_box] if self.relative_box else None,
            "frame_sequence": self.frame_sequence,
            "verification_score": self.verification_score,
            "verification_method": self.verification_method,
            "velocity": [round(float(self.velocity[0]), 2), round(float(self.velocity[1]), 2)],
        }


@dataclass
class CaptureCandidate:
    """Evaluated candidate face frame under consideration for permanent saving."""
    candidate_id: str
    track_id: int
    timestamp: float
    category: str  # "overall", "frontal", "left_angle", "right_angle", "high_res"
    quality_score: float
    pose: FacePose
    crop_original: np.ndarray
    frame_sequence: int
    bbox_in_frame: Tuple[float, float, float, float]
    normalized_bbox: Optional[Tuple[float, float, float, float]] = None
    needs_super_resolution: bool = False
    verification_score: float = 0.0
    verification_method: Optional[str] = None


@dataclass
class CaptureRecord:
    """Persisted facial capture metadata and asset paths."""
    capture_id: str
    track_id: int
    session_id: str
    timestamp: float
    category: str  # "overall", "frontal", "left_angle", "right_angle", "high_res"
    quality_score: float
    pose_category: str
    yaw: float
    pitch: float
    original_path: str
    crop_width: int
    crop_height: int
    enhanced_path: Optional[str] = None
    is_enhanced: bool = False
    enhancement_method: Optional[str] = None
    file_size_bytes: int = 0
    created_at: float = field(default_factory=lambda: 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capture_id": self.capture_id,
            "track_id": self.track_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "category": self.category,
            "quality_score": round(float(self.quality_score), 1),
            "pose_category": self.pose_category,
            "yaw": round(float(self.yaw), 1),
            "pitch": round(float(self.pitch), 1),
            "original_url": f"/api/people/captures/{self.capture_id}/image?type=original",
            "enhanced_url": f"/api/people/captures/{self.capture_id}/image?type=enhanced" if self.is_enhanced else None,
            "is_enhanced": self.is_enhanced,
            "enhancement_method": self.enhancement_method,
            "crop_width": self.crop_width,
            "crop_height": self.crop_height,
            "file_size_bytes": self.file_size_bytes,
        }


@dataclass
class PersonProfile:
    """Anonymous session profile for a tracked individual with best captures."""
    session_id: str
    track_id: int
    camera_id: str
    first_seen: float
    last_seen: float
    duration: float
    status: str = "active"  # "active", "lost", "completed"
    best_capture: Optional[CaptureRecord] = None
    captures_by_category: Dict[str, CaptureRecord] = field(default_factory=dict)
    capture_history: List[CaptureRecord] = field(default_factory=list)
    total_observations: int = 0
    best_quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration": round(float(self.duration), 2),
            "status": self.status,
            "best_capture": self.best_capture.to_dict() if self.best_capture else None,
            "captures_by_category": {
                k: v.to_dict() for k, v in self.captures_by_category.items()
            },
            "capture_history_count": len(self.capture_history),
            "total_observations": self.total_observations,
            "best_quality_score": round(float(self.best_quality_score), 1),
            "metadata": self.metadata,
        }


@dataclass
class AnalysisState:
    """Global operational state of the people analysis engine."""
    is_running: bool = False
    started_at: Optional[float] = None
    stopped_at: Optional[float] = None
    active_tracks_count: int = 0
    total_profiles_count: int = 0
    total_captures_count: int = 0
    pending_enhancements: int = 0
    fps: float = 0.0
    status_message: str = "Idle"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "active_tracks_count": self.active_tracks_count,
            "total_profiles_count": self.total_profiles_count,
            "total_captures_count": self.total_captures_count,
            "pending_enhancements": self.pending_enhancements,
            "fps": round(float(self.fps), 1),
            "status_message": self.status_message,
            "error": self.error,
        }


@dataclass
class ProcessingMetrics:
    """Operational latency, throughput, and hardware resource metrics."""
    frame_sequence: int = 0
    acquisition_fps: float = 0.0
    detection_ms: float = 0.0
    tracking_ms: float = 0.0
    face_detection_ms: float = 0.0
    quality_eval_ms: float = 0.0
    enhancement_ms: float = 0.0
    active_tracks: int = 0
    queue_depth: int = 0
    dropped_frames: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_sequence": self.frame_sequence,
            "acquisition_fps": round(float(self.acquisition_fps), 1),
            "detection_ms": round(float(self.detection_ms), 2),
            "tracking_ms": round(float(self.tracking_ms), 2),
            "face_detection_ms": round(float(self.face_detection_ms), 2),
            "quality_eval_ms": round(float(self.quality_eval_ms), 2),
            "enhancement_ms": round(float(self.enhancement_ms), 2),
            "active_tracks": self.active_tracks,
            "queue_depth": self.queue_depth,
            "dropped_frames": self.dropped_frames,
        }
