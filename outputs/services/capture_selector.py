"""Intelligent face capture selector for multi-person tracking.

Continuously evaluates facial observations per active track to select, update,
and rate-limit capture events across multiple pose and quality categories:
- overall: Highest quality score face frame across all angles
- frontal: Best frontal view (yaw within ±15 degrees)
- left_angle: Best left-facing view (-45 to -15 degrees)
- right_angle: Best right-facing view (+15 to +45 degrees)
- high_res: Highest pixel resolution / face area with acceptable quality

Enforces bounded candidate memory buffers, anti-jitter delta thresholds,
disk write rate limits with quality jump overrides, and preserves captures
for briefly visible individuals.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

try:
    from .face_contracts import (
        CaptureCandidate,
        FaceObservation,
        FacePose,
        FaceQualityResult,
    )
except ImportError:
    from services.face_contracts import (
        CaptureCandidate,
        FaceObservation,
        FacePose,
        FaceQualityResult,
    )


# Recognized capture categories
CATEGORY_OVERALL = "overall"
CATEGORY_FRONTAL = "frontal"
CATEGORY_LEFT_ANGLE = "left_angle"
CATEGORY_RIGHT_ANGLE = "right_angle"
CATEGORY_HIGH_RES = "high_res"

VALID_CATEGORIES = (
    CATEGORY_OVERALL,
    CATEGORY_FRONTAL,
    CATEGORY_LEFT_ANGLE,
    CATEGORY_RIGHT_ANGLE,
    CATEGORY_HIGH_RES,
)

_DEFAULT_PORTRAIT_PROPORTIONS = (0.40, 0.32, 0.34)


@dataclass
class CaptureSelectorConfig:
    """Tunable thresholds and behavioral parameters for capture selection."""
    buffer_size: int = 5
    quality_delta_threshold: float = 4.0
    rate_limit_seconds: float = 0.8
    quality_jump_threshold: float = 15.0
    min_quality_threshold: float = 10.0
    high_res_min_quality: float = 20.0
    high_res_min_pixels: int = 0
    allow_best_available: bool = False
    min_source_pixels: int = 24
    resolution_delta_ratio: float = 0.15  # 15% face area increase
    frontal_yaw_range: Tuple[float, float] = (-15.0, 15.0)
    left_angle_yaw_range: Tuple[float, float] = (-45.0, -15.0)
    right_angle_yaw_range: Tuple[float, float] = (15.0, 45.0)
    portrait_headroom: float = 0.40
    portrait_chin: float = 0.32
    portrait_sides: float = 0.34
    min_capture_dim: int = 256
    auto_enhance_small_faces: bool = False
    require_verified_face: bool = False


@dataclass
class CaptureDecision:
    """Decision output for an evaluated observation in a specific category."""
    track_id: int
    category: str
    candidate: CaptureCandidate
    is_selected: bool
    should_save_disk: bool
    reason: str
    quality_delta: float = 0.0
    previous_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "category": self.category,
            "candidate_id": self.candidate.candidate_id,
            "quality_score": round(float(self.candidate.quality_score), 1),
            "is_selected": self.is_selected,
            "should_save_disk": self.should_save_disk,
            "reason": self.reason,
            "quality_delta": round(float(self.quality_delta), 1),
            "previous_score": (
                round(float(self.previous_score), 1)
                if self.previous_score is not None
                else None
            ),
        }


@dataclass
class TrackCaptureState:
    """Maintains active candidate buffers and category selections for one person track."""
    track_id: int
    candidate_buffer: List[CaptureCandidate] = field(default_factory=list)
    selected_captures: Dict[str, CaptureCandidate] = field(default_factory=dict)
    last_saved_captures: Dict[str, CaptureCandidate] = field(default_factory=dict)
    last_disk_write_time: Optional[float] = None
    last_saved_quality_score: float = 0.0
    absolute_best_candidate: Optional[CaptureCandidate] = None
    observation_count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0


def determine_pose_category(
    pose: Optional[FacePose],
    frontal_range: Tuple[float, float] = (-15.0, 15.0),
    left_range: Tuple[float, float] = (-45.0, -15.0),
    right_range: Tuple[float, float] = (15.0, 45.0),
) -> Optional[str]:
    """Classify 3D head yaw orientation into frontal, left_angle, or right_angle."""
    if pose is None:
        return CATEGORY_FRONTAL

    yaw = float(pose.yaw)

    # Frontal check (inclusive of both boundaries)
    if frontal_range[0] <= yaw <= frontal_range[1]:
        return CATEGORY_FRONTAL

    # Left-angle check: yaw in [-45, -15)
    if left_range[0] <= yaw < left_range[1]:
        return CATEGORY_LEFT_ANGLE

    # Right-angle check: yaw in (15, 45]
    if right_range[0] < yaw <= right_range[1]:
        return CATEGORY_RIGHT_ANGLE

    return None


class CaptureSelector:
    """Decides when to capture, update, and rate-limit facial images for tracked people."""

    def __init__(self, config: Optional[CaptureSelectorConfig] = None, **kwargs: Any) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = CaptureSelectorConfig(**kwargs)

        self._tracks: Dict[int, TrackCaptureState] = {}
        self._lock = threading.RLock()
        self._total_observations = 0
        self._total_disk_events = 0

    @property
    def active_track_ids(self) -> List[int]:
        with self._lock:
            return list(self._tracks.keys())

    def _get_or_create_track(self, track_id: int, timestamp: float) -> TrackCaptureState:
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackCaptureState(
                track_id=track_id,
                first_seen=timestamp,
                last_seen=timestamp,
            )
        state = self._tracks[track_id]
        state.last_seen = timestamp
        state.observation_count += 1
        return state

    def _extract_and_format_candidate(
        self,
        observation: FaceObservation,
        category: str = CATEGORY_OVERALL,
        frame_sequence: int = 0,
        frame_image: Optional[np.ndarray] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        headroom: float = 0.35,
        chin: float = 0.25,
        sides: float = 0.30,
        min_dim: int = 256,
        auto_enhance_small: bool = False,
    ) -> CaptureCandidate:
        """Extract head-and-shoulders portrait crop strictly inside frame bounds.

        Original pixels are never letterboxed, resized, or enhanced before persistence.
        Small or sub-studio-clarity crops are flagged for optional super-resolution.
        """
        crop = None
        if frame_image is not None and frame_image.size > 0:
            x1, y1, x2, y2 = observation.bbox
            bw = max(1.0, float(x2 - x1))
            bh = max(1.0, float(y2 - y1))
            h, w = frame_image.shape[:2]

            # Generous portrait framing: crown/hair headroom, chin/neck/collar,
            # and balanced lateral margins. Strictly clipped to [0, w] x [0, h]
            # so black letterbox/border pixels are never synthesized.
            crop_y1 = max(0, min(h, int(round(y1 - headroom * bh))))
            crop_y2 = max(0, min(h, int(round(y2 + chin * bh))))
            crop_x1 = max(0, min(w, int(round(x1 - sides * bw))))
            crop_x2 = max(0, min(w, int(round(x2 + sides * bw))))

            if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                crop = frame_image[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        if crop is None or crop.size == 0:
            if observation.face_crop is not None and observation.face_crop.size > 0:
                crop = observation.face_crop.copy()
            else:
                crop = np.zeros((1, 1, 3), dtype=np.uint8)

        pose = observation.pose or FacePose(
            yaw=0.0, pitch=0.0, roll=0.0, category="frontal", confidence=1.0
        )
        quality_score = float(observation.quality.score) if observation.quality else 0.0

        # Flag small crops or sub-88 quality portraits for optional super-resolution
        ch, cw = crop.shape[:2]
        needs_sr = bool(
            (ch < min_dim or cw < min_dim or quality_score < 88.0) and (ch > 1 and cw > 1)
        )

        # Originals are always native pixels. Optional enlargement is a separate
        # processed asset after persistence, never a substitute for source detail.
        normalized_bbox = None
        if frame_shape and frame_shape[0] > 0 and frame_shape[1] > 0:
            fh, fw = frame_shape[:2]
            x1, y1, x2, y2 = observation.bbox
            normalized_bbox = (
                max(0.0, min(1.0, float(x1) / fw)),
                max(0.0, min(1.0, float(y1) / fh)),
                max(0.0, min(1.0, float(x2) / fw)),
                max(0.0, min(1.0, float(y2) / fh)),
            )

        cand_id = f"cand_{observation.track_id}_{category}_{uuid.uuid4().hex[:8]}"

        candidate = CaptureCandidate(
            candidate_id=cand_id,
            track_id=observation.track_id,
            timestamp=observation.timestamp,
            category=category,
            quality_score=quality_score,
            pose=pose,
            crop_original=crop,
            frame_sequence=frame_sequence,
            bbox_in_frame=observation.bbox,
            normalized_bbox=normalized_bbox,
            needs_super_resolution=needs_sr,
            verification_score=observation.verification_score,
            verification_method=observation.verification_method,
        )
        return candidate

    def create_candidate(
        self,
        observation: FaceObservation,
        category: str = CATEGORY_OVERALL,
        frame_sequence: int = 0,
        frame_image: Optional[np.ndarray] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        margin: Optional[float] = None,
        headroom: Optional[float] = None,
        chin: Optional[float] = None,
        sides: Optional[float] = None,
        auto_enhance_small: bool = False,
    ) -> CaptureCandidate:
        """Construct a validated CaptureCandidate instance from a FaceObservation."""
        cfg_hr = getattr(self.config, "portrait_headroom", 0.40)
        cfg_ch = getattr(self.config, "portrait_chin", 0.32)
        cfg_sd = getattr(self.config, "portrait_sides", 0.34)
        cfg_min_dim = getattr(self.config, "min_capture_dim", 256)

        if headroom is not None or chin is not None or sides is not None:
            hr = cfg_hr if headroom is None else headroom
            ch = cfg_ch if chin is None else chin
            sd = cfg_sd if sides is None else sides
            return self._extract_and_format_candidate(
                observation=observation,
                category=category,
                frame_sequence=frame_sequence,
                frame_image=frame_image,
                frame_shape=frame_shape,
                headroom=hr,
                chin=ch,
                sides=sd,
                min_dim=cfg_min_dim,
                auto_enhance_small=auto_enhance_small,
            )
        elif margin is not None:
            return self._extract_and_format_candidate(
                observation=observation,
                category=category,
                frame_sequence=frame_sequence,
                frame_image=frame_image,
                frame_shape=frame_shape,
                headroom=margin,
                chin=margin,
                sides=margin,
                min_dim=cfg_min_dim,
                auto_enhance_small=auto_enhance_small,
            )
        else:
            # If caller explicitly configured custom portrait proportions on config,
            # honor them; otherwise preserve symmetric 0.25 direct-helper contract.
            if (cfg_hr, cfg_ch, cfg_sd) != _DEFAULT_PORTRAIT_PROPORTIONS:
                hr, ch, sd = cfg_hr, cfg_ch, cfg_sd
            else:
                hr, ch, sd = 0.25, 0.25, 0.25
            return self._extract_and_format_candidate(
                observation=observation,
                category=category,
                frame_sequence=frame_sequence,
                frame_image=frame_image,
                frame_shape=frame_shape,
                headroom=hr,
                chin=ch,
                sides=sd,
                min_dim=cfg_min_dim,
                auto_enhance_small=auto_enhance_small,
            )

    def _add_to_buffer(self, state: TrackCaptureState, candidate: CaptureCandidate) -> None:
        """Maintain bounded candidate buffer (e.g. keeping up to 5 best recent candidates)."""
        buf = state.candidate_buffer
        if len(buf) < self.config.buffer_size:
            buf.append(candidate)
        else:
            # Bounded capacity reached: find the lowest quality score in buffer
            min_idx = min(range(len(buf)), key=lambda i: buf[i].quality_score)
            if candidate.quality_score > buf[min_idx].quality_score:
                buf[min_idx] = candidate
            elif (
                candidate.quality_score == buf[min_idx].quality_score
                and candidate.timestamp >= buf[min_idx].timestamp
            ):
                # Prefer more recent if identical score
                buf[min_idx] = candidate

        # Sort descending by quality score so buffer[0] is always highest quality
        buf.sort(key=lambda c: c.quality_score, reverse=True)

    def _determine_applicable_categories(
        self,
        pose: FacePose,
        quality_score: float,
        is_acceptable: bool,
    ) -> List[str]:
        """Determine which capture categories this facial observation qualifies for."""
        categories = [CATEGORY_OVERALL]

        pose_cat = determine_pose_category(
            pose,
            frontal_range=self.config.frontal_yaw_range,
            left_range=self.config.left_angle_yaw_range,
            right_range=self.config.right_angle_yaw_range,
        )
        if pose_cat in (CATEGORY_FRONTAL, CATEGORY_LEFT_ANGLE, CATEGORY_RIGHT_ANGLE):
            categories.append(pose_cat)

        # High resolution category eligibility
        if is_acceptable and quality_score >= max(
            self.config.high_res_min_quality, self.config.min_quality_threshold
        ):
            categories.append(CATEGORY_HIGH_RES)

        return categories

    def _calculate_face_area(self, candidate: CaptureCandidate) -> float:
        """Calculate pixel resolution face area from crop or bounding box."""
        if (
            candidate.crop_original is not None
            and candidate.crop_original.ndim >= 2
            and candidate.crop_original.size > 0
        ):
            h, w = candidate.crop_original.shape[:2]
            if h > 1 or w > 1:
                return float(w * h)

        x1, y1, x2, y2 = candidate.bbox_in_frame
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

    def evaluate_observation(
        self,
        observation: FaceObservation,
        frame_sequence: int = 0,
        frame_image: Optional[np.ndarray] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> List[CaptureDecision]:
        """Evaluate a single face observation, updating candidates and rate-limiting captures."""
        with self._lock:
            if self.config.require_verified_face and (
                observation.verification_score < 0.90
                or observation.verification_method != "yunet+res10ssd-v1"
                or observation.frame_sequence != frame_sequence
                or observation.quality is None
                or not bool(observation.quality.is_acceptable)
            ):
                return []

            pose = observation.pose or FacePose(
                yaw=0.0, pitch=0.0, roll=0.0, category="frontal", confidence=1.0
            )
            quality_score = (
                float(observation.quality.score) if observation.quality is not None else 0.0
            )
            is_acceptable = (
                bool(observation.quality.is_acceptable)
                if observation.quality is not None
                else True
            )

            # Unusable faces must never enter any category or the finalization
            # fallback buffer. A high average score cannot override a failed gate.
            best_available = False  # failed quality gates cannot be bypassed
            if not is_acceptable:
                return []

            self._total_observations += 1
            now = float(observation.timestamp)
            state = self._get_or_create_track(observation.track_id, now)

            # Create base candidate with generous head-and-shoulders portrait proportions
            if frame_image is not None:
                base_candidate = self._extract_and_format_candidate(
                    observation,
                    category=CATEGORY_OVERALL,
                    frame_sequence=frame_sequence,
                    frame_image=frame_image,
                    frame_shape=frame_shape,
                    headroom=getattr(self.config, "portrait_headroom", 0.40),
                    chin=getattr(self.config, "portrait_chin", 0.32),
                    sides=getattr(self.config, "portrait_sides", 0.34),
                    min_dim=getattr(self.config, "min_capture_dim", 256),
                    auto_enhance_small=False,
                )
            else:
                base_candidate = self.create_candidate(
                    observation,
                    category=CATEGORY_OVERALL,
                    frame_sequence=frame_sequence,
                    frame_image=frame_image,
                    frame_shape=frame_shape,
                )

            # Update absolute best candidate seen over track lifetime (for brief visits)
            if (
                state.absolute_best_candidate is None
                or quality_score > state.absolute_best_candidate.quality_score
                or (
                    quality_score == state.absolute_best_candidate.quality_score
                    and self._calculate_face_area(base_candidate)
                    > self._calculate_face_area(state.absolute_best_candidate)
                )
            ):
                state.absolute_best_candidate = base_candidate

            # Maintain bounded buffer per track
            self._add_to_buffer(state, base_candidate)

            applicable_categories = self._determine_applicable_categories(
                pose=pose,
                quality_score=quality_score,
                is_acceptable=is_acceptable,
            )
            if not is_acceptable:
                applicable_categories = [CATEGORY_OVERALL]
            if (
                min(
                    observation.bbox[2] - observation.bbox[0],
                    observation.bbox[3] - observation.bbox[1],
                )
                < self.config.high_res_min_pixels
            ):
                applicable_categories = [
                    c for c in applicable_categories if c != CATEGORY_HIGH_RES
                ]

            # Determine disk write eligibility for this observation event
            last_write = state.last_disk_write_time
            last_saved_score = state.last_saved_quality_score
            time_elapsed = now - last_write if last_write is not None else float("inf")
            jump_over_disk = quality_score - last_saved_score

            allow_disk_write = False
            is_jump_override = False

            if last_write is None:
                allow_disk_write = True
            elif time_elapsed >= self.config.rate_limit_seconds:
                allow_disk_write = True
            elif jump_over_disk > self.config.quality_jump_threshold:
                allow_disk_write = True
                is_jump_override = True
            else:
                allow_disk_write = False

            decisions: List[CaptureDecision] = []
            any_disk_saved = False

            for cat in applicable_categories:
                # Clone candidate with the specific category
                candidate_for_cat = replace(
                    base_candidate,
                    category=cat,
                    candidate_id=f"cand_{observation.track_id}_{cat}_{uuid.uuid4().hex[:8]}",
                )

                existing = state.selected_captures.get(cat)
                is_upgrade = False
                reason = "rejected"
                prev_score = existing.quality_score if existing else None
                quality_delta = (
                    quality_score - existing.quality_score
                    if existing
                    else quality_score
                )

                if existing is None:
                    # Initial evaluation for this category
                    min_q = (
                        self.config.high_res_min_quality
                        if cat == CATEGORY_HIGH_RES
                        else self.config.min_quality_threshold
                    )
                    if best_available and cat == CATEGORY_OVERALL:
                        min_q = 0.0
                    if quality_score >= min_q:
                        is_upgrade = True
                        reason = (
                            "best_available_source" if not is_acceptable else "initial_capture"
                        )
                    else:
                        reason = "below_min_quality"
                elif cat == CATEGORY_HIGH_RES:
                    # High resolution category evaluation
                    old_area = self._calculate_face_area(existing)
                    new_area = self._calculate_face_area(candidate_for_cat)
                    res_ratio = (
                        (new_area - old_area) / max(1.0, old_area)
                        if old_area > 0
                        else 0.0
                    )

                    # Upgrade if significant resolution increase with acceptable quality,
                    # or significant quality score jump at similar/larger resolution
                    if (
                        res_ratio >= self.config.resolution_delta_ratio
                        and quality_score >= self.config.high_res_min_quality
                    ):
                        is_upgrade = True
                        reason = "resolution_upgrade"
                    elif (
                        quality_delta >= self.config.quality_delta_threshold
                        and new_area >= old_area * 0.90
                    ):
                        is_upgrade = True
                        reason = "quality_upgrade"
                    else:
                        reason = (
                            "insufficient_resolution_increase"
                            if res_ratio < self.config.resolution_delta_ratio
                            else "insufficient_quality_delta"
                        )
                else:
                    # Smart best-shot upgrade for overall / angular views:
                    # 1. Standard anti-jitter quality delta upgrade
                    # 2. Close-up studio upgrade when face area grows >= 50% AND quality improves
                    old_area = self._calculate_face_area(existing)
                    new_area = self._calculate_face_area(candidate_for_cat)
                    if quality_delta >= self.config.quality_delta_threshold:
                        is_upgrade = True
                        reason = "quality_upgrade"
                    elif (
                        old_area > 0
                        and new_area >= old_area * 1.50
                        and quality_delta >= max(1.5, 0.5 * self.config.quality_delta_threshold)
                        and quality_score >= 65.0
                    ):
                        is_upgrade = True
                        reason = "quality_upgrade"
                    else:
                        reason = "insufficient_quality_delta"

                should_save_disk = False
                if is_upgrade:
                    state.selected_captures[cat] = candidate_for_cat

                    if allow_disk_write:
                        should_save_disk = True
                        if is_jump_override:
                            reason += "_jump_override"
                        state.last_saved_captures[cat] = candidate_for_cat
                        any_disk_saved = True
                    else:
                        should_save_disk = False

                decisions.append(
                    CaptureDecision(
                        track_id=observation.track_id,
                        category=cat,
                        candidate=candidate_for_cat,
                        is_selected=is_upgrade,
                        should_save_disk=should_save_disk,
                        reason=reason,
                        quality_delta=quality_delta,
                        previous_score=prev_score,
                    )
                )

            if any_disk_saved:
                state.last_disk_write_time = now
                state.last_saved_quality_score = max(
                    state.last_saved_quality_score, quality_score
                )
                self._total_disk_events += 1

            return decisions

    def evaluate_candidate(self, candidate: CaptureCandidate) -> CaptureDecision:
        """Evaluate an already-constructed CaptureCandidate directly."""
        obs = FaceObservation(
            observation_id=f"eval_{candidate.candidate_id}",
            track_id=candidate.track_id,
            timestamp=candidate.timestamp,
            bbox=candidate.bbox_in_frame,
            confidence=1.0,
            pose=candidate.pose,
            quality=FaceQualityResult(
                score=candidate.quality_score,
                sharpness=100.0,
                motion_blur=0.0,
                resolution_score=100.0,
                lighting_score=100.0,
                pose_score=100.0,
                size_score=100.0,
                is_acceptable=True,
            ),
            face_crop=candidate.crop_original,
            frame_sequence=candidate.frame_sequence,
            verification_score=candidate.verification_score,
            verification_method=candidate.verification_method,
        )
        decisions = self.evaluate_observation(
            obs,
            frame_sequence=candidate.frame_sequence,
        )
        # Find matching decision or return first
        for d in decisions:
            if d.category == candidate.category:
                return d
        return decisions[0] if decisions else CaptureDecision(
            track_id=candidate.track_id,
            category=candidate.category,
            candidate=candidate,
            is_selected=False,
            should_save_disk=False,
            reason="not_applicable",
        )

    def process_frame(
        self,
        observations: List[FaceObservation],
        frame_sequence: int = 0,
        frame_image: Optional[np.ndarray] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> List[CaptureDecision]:
        """Batch process all observations detected in a video frame."""
        all_decisions: List[CaptureDecision] = []
        for obs in observations:
            decisions = self.evaluate_observation(
                observation=obs,
                frame_sequence=frame_sequence,
                frame_image=frame_image,
                frame_shape=frame_shape,
            )
            all_decisions.extend(decisions)
        return all_decisions

    def get_candidate_buffer(self, track_id: int) -> List[CaptureCandidate]:
        """Return the bounded candidate buffer for a track."""
        with self._lock:
            state = self._tracks.get(track_id)
            if state is None:
                return []
            return list(state.candidate_buffer)

    def get_best_captures(self, track_id: int) -> Dict[str, CaptureCandidate]:
        """Return all category captures selected for this track."""
        with self._lock:
            state = self._tracks.get(track_id)
            if state is None:
                return {}
            return dict(state.selected_captures)

    def get_track_captures(self, track_id: int) -> Dict[str, CaptureCandidate]:
        """Alias for get_best_captures."""
        return self.get_best_captures(track_id)

    def get_best_capture(
        self, track_id: int, category: str = CATEGORY_OVERALL
    ) -> Optional[CaptureCandidate]:
        """Return the best capture for a specific category."""
        with self._lock:
            state = self._tracks.get(track_id)
            if state is None:
                return None
            return state.selected_captures.get(category)

    def get_absolute_best(self, track_id: int) -> Optional[CaptureCandidate]:
        """Return the single highest quality candidate ever observed for a track."""
        with self._lock:
            state = self._tracks.get(track_id)
            if state is None:
                return None
            return state.absolute_best_candidate

    def finalize_track(
        self, track_id: int, fallback_to_any: bool = True
    ) -> Dict[str, CaptureCandidate]:
        """Finalize a track that has completed or been lost.

        Guarantees that people appearing only briefly retain the highest quality
        facial image observed before track ended.
        """
        with self._lock:
            state = self._tracks.get(track_id)
            if state is None:
                return {}

            results = dict(state.selected_captures)

            # If no category met strict thresholds, fall back to highest quality candidate seen
            if not results and fallback_to_any and state.absolute_best_candidate is not None:
                fallback_cand = replace(
                    state.absolute_best_candidate,
                    category=CATEGORY_OVERALL,
                )
                results[CATEGORY_OVERALL] = fallback_cand
                state.selected_captures[CATEGORY_OVERALL] = fallback_cand

            return results

    def finalize_track_decisions(
        self, track_id: int, fallback_to_any: bool = True
    ) -> List[CaptureDecision]:
        """Finalize track and return capture decisions for any unwritten best candidates."""
        with self._lock:
            state = self._tracks.get(track_id)
            if state is None:
                return []

            final_captures = self.finalize_track(track_id, fallback_to_any=fallback_to_any)
            decisions: List[CaptureDecision] = []

            for cat, cand in final_captures.items():
                last_saved = state.last_saved_captures.get(cat)
                needs_disk_save = (
                    last_saved is None
                    or cand.quality_score > last_saved.quality_score
                )
                if needs_disk_save:
                    state.last_saved_captures[cat] = cand
                    state.last_saved_quality_score = max(
                        state.last_saved_quality_score, cand.quality_score
                    )
                    self._total_disk_events += 1

                decisions.append(
                    CaptureDecision(
                        track_id=track_id,
                        category=cat,
                        candidate=cand,
                        is_selected=True,
                        should_save_disk=needs_disk_save,
                        reason="track_finalized",
                        quality_delta=(
                            cand.quality_score - last_saved.quality_score
                            if last_saved
                            else cand.quality_score
                        ),
                        previous_score=last_saved.quality_score if last_saved else None,
                    )
                )

            return decisions

    def clear_track(self, track_id: int) -> None:
        """Remove all tracked state and candidates for the given track_id."""
        with self._lock:
            self._tracks.pop(track_id, None)

    def prune_stale_tracks(
        self, max_age_seconds: float = 30.0, current_time: Optional[float] = None
    ) -> int:
        """Prune inactive tracks to prevent memory leaks in continuous streaming."""
        with self._lock:
            now = (
                current_time
                if current_time is not None
                else (max((s.last_seen for s in self._tracks.values()), default=0.0))
            )
            stale_ids = [
                tid
                for tid, state in self._tracks.items()
                if (now - state.last_seen) > max_age_seconds
            ]
            for tid in stale_ids:
                del self._tracks[tid]
            return len(stale_ids)

    def reset(self, *args: Any, **kwargs: Any) -> None:
        """Reset all internal state, clearing tracks and metrics."""
        with self._lock:
            self._tracks.clear()
            self._total_observations = 0
            self._total_disk_events = 0

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics and operational statistics."""
        with self._lock:
            return {
                "active_tracks_count": len(self._tracks),
                "total_observations": self._total_observations,
                "total_disk_events": self._total_disk_events,
                "tracks": {
                    tid: {
                        "buffered_candidates": len(s.candidate_buffer),
                        "selected_categories": list(s.selected_captures.keys()),
                        "last_saved_quality": s.last_saved_quality_score,
                        "observation_count": s.observation_count,
                    }
                    for tid, s in self._tracks.items()
                },
            }
