"""Intelligent multi-factor face image quality scoring system (0-100).

Assesses candidate face crops across sharpness, motion blur, source resolution,
lighting & exposure, 3D pose usability, and landmark confidence.
Returns FaceQualityResult instances conforming to services.face_contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

try:
    from services.face_contracts import FaceQualityResult, FacePose, FaceObservation
except ImportError:
    try:
        from outputs.services.face_contracts import FaceQualityResult, FacePose, FaceObservation
    except ImportError:
        from .face_contracts import FaceQualityResult, FacePose, FaceObservation


@dataclass
class QualityThresholds:
    """Configurable pass/fail thresholds for face capture acceptance."""
    min_score: float = 35.0
    min_sharpness: float = 5.0
    min_width: int = 32
    min_height: int = 32
    max_motion_blur: float = 0.85
    min_lighting_score: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_score": self.min_score,
            "min_sharpness": self.min_sharpness,
            "min_width": self.min_width,
            "min_height": self.min_height,
            "max_motion_blur": self.max_motion_blur,
            "min_lighting_score": self.min_lighting_score,
        }


@dataclass
class QualityWeights:
    """Relative factor weighting for composite score calculation."""
    clarity_weight: float = 0.32
    resolution_weight: float = 0.26
    lighting_weight: float = 0.22
    pose_weight: float = 0.15
    stability_weight: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clarity_weight": self.clarity_weight,
            "resolution_weight": self.resolution_weight,
            "lighting_weight": self.lighting_weight,
            "pose_weight": self.pose_weight,
            "stability_weight": self.stability_weight,
        }


def get_quality_tier(score: float) -> str:
    """Map calibrated composite quality score (0-100) to human-readable tier.

    Tiers:
      0-35:   Poor / Unusable
      35-60:  Fair / Acceptable minimum
      60-80:  Good
      80-100: Excellent high-clarity capture
    """
    if score < 35.0:
        return "poor"
    elif score < 60.0:
        return "fair"
    elif score < 80.0:
        return "good"
    else:
        return "excellent"


class FaceQualityAssessor:
    """Comprehensive multi-factor face quality assessment engine."""

    def __init__(
        self,
        thresholds: Optional[QualityThresholds] = None,
        weights: Optional[QualityWeights] = None,
    ):
        self.thresholds = thresholds or QualityThresholds()
        self.weights = weights or QualityWeights()

    def assess(
        self,
        crop: Optional[np.ndarray],
        pose: Optional[Union[FacePose, Dict[str, Any]]] = None,
        landmarks: Optional[List[Tuple[float, float]]] = None,
        landmark_confidence: Optional[float] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> FaceQualityResult:
        """Score candidate face crop using multi-factor metrics."""
        # 1. Null / empty crop guard
        if (
            crop is None
            or not isinstance(crop, np.ndarray)
            or crop.size == 0
            or crop.ndim < 2
            or crop.shape[0] == 0
            or crop.shape[1] == 0
        ):
            return FaceQualityResult(
                score=0.0,
                sharpness=0.0,
                motion_blur=1.0,
                resolution_score=0.0,
                lighting_score=0.0,
                pose_score=0.0,
                size_score=0.0,
                stability_score=0.0,
                is_acceptable=False,
                metrics={
                    "empty_crop": 1.0,
                    "laplacian_var": 0.0,
                    "tenengrad": 0.0,
                    "tier_code": 0.0,
                },
            )

        h, w = crop.shape[:2]

        # Convert crop to grayscale uint8 if needed
        gray = self._to_gray_uint8(crop)
        # Bound spectral work on close-up UHD faces. Resolution scoring and
        # source-pixel gates still use the untouched native dimensions above.
        scale_ratio = 1.0
        if max(gray.shape) > 256:
            scale_ratio = 256.0 / max(gray.shape)
            gray = cv2.resize(
                gray,
                (max(1, round(w * scale_ratio)), max(1, round(h * scale_ratio))),
                interpolation=cv2.INTER_AREA,
            )

        local_landmarks = self._to_crop_local_landmarks(landmarks, w, h, bbox)
        scaled_landmarks = (
            [(lx * scale_ratio, ly * scale_ratio) for lx, ly in local_landmarks]
            if local_landmarks
            else None
        )

        # 2. Sharpness & focus (core facial zone blended with anti-clutter full crop)
        lap_var, tenengrad, sharp_score = self.compute_sharpness(gray, landmarks=scaled_landmarks)

        # 3. Motion blur estimation (Sobel directional anisotropy + HF/LF energy ratio)
        motion_blur, r_dir, r_hf = self.compute_motion_blur(gray, landmarks=scaled_landmarks)

        # Effective clarity score combines focus and motion blur
        clarity_factor = float(np.clip((1.0 - motion_blur) ** 1.85, 0.04, 1.0))
        clarity_score = float(np.clip(sharp_score * clarity_factor, 0.0, 100.0))

        # 4. Real source resolution / face size
        res_score, size_score = self.compute_resolution_and_size(w, h)

        # 5. Lighting & exposure (inner-face clipped highlights & crushed shadows aware)
        lighting_score, mean_lum, contrast_std, exp_factor = self.compute_lighting(
            gray, landmarks=scaled_landmarks
        )

        # 6. Pose usability
        yaw, pitch, roll, pose_cat, pose_conf = self._parse_or_estimate_pose(pose, landmarks)
        pose_score = self.compute_pose_score(yaw, pitch, roll, pose_conf)

        # 7. Landmark detection confidence & stability
        stability_score = self.compute_stability_score(landmarks, landmark_confidence, w, h)

        # 8. Composite score calculation with critical factor gating
        w_cfg = self.weights
        base_score = (
            w_cfg.clarity_weight * clarity_score
            + w_cfg.resolution_weight * res_score
            + w_cfg.lighting_weight * lighting_score
            + w_cfg.pose_weight * pose_score
            + w_cfg.stability_weight * stability_score
        )

        gate_light = min(1.0, (lighting_score / 35.0) ** 1.2) if lighting_score < 35.0 else 1.0
        gate_res = min(1.0, (res_score / 70.0) ** 0.75) if res_score < 70.0 else 1.0
        gate_clarity = min(1.0, (clarity_score / 30.0) ** 1.3) if clarity_score < 30.0 else 1.0
        gate_blur = min(1.0, ((1.0 - motion_blur) / 0.45) ** 1.35) if motion_blur > 0.55 else 1.0
        gate_pose = min(1.0, (pose_score / 68.0) ** 0.5) if pose_score < 68.0 else 1.0

        raw_composite = base_score * gate_light * gate_res * gate_clarity * gate_blur * gate_pose
        # Calibrate top-end soft knee so crisp in-focus faces score cleanly in 70-98 range
        if raw_composite > 93.0:
            raw_composite = 93.0 + 5.0 * (1.0 - np.exp(-(raw_composite - 93.0) / 5.0))
        composite_score = float(np.clip(raw_composite, 0.0, 98.0))

        # 9. Boolean acceptability check
        is_acc = bool(
            composite_score >= self.thresholds.min_score
            and lap_var >= self.thresholds.min_sharpness
            and w >= self.thresholds.min_width
            and h >= self.thresholds.min_height
            and motion_blur <= self.thresholds.max_motion_blur
            and lighting_score >= self.thresholds.min_lighting_score
        )

        tier = get_quality_tier(composite_score)
        tier_codes = {"poor": 0.0, "fair": 1.0, "good": 2.0, "excellent": 3.0}

        core_pixels = self._extract_core_pixels(gray, scaled_landmarks)
        clip_hi_pct = float(np.mean(core_pixels > 248)) if core_pixels.size > 0 else 0.0
        crush_lo_pct = float(np.mean(core_pixels < 10)) if core_pixels.size > 0 else 0.0

        metrics: Dict[str, float] = {
            "laplacian_var": round(float(lap_var), 3),
            "tenengrad": round(float(tenengrad), 3),
            "sharpness_score": round(float(sharp_score), 2),
            "clarity_score": round(float(clarity_score), 2),
            "directional_ratio": round(float(r_dir), 3),
            "spectral_hf_ratio": round(float(r_hf), 4),
            "mean_luminance": round(float(mean_lum), 2),
            "contrast_std": round(float(contrast_std), 2),
            "exposure_factor": round(float(exp_factor), 3),
            "clipped_highlight_ratio": round(clip_hi_pct, 4),
            "crushed_shadow_ratio": round(crush_lo_pct, 4),
            "face_width": float(w),
            "face_height": float(h),
            "face_area": float(w * h),
            "yaw": round(float(yaw), 2),
            "pitch": round(float(pitch), 2),
            "roll": round(float(roll), 2),
            "tier_code": tier_codes.get(tier, 0.0),
        }

        if bbox is not None:
            metrics["bbox_w"] = float(abs(bbox[2] - bbox[0]))
            metrics["bbox_h"] = float(abs(bbox[3] - bbox[1]))

        return FaceQualityResult(
            score=round(composite_score, 1),
            sharpness=round(float(lap_var), 2),
            motion_blur=round(float(motion_blur), 3),
            resolution_score=round(float(res_score), 1),
            lighting_score=round(float(lighting_score), 1),
            pose_score=round(float(pose_score), 1),
            size_score=round(float(size_score), 1),
            stability_score=round(float(stability_score), 1),
            is_acceptable=is_acc,
            metrics=metrics,
        )

    def assess_crop(
        self,
        crop: Optional[np.ndarray],
        pose: Optional[Union[FacePose, Dict[str, Any]]] = None,
        landmarks: Optional[List[Tuple[float, float]]] = None,
        landmark_confidence: Optional[float] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> FaceQualityResult:
        """Alias for assess()."""
        return self.assess(
            crop=crop,
            pose=pose,
            landmarks=landmarks,
            landmark_confidence=landmark_confidence,
            bbox=bbox,
        )

    def assess_observation(self, observation: FaceObservation) -> FaceQualityResult:
        """Score a FaceObservation instance and attach the computed quality result."""
        result = self.assess(
            crop=observation.face_crop,
            pose=observation.pose,
            landmarks=observation.landmarks,
            landmark_confidence=observation.confidence,
            bbox=observation.bbox,
        )
        observation.quality = result
        return result

    def _to_crop_local_landmarks(
        self,
        landmarks: Optional[List[Tuple[float, float]]],
        width: int,
        height: int,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[List[Tuple[float, float]]]:
        """Map 5-point landmarks into crop-local coordinates when valid."""
        if not landmarks or len(landmarks) < 5 or width <= 0 or height <= 0:
            return None

        pts = [(float(p[0]), float(p[1])) for p in landmarks[:5]]
        in_crop = all(
            -0.08 * width <= x <= 1.08 * width and -0.08 * height <= y <= 1.08 * height
            for x, y in pts
        )
        if in_crop:
            return pts

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            bw = max(1.0, float(x2 - x1))
            bh = max(1.0, float(y2 - y1))
            shifted = [
                ((x - float(x1)) * (width / bw), (y - float(y1)) * (height / bh))
                for x, y in pts
            ]
            if all(
                -0.08 * width <= x <= 1.08 * width and -0.08 * height <= y <= 1.08 * height
                for x, y in shifted
            ):
                return shifted

        return None

    def _build_core_mask(
        self,
        height: int,
        width: int,
        landmarks: Optional[List[Tuple[float, float]]] = None,
    ) -> Optional[np.ndarray]:
        """Build boolean mask over the central facial core (landmark zone or inscribed ellipse).

        Excludes outer background corners behind ears/hair AND the outer face-silhouette
        step edge so background textures cannot inflate sharpness of a blurry face.
        """
        if height < 12 or width < 12:
            return None

        if landmarks and len(landmarks) >= 5:
            le, re, nose, lm, rm = landmarks[:5]
            if le[0] < re[0] and min(le[1], re[1]) < max(lm[1], rm[1]):
                xs = [p[0] for p in landmarks[:5]]
                ys = [p[1] for p in landmarks[:5]]
                span_x = max(xs) - min(xs)
                span_y = max(ys) - min(ys)
                if span_x >= 0.15 * width and span_y >= 0.15 * height:
                    x1 = max(1, int(np.floor(min(xs) - 0.25 * span_x)))
                    x2 = min(width - 1, int(np.ceil(max(xs) + 0.25 * span_x)))
                    y1 = max(1, int(np.floor(min(ys) - 0.30 * span_y)))
                    y2 = min(height - 1, int(np.ceil(max(ys) + 0.24 * span_y)))
                    if (x2 - x1) >= max(6, int(0.30 * width)) and (y2 - y1) >= max(6, int(0.30 * height)):
                        mask = np.zeros((height, width), dtype=bool)
                        mask[y1:y2, x1:x2] = True
                        return mask

        # Inscribed central core facial ellipse inside inner ~68%x70% zone:
        # Covers eyebrows, eyes, nose, cheeks, and lips while excluding outer corners & rim
        cy = (height - 1) * 0.5
        cx = (width - 1) * 0.5
        ry = max(3.0, height * 0.34)
        rx = max(3.0, width * 0.33)
        yy, xx = np.ogrid[:height, :width]
        ellipse = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
        if int(np.count_nonzero(ellipse)) >= 16:
            return ellipse
        return None

    def _extract_core_pixels(
        self,
        gray: np.ndarray,
        landmarks: Optional[List[Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """Return 1D array of grayscale values inside the central core facial zone."""
        mask = self._build_core_mask(gray.shape[0], gray.shape[1], landmarks)
        if mask is None:
            return gray.ravel()
        return gray[mask]

    def _extract_core_region(
        self,
        gray: np.ndarray,
        landmarks: Optional[List[Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """Extract central 2D core facial sub-box for spectral/FFT analysis."""
        gh, gw = gray.shape[:2]
        if gh < 12 or gw < 12:
            return gray

        if landmarks and len(landmarks) >= 5:
            le, re, nose, lm, rm = landmarks[:5]
            if le[0] < re[0] and min(le[1], re[1]) < max(lm[1], rm[1]):
                xs = [p[0] for p in landmarks[:5]]
                ys = [p[1] for p in landmarks[:5]]
                span_x = max(xs) - min(xs)
                span_y = max(ys) - min(ys)
                if span_x >= 0.15 * gw and span_y >= 0.15 * gh:
                    x1 = max(0, int(np.floor(min(xs) - 0.25 * span_x)))
                    x2 = min(gw, int(np.ceil(max(xs) + 0.25 * span_x)))
                    y1 = max(0, int(np.floor(min(ys) - 0.30 * span_y)))
                    y2 = min(gh, int(np.ceil(max(ys) + 0.24 * span_y)))
                    if (x2 - x1) >= 8 and (y2 - y1) >= 8:
                        return gray[y1:y2, x1:x2]

        # Inner central box inscribed inside typical oval face so corners exclude background
        mx = max(1, int(round(gw * 0.22)))
        my = max(1, int(round(gh * 0.20)))
        if (gh - 2 * my) >= 8 and (gw - 2 * mx) >= 8:
            return gray[my : gh - my, mx : gw - mx]
        return gray

    def compute_sharpness(
        self,
        gray: np.ndarray,
        landmarks: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[float, float, float]:
        """Calculate focus and facial micro-contrast using core-focused Laplacian & Tenengrad.

        Focuses evaluation on the central core facial region (inscribed core ellipse
        or eye-nose-mouth landmark zone) blended with capped full-crop gradients so
        high-frequency background textures behind ears/hair cannot inflate a blurry face.

        Returns:
            (laplacian_variance, tenengrad_magnitude, calibrated_sharpness_score)
        """
        if gray.size < 16 or gray.shape[0] < 4 or gray.shape[1] < 4:
            return 0.0, 0.0, 0.0

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_sq = gx ** 2 + gy ** 2

        full_lap = float(np.var(laplacian))
        full_ten = float(np.mean(grad_sq))
        if not np.isfinite(full_lap):
            full_lap = 0.0
        if not np.isfinite(full_ten):
            full_ten = 0.0

        core_mask = self._build_core_mask(gray.shape[0], gray.shape[1], landmarks)
        if core_mask is None:
            lap_var, tenengrad = full_lap, full_ten
        else:
            core_lap = float(np.var(laplacian[core_mask]))
            core_ten = float(np.mean(grad_sq[core_mask]))
            if not np.isfinite(core_lap):
                core_lap = 0.0
            if not np.isfinite(core_ten):
                core_ten = 0.0
            # Cap full-crop contribution relative to core facial sharpness so noisy/textured
            # backgrounds behind ears/hair cannot inflate a blurry face's sharpness score
            eff_full_lap = min(full_lap, max(core_lap * 1.30, core_lap + 3.0))
            eff_full_ten = min(full_ten, max(core_ten * 1.30, core_ten + 40.0))
            lap_var = 0.78 * core_lap + 0.22 * eff_full_lap
            tenengrad = 0.78 * core_ten + 0.22 * eff_full_ten

        # Calibrated score mapping: 20 is ~37, 50 is ~68, 100+ is >90 (crisp faces 70-98)
        lap_sc = 100.0 * (1.0 - np.exp(-lap_var / 44.0))
        t_sc = 100.0 * (1.0 - np.exp(-tenengrad / 1150.0))
        sharp_score = float(np.clip(0.65 * lap_sc + 0.35 * t_sc, 0.0, 100.0))

        return float(lap_var), float(tenengrad), float(sharp_score)

    @staticmethod
    def _directional_anisotropy(patch: np.ndarray) -> Tuple[float, float]:
        """Compute Sobel gradient directional isotropy and weakest-axis HF/LF ratio.

        Evaluates L2 variance across (0, 45, 90, 135 deg) plus |Gx| vs |Gy| and
        diagonal |G45| vs |G135| L1 imbalance, and measures fine-vs-coarse gradient
        energy along the weakest directional axis to catch linear motion smear.
        """
        gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        g45 = (gx + gy) * 0.7071067811865475
        g135 = (gx - gy) * 0.7071067811865475

        grads = (gx, gy, g45, g135)
        vars_dir = [float(np.var(g)) for g in grads]
        min_idx = int(np.argmin(vars_dir))
        min_v, max_v = vars_dir[min_idx], max(vars_dir)
        if max_v <= 1e-5:
            return 1.0, 0.0

        r_var = float(min_v / (max_v + 1e-5))

        ax, ay = float(np.mean(np.abs(gx))), float(np.mean(np.abs(gy)))
        a45, a135 = float(np.mean(np.abs(g45))), float(np.mean(np.abs(g135)))
        r_xy = min(ax, ay) / (max(ax, ay) + 1e-5) if max(ax, ay) > 1e-5 else 1.0
        r_diag = min(a45, a135) / (max(a45, a135) + 1e-5) if max(a45, a135) > 1e-5 else 1.0
        r_l1_sq = min(r_xy, r_diag) ** 2

        r_dir = float(min(r_var, 0.75 * r_var + 0.25 * r_l1_sq))

        # Along the weakest gradient axis, measure fine vs coarse gradient energy
        coarse_patch = cv2.GaussianBlur(patch.astype(np.float64), (7, 7), 1.8)
        cgx = cv2.Sobel(coarse_patch, cv2.CV_64F, 1, 0, ksize=3)
        cgy = cv2.Sobel(coarse_patch, cv2.CV_64F, 0, 1, ksize=3)
        cg45 = (cgx + cgy) * 0.7071067811865475
        cg135 = (cgx - cgy) * 0.7071067811865475
        c_grads = (cgx, cgy, cg45, cg135)

        fine_weak = float(np.mean(np.abs(grads[min_idx])))
        coarse_weak = float(np.mean(np.abs(c_grads[min_idx])))
        weak_hf_ratio = fine_weak / (coarse_weak + 1e-5) if coarse_weak > 1e-4 else 0.0

        return r_dir, float(weak_hf_ratio)

    def compute_motion_blur(
        self,
        gray: np.ndarray,
        landmarks: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[float, float, float]:
        """Calculate motion blur (0.0 to 1.0) using Sobel directional anisotropy and HF/LF energy.

        Returns:
            (motion_blur_score, directional_ratio, spectral_hf_ratio)
        """
        h, w = gray.shape[:2]
        if h < 8 or w < 8:
            return 1.0, 0.0, 0.0

        # 1. Directional Sobel anisotropy (|Gx| vs |Gy| + diagonal imbalance)
        r_dir_full, weak_hf_full = self._directional_anisotropy(gray)
        core = self._extract_core_region(gray, landmarks)
        if core is not gray and core.shape[0] >= 12 and core.shape[1] >= 12:
            r_dir_core, weak_hf_core = self._directional_anisotropy(core)
            # Do not let isotropic background clutter hide directional smear on the face
            r_dir = float(min(r_dir_full, 0.75 * r_dir_core + 0.25 * r_dir_full))
            weak_hf = float(min(weak_hf_full, 0.75 * weak_hf_core + 0.25 * weak_hf_full))
        else:
            r_dir = r_dir_full
            weak_hf = weak_hf_full

        b_dir = float(np.clip((0.40 - r_dir) / 0.30, 0.0, 1.0))

        # 2. 2D FFT spectral high-frequency to low-frequency energy ratio
        spec_roi = core if (core.shape[0] >= 16 and core.shape[1] >= 16) else gray
        sh, sw = spec_roi.shape[:2]
        f = np.fft.fft2(spec_roi)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        cy, cx = sh // 2, sw // 2
        y, x = np.ogrid[:sh, :sw]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_r = max(1.0, min(sh, sw) / 2.0)
        hf_mask = dist > (0.30 * max_r)
        lf_mask = dist <= max(1.0, 0.12 * max_r)

        hf_power = float(np.mean(mag[hf_mask])) if np.any(hf_mask) else 0.0
        lf_power = float(np.mean(mag[lf_mask])) if np.any(lf_mask) else 1e-5
        r_hf = float(hf_power / (lf_power + 1e-5))
        b_spec = float(np.clip((0.022 - r_hf) / 0.018, 0.0, 1.0))

        # Weakest-axis fine-to-coarse gradient ratio (collapses under linear smear)
        b_axis_hf = float(np.clip((1.35 - weak_hf) / 0.55, 0.0, 1.0))
        b_freq = 0.80 * b_spec + 0.20 * b_axis_hf

        motion_blur = float(np.clip(0.55 * b_dir + 0.45 * b_freq, 0.0, 1.0))
        return float(motion_blur), float(r_dir), float(r_hf)

    def compute_resolution_and_size(self, width: int, height: int) -> Tuple[float, float]:
        """Calculate resolution and size usability scores (0-100).

        Returns:
            (resolution_score, size_score)
        """
        if width <= 0 or height <= 0:
            return 0.0, 0.0

        area = width * height
        min_dim = min(width, height)
        aspect_ratio = max(width, height) / max(1, min_dim)
        aspect_penalty = max(0.2, 1.0 - max(0.0, aspect_ratio - 1.6) / 2.0)

        eq_dim = np.sqrt(float(area))
        if eq_dim < 24.0:
            res_score = max(0.0, (eq_dim / 24.0) ** 1.5 * 15.0)
        elif eq_dim < 40.0:
            res_score = 15.0 + (eq_dim - 24.0) / 16.0 * 20.0
        elif eq_dim < 80.0:
            res_score = 35.0 + (eq_dim - 40.0) / 40.0 * 30.0
        elif eq_dim < 120.0:
            res_score = 65.0 + (eq_dim - 80.0) / 40.0 * 25.0
        else:
            res_score = min(100.0, 90.0 + (eq_dim - 120.0) / 40.0 * 10.0)

        if min_dim < 24.0:
            size_base = max(0.0, (min_dim / 24.0) ** 1.5 * 15.0)
        elif min_dim < 40.0:
            size_base = 15.0 + (min_dim - 24.0) / 16.0 * 20.0
        elif min_dim < 80.0:
            size_base = 35.0 + (min_dim - 40.0) / 40.0 * 30.0
        elif min_dim < 120.0:
            size_base = 65.0 + (min_dim - 80.0) / 40.0 * 25.0
        else:
            size_base = min(100.0, 90.0 + (min_dim - 120.0) / 40.0 * 10.0)

        size_score = float(np.clip(size_base * aspect_penalty, 0.0, 100.0))
        return float(res_score), float(size_score)

    def compute_lighting(
        self,
        gray: np.ndarray,
        landmarks: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[float, float, float, float]:
        """Calculate lighting & exposure quality on the inner face and overall crop.

        Penalizes severe clipped highlights (> 248) or crushed shadow regions (< 10)
        on the inner face while rewarding clean, balanced dynamic range.

        Returns:
            (lighting_score, mean_luminance, contrast_std, exposure_factor)
        """
        if gray.size == 0:
            return 0.0, 0.0, 0.0, 0.0

        core_pixels = self._extract_core_pixels(gray, landmarks)
        full_mean = float(np.mean(gray))
        core_mean = float(np.mean(core_pixels)) if core_pixels.size > 0 else full_mean
        mean_lum = 0.78 * core_mean + 0.22 * full_mean
        std_contrast = float(np.std(core_pixels)) * 0.70 + float(np.std(gray)) * 0.30

        # Pitch-black or pure washed-out white check
        if full_mean <= 0.5 or full_mean >= 254.5 or std_contrast < 1e-4:
            if full_mean <= 1.0 or full_mean >= 254.0:
                return 0.0, float(mean_lum), float(std_contrast), 0.0

        # Exposure factor for underexposed (< 40) or overexposed (> 220)
        if mean_lum < 40.0:
            exp_factor = max(0.0, (mean_lum / 40.0) ** 1.8 * 0.40)
        elif mean_lum > 220.0:
            exp_factor = max(0.0, ((255.0 - mean_lum) / 35.0) ** 1.8 * 0.40)
        elif 40.0 <= mean_lum < 75.0:
            exp_factor = 0.40 + 0.60 * ((mean_lum - 40.0) / 35.0)
        elif 175.0 < mean_lum <= 220.0:
            exp_factor = 0.40 + 0.60 * ((220.0 - mean_lum) / 45.0)
        else:
            exp_factor = 1.0

        # Inner-face highlight clipping (> 248) and shadow crushing (< 10) penalties
        eval_patch = core_pixels if core_pixels.size > 0 else gray.ravel()
        clip_hi_pct = float(np.mean(eval_patch > 248))
        crush_lo_pct = float(np.mean(eval_patch < 10))

        clip_penalty = 1.0
        if clip_hi_pct > 0.12:
            # Penalize severe blown-out facial highlights while allowing natural eye catchlights
            clip_penalty *= max(0.10, 1.0 - min(1.0, (clip_hi_pct - 0.12) / 0.50) ** 1.15 * 0.90)
        if crush_lo_pct > 0.18:
            # Penalize crushed backlit silhouette faces
            clip_penalty *= max(0.10, 1.0 - min(1.0, (crush_lo_pct - 0.18) / 0.55) ** 1.15 * 0.90)

        exp_factor = float(exp_factor * clip_penalty)

        # Contrast & clean dynamic range evaluation
        p5, p95 = np.percentile(eval_patch, [5, 95])
        dyn_range = float(p95 - p5)

        if std_contrast < 8.0:
            cont_factor = max(0.0, (std_contrast / 8.0) * 0.25)
        elif std_contrast < 22.0:
            cont_factor = 0.25 + 0.55 * ((std_contrast - 8.0) / 14.0)
        elif std_contrast <= 85.0:
            cont_factor = 0.80 + 0.20 * min(1.0, (std_contrast - 22.0) / 20.0)
        else:
            extreme_pct = float(np.mean((eval_patch <= 5) | (eval_patch > 248)))
            cont_factor = max(0.35, 1.0 - extreme_pct * 0.80)

        # Reward clean tonal dynamic range when free of severe clipping/crushing
        if dyn_range >= 45.0 and clip_hi_pct <= 0.08 and crush_lo_pct <= 0.10:
            cont_factor = min(1.03, cont_factor * 1.02)

        lighting_score = float(np.clip(exp_factor * cont_factor * 100.0, 0.0, 100.0))
        return float(lighting_score), float(mean_lum), float(std_contrast), float(exp_factor)

    def compute_pose_score(
        self,
        yaw: float,
        pitch: float,
        roll: float,
        confidence: float = 1.0,
    ) -> float:
        """Reward frontal (1.0) and moderate angle (0.85), penalize extreme profile or down-pitch."""
        ay = abs(yaw)
        if ay <= 15.0:
            yaw_f = 1.0
        elif ay <= 35.0:
            yaw_f = 1.0 - (ay - 15.0) / 20.0 * 0.15
        elif ay <= 55.0:
            yaw_f = 0.85 - (ay - 35.0) / 20.0 * 0.35
        else:
            yaw_f = max(0.25, 0.50 - (ay - 55.0) / 35.0 * 0.25)

        if -12.0 <= pitch <= 15.0:
            pitch_f = 1.0
        elif pitch < -12.0:
            pitch_f = max(0.30, 1.0 - ((-pitch) - 12.0) / 38.0 * 0.65)
        else:
            pitch_f = max(0.40, 1.0 - (pitch - 15.0) / 35.0 * 0.55)

        ar = abs(roll)
        roll_f = 1.0 if ar <= 15.0 else max(0.50, 1.0 - (ar - 15.0) / 45.0 * 0.40)

        conf = max(0.2, min(1.0, float(confidence)))
        return float(np.clip(yaw_f * pitch_f * roll_f * 100.0 * conf, 0.0, 100.0))

    def estimate_pose_from_landmarks(
        self,
        landmarks: List[Tuple[float, float]],
    ) -> Tuple[float, float, float, str]:
        """Estimate yaw, pitch, roll and category from 5-point facial landmarks."""
        if len(landmarks) < 5:
            return 0.0, 0.0, 0.0, "uncertain"

        le, re, nose, lm, rm = [np.array(p, dtype=float) for p in landmarks[:5]]
        eye_mid = (le + re) / 2.0
        eye_dist = float(np.linalg.norm(re - le))
        if eye_dist < 1e-4:
            return 0.0, 0.0, 0.0, "uncertain"

        # Yaw approximation from nose shift relative to eye midpoint
        dx_nose = nose[0] - eye_mid[0]
        yaw_ratio = dx_nose / (eye_dist / 2.0)
        yaw_deg = float(np.clip(yaw_ratio * 45.0, -90.0, 90.0))

        # Roll from ocular slant angle
        d_eye = re - le
        roll_deg = float(np.degrees(np.arctan2(d_eye[1], d_eye[0])))

        # Pitch from vertical nose position between eyes and mouth
        mouth_mid = (lm + rm) / 2.0
        total_vert = mouth_mid[1] - eye_mid[1]
        if total_vert > 1e-4:
            nose_vert = nose[1] - eye_mid[1]
            vert_ratio = nose_vert / total_vert
            pitch_deg = float(np.clip((vert_ratio - 0.52) * -90.0, -90.0, 90.0))
        else:
            pitch_deg = 0.0

        ay = abs(yaw_deg)
        if ay <= 15.0:
            category = "frontal"
        elif ay > 45.0:
            category = "profile"
        elif yaw_deg < 0:
            category = "left_angle"
        else:
            category = "right_angle"

        return float(yaw_deg), float(pitch_deg), float(roll_deg), category

    def compute_stability_score(
        self,
        landmarks: Optional[List[Tuple[float, float]]] = None,
        landmark_confidence: Optional[float] = None,
        width: int = 0,
        height: int = 0,
    ) -> float:
        """Evaluate landmark detection confidence and geometric consistency."""
        conf = 1.0 if landmark_confidence is None else float(np.clip(landmark_confidence, 0.0, 1.0))
        if not landmarks or len(landmarks) < 5:
            return float(np.clip(conf * 100.0, 0.0, 100.0))

        geom_factor = 1.0
        le, re, nose, lm, rm = landmarks[:5]

        # 1. Left eye should be to left of right eye
        if le[0] >= re[0]:
            geom_factor *= 0.6

        # 2. Eyes should be above mouth
        if le[1] >= lm[1] or re[1] >= rm[1]:
            geom_factor *= 0.5

        # 3. Nose should be between eyes and mouth vertically
        if nose[1] <= min(le[1], re[1]) or nose[1] >= max(lm[1], rm[1]):
            geom_factor *= 0.7

        # 4. Landmarks within bounds
        if width > 0 and height > 0:
            oob = sum(
                1 for x, y in landmarks[:5] if x < -5 or x > width + 5 or y < -5 or y > height + 5
            )
            if oob > 0:
                geom_factor *= max(0.3, 1.0 - 0.2 * oob)

        return float(np.clip(conf * geom_factor * 100.0, 0.0, 100.0))

    def _to_gray_uint8(self, crop: np.ndarray) -> np.ndarray:
        """Safely convert any numpy crop array to 2D uint8 grayscale."""
        if crop.ndim == 2:
            gray = crop
        elif crop.ndim == 3:
            channels = crop.shape[2]
            if channels == 4:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)
            elif channels == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            elif channels == 1:
                gray = crop[:, :, 0]
            else:
                gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.reshape(crop.shape[0], crop.shape[1])

        if gray.dtype != np.uint8:
            if np.issubdtype(gray.dtype, np.floating) and np.max(gray) <= 1.0:
                gray = (np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                gray = np.clip(gray, 0, 255).astype(np.uint8)

        return gray

    def _parse_or_estimate_pose(
        self,
        pose: Optional[Union[FacePose, Dict[str, Any]]] = None,
        landmarks: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[float, float, float, str, float]:
        """Extract or estimate pose angles, category and confidence."""
        if pose is not None:
            if isinstance(pose, FacePose):
                return (
                    float(pose.yaw),
                    float(pose.pitch),
                    float(pose.roll),
                    str(pose.category),
                    float(pose.confidence),
                )
            elif isinstance(pose, dict):
                return (
                    float(pose.get("yaw", 0.0)),
                    float(pose.get("pitch", 0.0)),
                    float(pose.get("roll", 0.0)),
                    str(pose.get("category", "frontal")),
                    float(pose.get("confidence", 1.0)),
                )

        if landmarks and len(landmarks) >= 5:
            yaw, pitch, roll, category = self.estimate_pose_from_landmarks(landmarks)
            return yaw, pitch, roll, category, 0.95

        return 0.0, 0.0, 0.0, "frontal", 1.0


def assess_face_quality(
    crop: Optional[np.ndarray],
    pose: Optional[Union[FacePose, Dict[str, Any]]] = None,
    landmarks: Optional[List[Tuple[float, float]]] = None,
    landmark_confidence: Optional[float] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    thresholds: Optional[QualityThresholds] = None,
    weights: Optional[QualityWeights] = None,
) -> FaceQualityResult:
    """Convenience function to evaluate face quality with default or custom configuration."""
    assessor = FaceQualityAssessor(thresholds=thresholds, weights=weights)
    return assessor.assess(
        crop=crop,
        pose=pose,
        landmarks=landmarks,
        landmark_confidence=landmark_confidence,
        bbox=bbox,
    )
