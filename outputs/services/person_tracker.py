"""Multi-object person tracker maintaining stable anonymous tracking IDs.

This module provides high-accuracy, ultra-low-latency cross-frame association of person
detections using Kalman filtering, Distance-IoU (DIoU) spatial proximity, motion
velocity consistency, and ByteTrack two-stage association.

Conforms strictly to `services.face_contracts.PersonTrack` and `services.face_contracts.PersonDetection`.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from services.face_contracts import PersonDetection, PersonTrack


def compute_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def compute_diou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Compute Distance-IoU (DIoU) between two bounding boxes (x1, y1, x2, y2).

    DIoU = IoU - (center_distance^2 / enclosing_diagonal^2).
    Returns values in range [-1.0, 1.0], where 1.0 represents perfect match.
    """
    iou = compute_iou(box1, box2)

    cx1 = (box1[0] + box1[2]) * 0.5
    cy1 = (box1[1] + box1[3]) * 0.5
    cx2 = (box2[0] + box2[2]) * 0.5
    cy2 = (box2[1] + box2[3]) * 0.5

    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    enc_x1 = min(box1[0], box2[0])
    enc_y1 = min(box1[1], box2[1])
    enc_x2 = max(box1[2], box2[2])
    enc_y2 = max(box1[3], box2[3])

    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-6
    return float(iou - rho2 / c2)


def _vectorized_diou(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
) -> np.ndarray:
    """Vectorized Distance-IoU calculation between (M, 4) and (N, 4) bounding box arrays."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.empty((len(boxes1), len(boxes2)), dtype=np.float64)

    b1 = np.asarray(boxes1, dtype=np.float64)
    b2 = np.asarray(boxes2, dtype=np.float64)

    # Intersection coordinates
    inter_x1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
    inter_y1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
    inter_x2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
    inter_y2 = np.minimum(b1[:, None, 3], b2[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Areas and IoU
    area1 = np.maximum(0.0, b1[:, 2] - b1[:, 0]) * np.maximum(0.0, b1[:, 3] - b1[:, 1])
    area2 = np.maximum(0.0, b2[:, 2] - b2[:, 0]) * np.maximum(0.0, b2[:, 3] - b2[:, 1])
    union = area1[:, None] + area2[None, :] - inter_area

    iou = np.zeros_like(union)
    valid = union > 0.0
    iou[valid] = inter_area[valid] / union[valid]

    # Center coordinates
    cx1 = (b1[:, None, 0] + b1[:, None, 2]) * 0.5
    cy1 = (b1[:, None, 1] + b1[:, None, 3]) * 0.5
    cx2 = (b2[None, :, 0] + b2[None, :, 2]) * 0.5
    cy2 = (b2[None, :, 1] + b2[None, :, 3]) * 0.5

    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Smallest enclosing box
    enc_x1 = np.minimum(b1[:, None, 0], b2[None, :, 0])
    enc_y1 = np.minimum(b1[:, None, 1], b2[None, :, 1])
    enc_x2 = np.maximum(b1[:, None, 2], b2[None, :, 2])
    enc_y2 = np.maximum(b1[:, None, 3], b2[None, :, 3])
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-6

    return iou - (rho2 / c2)


class KalmanBoxTracker:
    """Hybrid Kalman / Alpha-Beta kinematic filter for 2D bounding boxes in image space.

    Tracks 4 independent 1D kinematic filters for [cx, cy, w, h] with velocity [vx, vy, vw, vh],
    dt clamping [0.008, 1.0], bounded predictive displacement, and smooth velocity damping on
    lost frames. Executes in microseconds per track without NumPy array allocation overhead.
    """

    __slots__ = (
        "cx", "cy", "w", "h",
        "vx", "vy", "vw", "vh",
        "p00_0", "p00_1", "p00_2", "p00_3",
        "p01_0", "p01_1", "p01_2", "p01_3",
        "p11_0", "p11_1", "p11_2", "p11_3",
        "last_timestamp", "last_bbox", "_last_dt", "_obs_count",
    )

    def __init__(self, bbox: Tuple[float, float, float, float], timestamp: float):
        x1, y1, x2, y2 = bbox
        self.cx = (x1 + x2) * 0.5
        self.cy = (y1 + y2) * 0.5
        self.w = max(1.0, float(x2 - x1))
        self.h = max(1.0, float(y2 - y1))

        self.vx = 0.0
        self.vy = 0.0
        self.vw = 0.0
        self.vh = 0.0

        pos_var = (0.05 * self.h) ** 2
        vel_var = (0.85 * self.h) ** 2

        self.p00_0 = pos_var
        self.p00_1 = pos_var
        self.p00_2 = pos_var
        self.p00_3 = pos_var

        self.p01_0 = 0.0
        self.p01_1 = 0.0
        self.p01_2 = 0.0
        self.p01_3 = 0.0

        self.p11_0 = vel_var
        self.p11_1 = vel_var
        self.p11_2 = vel_var * 0.15
        self.p11_3 = vel_var * 0.15

        self.last_timestamp = float(timestamp)
        self.last_bbox = bbox
        self._last_dt = 0.0333
        self._obs_count = 1

    @property
    def p(self) -> np.ndarray:
        return np.array([self.cx, self.cy, self.w, self.h], dtype=np.float64)

    @property
    def v(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vw, self.vh], dtype=np.float64)

    def predict(self, timestamp: float, is_lost: bool = False) -> Tuple[float, float, float, float]:
        """Advance state vector by clamped dt in [0.008, 1.0] and return predicted bounding box."""
        raw_dt = float(timestamp) - self.last_timestamp
        dt = 0.008 if raw_dt < 0.008 else (1.0 if raw_dt > 1.0 else raw_dt)
        step_dt = 0.0 if raw_dt <= 0.0 else dt
        self._last_dt = dt

        if is_lost and step_dt > 0.0:
            damping = 0.90 ** (step_dt / 0.1)
            scale_damping = 0.65 ** (step_dt / 0.1)
            self.vx *= damping
            self.vy *= damping
            self.vw *= scale_damping
            self.vh *= scale_damping

        max_disp = max(4.5 * math.hypot(self.w, self.h), 350.0)
        dx = self.vx * step_dt
        if dx > max_disp:
            dx = max_disp
        elif dx < -max_disp:
            dx = -max_disp

        dy = self.vy * step_dt
        if dy > max_disp:
            dy = max_disp
        elif dy < -max_disp:
            dy = -max_disp

        max_dw = 0.35 * self.w
        dw = self.vw * step_dt
        if dw > max_dw:
            dw = max_dw
        elif dw < -max_dw:
            dw = -max_dw

        max_dh = 0.35 * self.h
        dh = self.vh * step_dt
        if dh > max_dh:
            dh = max_dh
        elif dh < -max_dh:
            dh = -max_dh

        self.cx += dx
        self.cy += dy
        self.w = max(1.0, self.w + dw)
        self.h = max(1.0, self.h + dh)

        h = self.h if self.h > 1.0 else 1.0
        std_p = 0.05 * h
        std_v = 0.85 * h
        dt2 = dt * dt
        var_p = (std_p * std_p) * dt
        var_v = (std_v * std_v) * dt
        two_dt = 2.0 * dt

        self.p00_0 += two_dt * self.p01_0 + dt2 * self.p11_0 + var_p
        self.p01_0 += dt * self.p11_0
        self.p11_0 += var_v

        self.p00_1 += two_dt * self.p01_1 + dt2 * self.p11_1 + var_p
        self.p01_1 += dt * self.p11_1
        self.p11_1 += var_v

        self.p00_2 += two_dt * self.p01_2 + dt2 * self.p11_2 + var_p
        self.p01_2 += dt * self.p11_2
        self.p11_2 += var_v * 0.2

        self.p00_3 += two_dt * self.p01_3 + dt2 * self.p11_3 + var_p
        self.p01_3 += dt * self.p11_3
        self.p11_3 += var_v * 0.2

        self.last_timestamp = float(timestamp)
        return self.get_bbox()

    def update(self, bbox: Tuple[float, float, float, float]):
        """Hybrid Kalman / Alpha-Beta measurement update with observed bounding box."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))

        y0 = cx - self.cx
        y1_ = cy - self.cy
        y2_ = w - self.w
        y3 = h - self.h

        dt = self._last_dt if self._last_dt >= 0.008 else 0.008
        inv_dt = 1.0 / dt

        std_m = 0.05 * h
        r0 = std_m * std_m
        r2 = (std_m * 1.5) ** 2

        s0 = self.p00_0 + r0 + 1e-6
        s1 = self.p00_1 + r0 + 1e-6
        s2 = self.p00_2 + r2 + 1e-6
        s3 = self.p00_3 + r2 + 1e-6

        raw_k0_0 = self.p00_0 / s0
        raw_k1_0 = self.p01_0 / s0
        raw_k0_1 = self.p00_1 / s1
        raw_k1_1 = self.p01_1 / s1
        raw_k0_2 = self.p00_2 / s2
        raw_k1_2 = self.p01_2 / s2
        raw_k0_3 = self.p00_3 / s3
        raw_k1_3 = self.p01_3 / s3

        # Critically-damped Alpha-Beta gain bounds ensure instant velocity lock on hit #2
        # while suppressing detector bbox jitter across steady frames.
        if self._obs_count == 1:
            k0_0 = k0_1 = 0.88
            beta_0 = beta_1 = 0.65
            k0_2 = k0_3 = 0.72
            beta_2 = beta_3 = 0.28
        else:
            large_jump = (abs(y0) > w) or (abs(y1_) > h)
            min_alpha = 0.90 if large_jump else 0.78
            k0_0 = min(0.94, max(min_alpha, raw_k0_0))
            k0_1 = min(0.94, max(min_alpha, raw_k0_1))
            beta_0 = min(0.55, max(0.35, raw_k1_0 * dt))
            beta_1 = min(0.55, max(0.35, raw_k1_1 * dt))
            k0_2 = min(0.82, max(0.58, raw_k0_2))
            k0_3 = min(0.82, max(0.58, raw_k0_3))
            beta_2 = min(0.32, max(0.15, raw_k1_2 * dt))
            beta_3 = min(0.32, max(0.15, raw_k1_3 * dt))

        self.cx += k0_0 * y0
        self.vx = max(-4000.0, min(4000.0, self.vx + beta_0 * inv_dt * y0))
        self.cy += k0_1 * y1_
        self.vy = max(-4000.0, min(4000.0, self.vy + beta_1 * inv_dt * y1_))
        self.w = max(1.0, self.w + k0_2 * y2_)
        self.vw = max(-1500.0, min(1500.0, self.vw + beta_2 * inv_dt * y2_))
        self.h = max(1.0, self.h + k0_3 * y3)
        self.vh = max(-1500.0, min(1500.0, self.vh + beta_3 * inv_dt * y3))

        self.p11_0 = max(1e-3, self.p11_0 - raw_k1_0 * self.p01_0)
        self.p11_1 = max(1e-3, self.p11_1 - raw_k1_1 * self.p01_1)
        self.p11_2 = max(1e-3, self.p11_2 - raw_k1_2 * self.p01_2)
        self.p11_3 = max(1e-3, self.p11_3 - raw_k1_3 * self.p01_3)

        self.p00_0 *= (1.0 - raw_k0_0)
        self.p01_0 *= (1.0 - raw_k0_0)
        self.p00_1 *= (1.0 - raw_k0_1)
        self.p01_1 *= (1.0 - raw_k0_1)
        self.p00_2 *= (1.0 - raw_k0_2)
        self.p01_2 *= (1.0 - raw_k0_2)
        self.p00_3 *= (1.0 - raw_k0_3)
        self.p01_3 *= (1.0 - raw_k0_3)

        self._obs_count += 1
        self.last_bbox = bbox

    def get_bbox(self) -> Tuple[float, float, float, float]:
        """Convert state center and dimensions back to (x1, y1, x2, y2)."""
        w = max(1.0, self.w)
        h = max(1.0, self.h)
        hw = w * 0.5
        hh = h * 0.5
        return (float(self.cx - hw), float(self.cy - hh), float(self.cx + hw), float(self.cy + hh))

    @property
    def velocity(self) -> Tuple[float, float]:
        """Instantaneous velocity (vx, vy) in pixels/sec."""
        return (float(self.vx), float(self.vy))

    @property
    def speed(self) -> float:
        """Scalar speed in pixels/sec."""
        return float(math.hypot(self.vx, self.vy))


class _InternalTrack:
    """Internal tracker container combining contract PersonTrack with Kalman state."""

    def __init__(
        self,
        track_id: int,
        session_id: str,
        bbox: Tuple[float, float, float, float],
        confidence: float,
        timestamp: float,
        confirmed: bool = False,
    ):
        cx = (bbox[0] + bbox[2]) * 0.5
        cy = (bbox[1] + bbox[3]) * 0.5

        self.track = PersonTrack(
            track_id=track_id,
            session_id=session_id,
            first_seen=float(timestamp),
            last_seen=float(timestamp),
            bbox=bbox,
            confidence=float(confidence),
            hits=1,
            state="active",
            confirmed=confirmed,
            speed_px_s=0.0,
            trajectory=[(round(float(timestamp), 4), round(float(cx), 2), round(float(cy), 2))],
            velocity=(0.0, 0.0),
        )

        self.kalman = KalmanBoxTracker(bbox, timestamp)
        self.predicted_bbox: Tuple[float, float, float, float] = bbox
        self.misses: int = 0


def _build_cost_matrix(
    tracks: Sequence[_InternalTrack],
    detections: Sequence[Tuple[Tuple[float, float, float, float], float]],
    max_area_ratio: float = 0.0,
) -> np.ndarray:
    """Build DIoU + scale-similarity + center-distance + motion-consistency cost matrix."""
    if len(tracks) == 0 or len(detections) == 0:
        return np.empty((len(tracks), len(detections)), dtype=np.float64)

    track_boxes = np.array([trk.predicted_bbox for trk in tracks], dtype=np.float64)
    det_boxes = np.array([d[0] for d in detections], dtype=np.float64)

    tw = np.maximum(1.0, track_boxes[:, 2] - track_boxes[:, 0])[:, None]
    th = np.maximum(1.0, track_boxes[:, 3] - track_boxes[:, 1])[:, None]
    dw = np.maximum(1.0, det_boxes[:, 2] - det_boxes[:, 0])[None, :]
    dh = np.maximum(1.0, det_boxes[:, 3] - det_boxes[:, 1])[None, :]

    # Vectorized IoU & DIoU in one pass
    inter_x1 = np.maximum(track_boxes[:, None, 0], det_boxes[None, :, 0])
    inter_y1 = np.maximum(track_boxes[:, None, 1], det_boxes[None, :, 1])
    inter_x2 = np.minimum(track_boxes[:, None, 2], det_boxes[None, :, 2])
    inter_y2 = np.minimum(track_boxes[:, None, 3], det_boxes[None, :, 3])
    inter_area = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)

    ta = tw * th
    da = dw * dh
    union = ta + da - inter_area
    iou = np.where(union > 0.0, inter_area / np.maximum(union, 1e-6), 0.0)

    pred_cx = (track_boxes[:, 0] + track_boxes[:, 2])[:, None] * 0.5
    pred_cy = (track_boxes[:, 1] + track_boxes[:, 3])[:, None] * 0.5
    det_cx = (det_boxes[:, 0] + det_boxes[:, 2])[None, :] * 0.5
    det_cy = (det_boxes[:, 1] + det_boxes[:, 3])[None, :] * 0.5

    rho2 = (pred_cx - det_cx) ** 2 + (pred_cy - det_cy) ** 2
    enc_x1 = np.minimum(track_boxes[:, None, 0], det_boxes[None, :, 0])
    enc_y1 = np.minimum(track_boxes[:, None, 1], det_boxes[None, :, 1])
    enc_x2 = np.maximum(track_boxes[:, None, 2], det_boxes[None, :, 2])
    enc_y2 = np.maximum(track_boxes[:, None, 3], det_boxes[None, :, 3])
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-6

    diou = iou - (rho2 / c2)
    cost_matrix = 1.0 - diou

    # Scale similarity cost + hard area-ratio gate to prevent crossing scale swaps
    scale_log_diff = 0.5 * (np.abs(np.log(tw / dw)) + np.abs(np.log(th / dh)))
    cost_matrix += 0.18 * scale_log_diff

    area_ratio = np.maximum(ta / da, da / ta)
    eff_area_gate = max_area_ratio if max_area_ratio > 0.0 else 6.25
    cost_matrix[area_ratio > eff_area_gate] = 1e6

    # Center-distance tie-breaker + hard spatial gate when non-overlapping
    center_dist = np.sqrt(rho2)
    avg_diag = 0.5 * (np.hypot(tw, th) + np.hypot(dw, dh))
    norm_dist = center_dist / np.maximum(avg_diag, 1.0)
    cost_matrix += 0.06 * np.minimum(norm_dist, 3.0)
    cost_matrix[(iou < 0.01) & (norm_dist > 2.6)] = 1e6

    # Motion-direction consistency penalty aligned with fused optical-flow / kinematic step
    track_last_boxes = np.array([trk.track.bbox for trk in tracks], dtype=np.float64)
    last_cx = (track_last_boxes[:, 0] + track_last_boxes[:, 2])[:, None] * 0.5
    last_cy = (track_last_boxes[:, 1] + track_last_boxes[:, 3])[:, None] * 0.5

    pred_dx = pred_cx - last_cx
    pred_dy = pred_cy - last_cy
    pred_step = np.hypot(pred_dx, pred_dy)

    if np.any(pred_step > 1.5):
        dx = det_cx - last_cx
        dy = det_cy - last_cy
        disp = np.hypot(dx, dy)
        denom = np.maximum(pred_step * disp, 1e-6)
        dot = (pred_dx * dx + pred_dy * dy) / denom
        opposing = (pred_step > 1.5) & (disp > 5.0) & (dot < 0.0)
        cost_matrix += np.where(opposing, 0.7 * (1.0 - dot), 0.0)

    return cost_matrix


def _linear_assignment(
    cost_matrix: np.ndarray,
    cost_threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Solve optimal bipartite matching via Hungarian algorithm."""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched = []
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= cost_threshold:
            matched.append((int(r), int(c)))
            unmatched_rows.discard(int(r))
            unmatched_cols.discard(int(c))

    return matched, sorted(unmatched_rows), sorted(unmatched_cols)


class PersonTracker:
    """Multi-object person tracker maintaining stable anonymous tracking IDs.

    Features:
    - ByteTrack two-stage association (high-score then low-score detections).
    - Decoupled Kalman filter motion prediction and velocity estimation.
    - DIoU and velocity consistency matching to avoid ID switches during trajectory crossing.
    - Track recovery for brief occlusions up to `lifetime` seconds.
    - Full lifecycle management (active -> lost -> ended).
    - Produces and updates `services.face_contracts.PersonTrack` instances.
    - Clean epoch resets for camera and analysis sessions.
    - Strictly anonymous: no cross-session biometric re-identification.
    """

    def __init__(
        self,
        session_id: str = "session_0",
        lifetime: float = 2.5,
        confirm_hits: int = 2,
        high_score_threshold: float = 0.5,
        low_score_threshold: float = 0.1,
        match_threshold: float = 1.35,
        match_threshold_low: float = 1.35,
        max_trajectory_length: int = 100,
        unconfirmed_lifetime: Optional[float] = None,
        max_area_ratio: float = 0.0,
    ):
        self.session_id = str(session_id)
        self.lifetime = float(lifetime)
        self.confirm_hits = int(confirm_hits)
        self.high_score_threshold = float(high_score_threshold)
        self.low_score_threshold = float(low_score_threshold)
        self.match_threshold = float(match_threshold)
        self.match_threshold_low = float(match_threshold_low)
        self.max_trajectory_length = int(max_trajectory_length)
        self.unconfirmed_lifetime = float(unconfirmed_lifetime) if unconfirmed_lifetime is not None else float(lifetime)
        self.max_area_ratio = float(max_area_ratio)

        self._active_tracks: Dict[int, _InternalTrack] = {}
        self._ended_tracks: Dict[int, PersonTrack] = {}
        self._next_id: int = 1
        self.last_timestamp: Optional[float] = None
        self.frame_count: int = 0

    def reset(self, session_id: Optional[str] = None):
        """Reset tracking state cleanly when camera or analysis restarts."""
        if session_id is not None:
            self.session_id = str(session_id)
        self._active_tracks.clear()
        self._ended_tracks.clear()
        self._next_id = 1
        self.last_timestamp = None
        self.frame_count = 0

    def reset_epoch(self, epoch: str):
        """Convenience method to reset tracker with a new session epoch."""
        self.reset(session_id=epoch)

    @property
    def active_tracks(self) -> List[PersonTrack]:
        """All tracks observed and confirmed active in the latest frame."""
        return [t.track for t in self._active_tracks.values() if t.track.state == "active"]

    @property
    def lost_tracks(self) -> List[PersonTrack]:
        """Tracks temporarily unobserved but within the occlusion recovery lifetime."""
        return [t.track for t in self._active_tracks.values() if t.track.state == "lost"]

    @property
    def ended_tracks(self) -> List[PersonTrack]:
        """Retired tracks whose absence exceeded the occlusion recovery lifetime."""
        return list(self._ended_tracks.values())

    @property
    def confirmed_tracks(self) -> List[PersonTrack]:
        """Active tracks that have met the confirmation threshold."""
        return [t.track for t in self._active_tracks.values() if t.track.state == "active" and t.track.confirmed]

    @property
    def all_tracks(self) -> List[PersonTrack]:
        """All session tracks including active, lost, and ended."""
        return [t.track for t in self._active_tracks.values()] + list(self._ended_tracks.values())

    def get_track(self, track_id: int) -> Optional[PersonTrack]:
        """Fetch a track by its integer track ID."""
        if track_id in self._active_tracks:
            return self._active_tracks[track_id].track
        if track_id in self._ended_tracks:
            return self._ended_tracks[track_id]
        return None

    def _normalize_detections(
        self,
        detections: Any,
    ) -> List[Tuple[Tuple[float, float, float, float], float]]:
        """Normalize various input detection representations into [((x1, y1, x2, y2), score), ...]."""
        normalized: List[Tuple[Tuple[float, float, float, float], float]] = []

        if detections is None:
            return normalized

        # Handle Ultralytics Boxes object
        if hasattr(detections, "data") and not isinstance(detections, np.ndarray):
            data = detections.data
            if hasattr(data, "cpu"):
                data = data.cpu().numpy()
            elif hasattr(data, "numpy"):
                data = data.numpy()
            detections = data

        # If NumPy ndarray
        if isinstance(detections, np.ndarray):
            if detections.size == 0:
                return normalized
            arr = detections.reshape(1, -1) if detections.ndim == 1 else detections
            for row in arr:
                if len(row) >= 4:
                    box = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
                    conf = float(row[4]) if len(row) > 4 else 1.0
                    if box[2] > box[0] and box[3] > box[1]:
                        normalized.append((box, conf))
            return normalized

        # If list / sequence of objects
        for item in detections:
            if isinstance(item, PersonDetection):
                box = (float(item.bbox[0]), float(item.bbox[1]), float(item.bbox[2]), float(item.bbox[3]))
                if box[2] > box[0] and box[3] > box[1]:
                    normalized.append((box, float(item.confidence)))
            elif isinstance(item, dict):
                raw_box = item.get("bbox", item.get("box"))
                if raw_box and len(raw_box) == 4:
                    box = (float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3]))
                    conf = float(item.get("confidence", item.get("score", 1.0)))
                    if box[2] > box[0] and box[3] > box[1]:
                        normalized.append((box, conf))
            elif isinstance(item, (list, tuple)):
                if len(item) >= 4:
                    box = (float(item[0]), float(item[1]), float(item[2]), float(item[3]))
                    conf = float(item[4]) if len(item) > 4 else 1.0
                    if box[2] > box[0] and box[3] > box[1]:
                        normalized.append((box, conf))

        return normalized

    def lock_face(
        self,
        track_id: int,
        face_bbox: Tuple[float, float, float, float],
        confidence: float = 1.0,
        landmarks: Optional[List[Tuple[float, float]]] = None,
        timestamp: Optional[float] = None,
        alpha: float = 0.3,
    ) -> bool:
        """Lock or update a face observation on a specific person track."""
        if track_id in self._active_tracks:
            trk = self._active_tracks[track_id]
            trk.track.lock_face(
                face_bbox=face_bbox,
                confidence=confidence,
                landmarks=landmarks,
                timestamp=timestamp if timestamp is not None else self.last_timestamp,
                alpha=alpha,
            )
            return True
        return False

    def associate_faces(
        self,
        face_observations: Any,
        timestamp: Optional[float] = None,
        alpha: float = 0.3,
    ) -> Dict[int, Dict[str, Any]]:
        """Associate face detections with the best-matching active person tracks and update locked faces.

        Returns:
            Dict mapping track_id to locked_face dictionary.
        """
        if not face_observations or not self._active_tracks:
            return {}

        ts = float(timestamp) if timestamp is not None else (self.last_timestamp or time.time())
        norm_faces = []
        for f in face_observations:
            if hasattr(f, "bbox"):
                fb = (float(f.bbox[0]), float(f.bbox[1]), float(f.bbox[2]), float(f.bbox[3]))
                fc = float(getattr(f, "confidence", 1.0))
                fl = getattr(f, "landmarks", None)
                norm_faces.append((fb, fc, fl))
            elif isinstance(f, dict):
                raw = f.get("bbox", f.get("box"))
                if raw and len(raw) == 4:
                    fb = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
                    fc = float(f.get("confidence", 1.0))
                    fl = f.get("landmarks")
                    norm_faces.append((fb, fc, fl))
            elif isinstance(f, (list, tuple)) and len(f) >= 4:
                fb = (float(f[0]), float(f[1]), float(f[2]), float(f[3]))
                fc = float(f[4]) if len(f) > 4 else 1.0
                fl = f[5] if len(f) > 5 and isinstance(f[5], list) else None
                norm_faces.append((fb, fc, fl))

        if not norm_faces:
            return {}

        results = {}
        tracks = list(self._active_tracks.values())

        # For each face, find candidate person track based on head containment
        for fb, fc, fl in norm_faces:
            fcx = (fb[0] + fb[2]) * 0.5
            fcy = (fb[1] + fb[3]) * 0.5
            fa = max(1.0, (fb[2] - fb[0]) * (fb[3] - fb[1]))

            best_trk = None
            best_score = -1e9

            for trk in tracks:
                pb = trk.track.bbox
                pw = max(1.0, float(pb[2] - pb[0]))
                ph = max(1.0, float(pb[3] - pb[1]))

                # Check if face center is in the upper body / head region
                in_x = (pb[0] - 0.20 * pw) <= fcx <= (pb[2] + 0.20 * pw)
                in_y = (pb[1] - 0.25 * ph) <= fcy <= (pb[1] + 0.65 * ph)

                if not (in_x and in_y):
                    continue

                # Intersection between face and person body
                ix1 = max(fb[0], pb[0])
                iy1 = max(fb[1], pb[1])
                ix2 = min(fb[2], pb[2])
                iy2 = min(fb[3], pb[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                containment = inter / fa

                # Distance to expected head center: center x of person, top 15% y of person
                head_cx = (pb[0] + pb[2]) * 0.5
                head_cy = pb[1] + 0.15 * ph
                dist = math.hypot((fcx - head_cx) / pw, (fcy - head_cy) / ph)

                score = containment * 2.0 - dist
                if score > best_score:
                    best_score = score
                    best_trk = trk

            if best_trk is not None and best_score > -0.5:
                best_trk.track.lock_face(
                    face_bbox=fb,
                    confidence=fc,
                    landmarks=fl,
                    timestamp=ts,
                    alpha=alpha,
                )
                results[best_trk.track.track_id] = best_trk.track.locked_face

        return results

    def update(
        self,
        detections: Any,
        timestamp: Optional[float] = None,
        source_fps: Optional[float] = None,
        return_lost: bool = False,
        predicted_boxes: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
        face_observations: Optional[Any] = None,
    ) -> List[PersonTrack]:
        """Process detections for the current frame and update all track states.

        Args:
            detections: List of PersonDetection, bounding boxes, or detector rows.
            timestamp: Frame acquisition timestamp in seconds. If None, derived from
                       source_fps or monotonic clock.
            source_fps: Optional camera frame rate used for cadence estimation.
            return_lost: If True, returns both active and lost tracks. Default False
                         (returns only actively observed tracks for this frame).
            predicted_boxes: Optional dictionary of pre-computed bounding boxes.
            face_observations: Optional face detections to lock/update on person tracks.

        Returns:
            List of PersonTrack instances matching current frame observations.
        """
        # Determine frame timestamp
        if timestamp is None:
            if self.last_timestamp is not None and source_fps and source_fps > 0:
                timestamp = self.last_timestamp + 1.0 / source_fps
            else:
                timestamp = time.time()
        else:
            timestamp = float(timestamp)

        self.frame_count += 1
        self.last_timestamp = timestamp

        norm_dets = self._normalize_detections(detections)

        # 1. Predict new locations for all currently maintained tracks
        track_list = list(self._active_tracks.values())
        for trk in track_list:
            is_lost = (trk.track.state == "lost")
            kin_box = trk.kalman.predict(timestamp, is_lost=is_lost)
            if predicted_boxes and trk.track.track_id in predicted_boxes:
                fb = predicted_boxes[trk.track.track_id]
                trk.predicted_bbox = (
                    0.85 * float(fb[0]) + 0.15 * kin_box[0],
                    0.85 * float(fb[1]) + 0.15 * kin_box[1],
                    0.85 * float(fb[2]) + 0.15 * kin_box[2],
                    0.85 * float(fb[3]) + 0.15 * kin_box[3],
                )
            else:
                trk.predicted_bbox = kin_box

        # 2. Split detections into high-confidence and low-confidence (ByteTrack strategy)
        high_dets = [d for d in norm_dets if d[1] >= self.high_score_threshold]
        low_dets = [d for d in norm_dets if self.low_score_threshold <= d[1] < self.high_score_threshold]

        # 3. First Association: Match all candidate tracks with high-confidence detections
        cost_1 = _build_cost_matrix(track_list, high_dets, self.max_area_ratio)
        matched_1, unmatched_trks_idx_1, unmatched_high_idx = _linear_assignment(
            cost_1, self.match_threshold
        )

        matched_tracks_1 = [track_list[t_idx] for t_idx, _ in matched_1]
        matched_high_dets = [high_dets[d_idx] for _, d_idx in matched_1]

        # 4. Second Association: Match remaining unmatched tracks with low-confidence detections
        unmatched_tracks_1 = [track_list[t_idx] for t_idx in unmatched_trks_idx_1]
        cost_2 = _build_cost_matrix(unmatched_tracks_1, low_dets, self.max_area_ratio)
        matched_2, unmatched_trks_idx_2, _ = _linear_assignment(
            cost_2, self.match_threshold_low
        )

        matched_tracks_2 = [unmatched_tracks_1[t_idx] for t_idx, _ in matched_2]
        matched_low_dets = [low_dets[d_idx] for _, d_idx in matched_2]

        # Combine all matched tracks and their corresponding detections
        all_matched_trks = matched_tracks_1 + matched_tracks_2
        all_matched_dets = matched_high_dets + matched_low_dets

        # 5. Update matched tracks
        for trk, (bbox, conf) in zip(all_matched_trks, all_matched_dets):
            trk.kalman.update(bbox)
            trk.misses = 0

            # State transition to active
            trk.track.state = "active"
            trk.track.last_seen = timestamp
            trk.track.hits += 1
            trk.track.confidence = float(conf)
            trk.track.bbox = bbox
            trk.track.update_locked_face_from_body()

            # Velocity and speed estimation
            vx, vy = trk.kalman.velocity
            trk.track.velocity = (float(vx), float(vy))
            trk.track.speed_px_s = round(float(math.hypot(vx, vy)), 2)

            # Update trajectory
            cx = (bbox[0] + bbox[2]) * 0.5
            cy = (bbox[1] + bbox[3]) * 0.5
            trk.track.trajectory.append((round(timestamp, 4), round(cx, 2), round(cy, 2)))
            if len(trk.track.trajectory) > self.max_trajectory_length:
                trk.track.trajectory = trk.track.trajectory[-self.max_trajectory_length:]

            # Confirmation rule: minimum hit count or high-confidence persistent observation
            if not trk.track.confirmed and (
                trk.track.hits >= self.confirm_hits or trk.track.confidence >= (self.high_score_threshold * 1.3)
            ):
                trk.track.confirmed = True

        # 6. Initialize new tracks from unmatched high-confidence detections
        for d_idx in unmatched_high_idx:
            bbox, conf = high_dets[d_idx]
            new_id = self._next_id
            self._next_id += 1

            # High confidence one-frame detection can enter confirmed state if threshold is 1
            confirmed = (self.confirm_hits <= 1)
            new_trk = _InternalTrack(
                track_id=new_id,
                session_id=self.session_id,
                bbox=bbox,
                confidence=conf,
                timestamp=timestamp,
                confirmed=confirmed,
            )
            self._active_tracks[new_id] = new_trk

        # 7. Handle unmatched tracks (state transition to "lost" or "ended")
        unmatched_tracks_final = [unmatched_tracks_1[t_idx] for t_idx in unmatched_trks_idx_2]
        retired_ids: List[int] = []

        for trk in unmatched_tracks_final:
            trk.misses += 1
            time_since_seen = timestamp - trk.track.last_seen

            # Unconfirmed spurious detections have a shorter lifetime
            eff_lifetime = self.lifetime if trk.track.confirmed else self.unconfirmed_lifetime

            if time_since_seen > eff_lifetime:
                # Transition: lost -> ended
                trk.track.state = "ended"
                trk.track.velocity = (0.0, 0.0)
                trk.track.speed_px_s = 0.0
                retired_ids.append(trk.track.track_id)
                self._ended_tracks[trk.track.track_id] = trk.track
            else:
                # Transition: active -> lost
                trk.track.state = "lost"
                trk.track.bbox = trk.predicted_bbox
                vx, vy = trk.kalman.velocity
                trk.track.velocity = (float(vx), float(vy))
                trk.track.speed_px_s = round(float(math.hypot(vx, vy)), 2)
                trk.track.update_locked_face_from_body()

        # Remove ended tracks from active collection
        for rid in retired_ids:
            if rid in self._active_tracks:
                del self._active_tracks[rid]

        # Cap ended tracks to prevent memory leaks in standing servers
        if len(self._ended_tracks) > 2000:
            oldest_keys = list(self._ended_tracks.keys())[:500]
            for k in oldest_keys:
                del self._ended_tracks[k]

        # If face observations were passed, associate them with active person tracks
        if face_observations is not None:
            self.associate_faces(face_observations, timestamp=timestamp)

        # Return tracks according to return_lost flag
        if return_lost:
            return [t.track for t in self._active_tracks.values() if t.track.state in ("active", "lost")]
        return [t.track for t in self._active_tracks.values() if t.track.state == "active"]

    def state(self) -> Dict[str, Any]:
        """Summary of current tracking state and counts."""
        active = self.active_tracks
        lost = self.lost_tracks
        confirmed = self.confirmed_tracks
        return {
            "session_id": self.session_id,
            "frame_count": self.frame_count,
            "active_tracks_count": len(active),
            "lost_tracks_count": len(lost),
            "confirmed_tracks_count": len(confirmed),
            "total_ended_tracks_count": len(self._ended_tracks),
            "tracks": [t.to_dict() for t in active],
        }
