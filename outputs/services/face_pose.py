"""Head pose estimation and facial landmark geometry evaluation module.

Evaluates observable face orientation from 5-point facial landmarks using
Perspective-n-Point (solvePnP) with a standard canonical 3D face model,
or geometric ratio-based estimation. Conforms strictly to services.face_contracts.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from services.face_contracts import FaceObservation, FacePose

logger = logging.getLogger(__name__)

# Standard canonical 3D face model points (in millimeters / arbitrary canonical units):
# Order:
#   [0] Right eye
#   [1] Left eye
#   [2] Nose tip
#   [3] Right mouth corner
#   [4] Left mouth corner
# Coordinate system:
#   X: right eye at -30 (subject's right = viewer's left in frontal view), left eye at +30
#   Y: eyes at +30 (upwards), mouth at -30 (downwards), nose at 0
#   Z: nose at 0 (protruding towards camera), eyes and mouth at -30 (planar base)
CANONICAL_3D_FACE_MODEL = np.array(
    [
        [-30.0, 30.0, -30.0],  # Right eye
        [30.0, 30.0, -30.0],   # Left eye
        [0.0, 0.0, 0.0],       # Nose tip
        [-20.0, -30.0, -30.0], # Right mouth corner
        [20.0, -30.0, -30.0],  # Left mouth corner
    ],
    dtype=np.float64,
)

# Base camera rotation mapping 3D model (Y-up, Z-out) to camera frame (Y-down, Z-away)
# at zero head rotation: 180 degree rotation around X axis.
_R_BASE = cv2.Rodrigues(np.array([np.pi, 0.0, 0.0], dtype=np.float64))[0]

# Orientation category threshold constants
FRONTAL_YAW_THRESHOLD: float = 15.0
FRONTAL_PITCH_THRESHOLD: float = 15.0
ANGLE_YAW_MIN: float = 15.0
ANGLE_YAW_MAX: float = 45.0
PROFILE_YAW_THRESHOLD: float = 45.0

# Geometry & confidence sanity thresholds
MIN_EYE_DISTANCE_PX: float = 3.0
MIN_LANDMARK_CONFIDENCE: float = 0.3
MAX_REPROJECTION_ERROR_RATIO: float = 0.5


def parse_landmarks(
    landmarks: Any,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Parse multiple landmark input formats into a canonical (5, 2) float64 array.

    Supported formats:
      - List/Tuple of 5 (x, y) or (x, y, conf) pairs
      - np.ndarray of shape (5, 2) or (5, 3) or (10,) or (15,)
      - Dict with keys ('right_eye', 'left_eye', 'nose', 'right_mouth', 'left_mouth')
        or ('re', 'le', 'nose_tip', 'rm', 'lm')
      - FaceObservation instance with a `.landmarks` attribute

    Returns:
      (points_2d, confidences): (5, 2) array of coordinates and optional (5,) array
      of confidences, or (None, None) if parsing fails.
    """
    if landmarks is None:
        return None, None

    # Handle FaceObservation or objects containing .landmarks
    if hasattr(landmarks, "landmarks"):
        landmarks = landmarks.landmarks
        if landmarks is None:
            return None, None

    # Handle dict representation
    if isinstance(landmarks, dict):
        re = landmarks.get("right_eye") or landmarks.get("re")
        le = landmarks.get("left_eye") or landmarks.get("le")
        nose = landmarks.get("nose") or landmarks.get("nose_tip")
        rm = landmarks.get("right_mouth") or landmarks.get("mouth_right") or landmarks.get("rm")
        lm = landmarks.get("left_mouth") or landmarks.get("mouth_left") or landmarks.get("lm")
        if re is not None and le is not None and nose is not None and rm is not None and lm is not None:
            landmarks = [re, le, nose, rm, lm]
        else:
            return None, None

    try:
        arr = np.asarray(landmarks, dtype=np.float64)
    except (ValueError, TypeError):
        return None, None

    # Flattened formats: 10 numbers or 15 numbers
    if arr.ndim == 1:
        if arr.size == 10:
            arr = arr.reshape((5, 2))
        elif arr.size == 15:
            arr = arr.reshape((5, 3))
        else:
            return None, None

    if arr.ndim != 2 or arr.shape[0] < 5:
        return None, None

    # Non-finite values check
    if not np.isfinite(arr[:5, :2]).all():
        return None, None

    confidences: Optional[np.ndarray] = None
    if arr.shape[1] >= 3:
        confidences = arr[:5, 2].copy()
    coords = arr[:5, :2].copy()
    return coords, confidences


def validate_face_landmarks(
    landmarks_2d: Optional[np.ndarray],
    min_eye_dist: float = MIN_EYE_DISTANCE_PX,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[bool, str]:
    """Validate 5-point landmark geometry for head pose estimation.

    Checks:
      - 5 finite 2D coordinates exist
      - Eye distance is above minimum threshold (not collapsed/degenerate)
      - Eye order: subject's right eye is to the left of subject's left eye in upright frame
      - Mouth is below eyes in upright frame
      - If bbox is given, landmarks reside reasonably within/near bbox boundaries

    Returns:
      (is_valid, reason)
    """
    if landmarks_2d is None:
        return False, "landmarks_missing"
    if landmarks_2d.shape != (5, 2):
        return False, f"invalid_shape_{landmarks_2d.shape}"
    if not np.isfinite(landmarks_2d).all():
        return False, "non_finite_coordinates"

    re, le, nose, rm, lm = landmarks_2d[:5]

    # Eye distance
    eye_dx = le[0] - re[0]
    eye_dy = le[1] - re[1]
    eye_dist = float(np.hypot(eye_dx, eye_dy))

    # Handle normalized coordinates ([0, 1]) scaling check
    is_normalized = (landmarks_2d >= -0.05).all() and (landmarks_2d <= 1.05).all() and (np.max(landmarks_2d) <= 1.5)
    effective_min_eye_dist = (min_eye_dist / 640.0) if is_normalized else min_eye_dist

    if eye_dist < effective_min_eye_dist:
        return False, f"eye_distance_too_small_{eye_dist:.2f}"

    # Eye roll angle
    roll_deg = float(np.degrees(np.arctan2(eye_dy, eye_dx)))
    # Rotate around eye midpoint into upright frame
    eye_center = (re + le) / 2.0
    rad = -np.radians(roll_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    r_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    upright = (landmarks_2d - eye_center) @ r_mat.T

    u_re, u_le, u_nose, u_rm, u_lm = upright[:5]

    # In upright frame, left eye must be to the right of right eye
    if u_le[0] <= u_re[0]:
        return False, "inverted_eye_order"

    # Mouth corners should be below eye line (positive Y in image coords)
    mouth_y = (u_rm[1] + u_lm[1]) / 2.0
    if mouth_y <= 0:
        return False, "mouth_above_eyes"

    # Mouth span check
    mouth_span = float(np.hypot(lm[0] - rm[0], lm[1] - rm[1]))
    effective_min_mouth = (min_eye_dist / 640.0) if is_normalized else (min_eye_dist * 0.5)
    if mouth_span < effective_min_mouth:
        return False, "mouth_corners_collapsed"

    # Optional BBox boundary check
    if bbox is not None and len(bbox) == 4:
        bx1, by1, bx2, by2 = bbox
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)
        # Margin of 40% of box size
        margin_x = 0.4 * bw
        margin_y = 0.4 * bh
        if (
            np.any(landmarks_2d[:, 0] < bx1 - margin_x)
            or np.any(landmarks_2d[:, 0] > bx2 + margin_x)
            or np.any(landmarks_2d[:, 1] < by1 - margin_y)
            or np.any(landmarks_2d[:, 1] > by2 + margin_y)
        ):
            return False, "landmarks_outside_bounding_box"

    return True, "valid"


def categorize_orientation(
    yaw: float,
    pitch: float,
    confidence: float = 1.0,
    is_reliable: bool = True,
    min_confidence: float = MIN_LANDMARK_CONFIDENCE,
) -> str:
    """Categorize head orientation into strict contract classes.

    Classes:
      - "frontal": abs(yaw) <= 15 and abs(pitch) <= 15
      - "left_angle": -45 <= yaw < -15
      - "right_angle": 15 < yaw <= 45
      - "profile": abs(yaw) > 45
      - "uncertain": landmark confidence or geometry is distorted, or pitch out of bounds
    """
    if (
        not is_reliable
        or confidence < min_confidence
        or not np.isfinite([yaw, pitch]).all()
        or abs(yaw) > 90.0
        or abs(pitch) > 60.0
    ):
        return "uncertain"

    if abs(yaw) > PROFILE_YAW_THRESHOLD:
        return "profile"
    elif -ANGLE_YAW_MAX <= yaw < -ANGLE_YAW_MIN:
        return "left_angle"
    elif ANGLE_YAW_MIN < yaw <= ANGLE_YAW_MAX:
        return "right_angle"
    elif abs(yaw) <= FRONTAL_YAW_THRESHOLD and abs(pitch) <= FRONTAL_PITCH_THRESHOLD:
        return "frontal"
    else:
        # abs(yaw) <= 15 but pitch > 15 (e.g. looking straight up/down beyond frontal limit)
        return "uncertain"


def estimate_pose_geometric(
    landmarks: Any,
    confidence: float = 1.0,
) -> FacePose:
    """Calculate pose from 5 facial landmarks using ratio-based landmark geometry.

    Computes:
      - roll: Eye line angle in image plane (degrees, tilt)
      - yaw: Nose horizontal offset relative to eye center in roll-corrected frame
             (degrees, -90 to +90: negative left, positive right)
      - pitch: Vertical nose offset relative to eye-mouth distance
               (degrees, -90 to +90: negative down, positive up)
      - category: strict orientation classification
    """
    pts, confs = parse_landmarks(landmarks)
    if confs is not None and len(confs) > 0:
        confidence = min(confidence, float(np.mean(confs)))

    is_valid, _ = validate_face_landmarks(pts)
    if not is_valid or pts is None:
        return FacePose(
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            category="uncertain",
            confidence=0.0,
        )

    re, le, nose, rm, lm = pts[:5]

    # 1. Roll: Angle between eyes
    dx = le[0] - re[0]
    dy = le[1] - re[1]
    roll_deg = float(np.degrees(np.arctan2(dy, dx)))

    # 2. Rotate to upright frame (eye center as origin)
    eye_center = (re + le) / 2.0
    rad = -np.radians(roll_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    r_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    upright = (pts - eye_center) @ r_mat.T

    u_re, u_le, u_nose, u_rm, u_lm = upright[:5]

    # Eye span
    half_eye_span = (u_le[0] - u_re[0]) / 2.0
    if half_eye_span <= 0:
        return FacePose(
            yaw=0.0,
            pitch=0.0,
            roll=round(roll_deg, 1),
            category="uncertain",
            confidence=0.0,
        )

    # 3. Yaw: Nose X offset from eye midpoint (0, 0 in eye_center frame)
    rx = u_nose[0] / half_eye_span
    # Map normalized horizontal ratio to yaw degrees (-90 to +90)
    yaw_deg = float(np.degrees(np.arcsin(np.clip(rx * 0.85, -1.0, 1.0))))
    yaw_deg = float(np.clip(yaw_deg, -90.0, 90.0))

    # 4. Pitch: Vertical position of nose between eyes and mouth
    mouth_y = (u_rm[1] + u_lm[1]) / 2.0
    if mouth_y <= 0:
        return FacePose(
            yaw=round(yaw_deg, 1),
            pitch=0.0,
            roll=round(roll_deg, 1),
            category="uncertain",
            confidence=0.0,
        )

    # In canonical frontal face, nose vertical position is approx 53.5% down from eyes to mouth
    ry = u_nose[1] / mouth_y
    # ry < 0.535 -> nose closer to eyes (looking up, positive pitch)
    # ry > 0.535 -> nose closer to mouth (looking down, negative pitch)
    pitch_deg = float(np.clip((0.535 - ry) * 90.0, -90.0, 90.0))

    category = categorize_orientation(
        yaw=yaw_deg,
        pitch=pitch_deg,
        confidence=confidence,
        is_reliable=True,
    )

    return FacePose(
        yaw=round(yaw_deg, 1),
        pitch=round(pitch_deg, 1),
        roll=round(roll_deg, 1),
        category=category,
        confidence=round(float(confidence), 3),
    )


def estimate_pose_pnp(
    landmarks: Any,
    image_size: Optional[Tuple[int, int]] = None,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    confidence: float = 1.0,
) -> FacePose:
    """Calculate pose from 5 facial landmarks using Perspective-n-Point (cv2.solvePnP).

    Uses CANONICAL_3D_FACE_MODEL reference points:
      - nose (0, 0, 0)
      - right eye (-30, 30, -30)
      - left eye (30, 30, -30)
      - right mouth (-20, -30, -30)
      - left mouth (20, -30, -30)

    Calculates:
      - yaw (degrees, -90 to +90: negative left, positive right)
      - pitch (degrees, negative down, positive up)
      - roll (degrees, tilt)
    """
    pts, confs = parse_landmarks(landmarks)
    if confs is not None and len(confs) > 0:
        confidence = min(confidence, float(np.mean(confs)))

    is_valid, reason = validate_face_landmarks(pts)
    if not is_valid or pts is None:
        return FacePose(
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            category="uncertain",
            confidence=0.0,
        )

    pts_2d = pts.copy()

    # Handle normalized coordinates by scaling to image size or standard 640x480 frame
    is_normalized = (pts_2d >= -0.05).all() and (pts_2d <= 1.05).all() and (np.max(pts_2d) <= 1.5)
    if is_normalized:
        w = float(image_size[0]) if image_size else 640.0
        h = float(image_size[1]) if image_size else 480.0
        pts_2d[:, 0] *= w
        pts_2d[:, 1] *= h

    # Construct camera calibration matrix if not provided
    if camera_matrix is None:
        if image_size:
            w, h = float(image_size[0]), float(image_size[1])
            focal_length = max(w, h)
            cx, cy = w / 2.0, h / 2.0
        else:
            cx = float(np.mean(pts_2d[:, 0]))
            cy = float(np.mean(pts_2d[:, 1]))
            eye_dist = float(np.linalg.norm(pts_2d[1] - pts_2d[0]))
            focal_length = max(eye_dist * 8.0, 150.0)

        cam_mat = np.array(
            [[focal_length, 0.0, cx], [0.0, focal_length, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    else:
        cam_mat = np.asarray(camera_matrix, dtype=np.float64)

    if dist_coeffs is None:
        dist = np.zeros((4, 1), dtype=np.float64)
    else:
        dist = np.asarray(dist_coeffs, dtype=np.float64)

    try:
        # OpenCV SQPNP is fast, robust, and supports 5 points without planar degeneracies
        success, rvec, tvec = cv2.solvePnP(
            CANONICAL_3D_FACE_MODEL,
            pts_2d,
            cam_mat,
            dist,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error as e:
        logger.debug("PnP solve failed with OpenCV error: %s", e)
        success = False

    if not success:
        # Fall back to geometric ratio estimation
        return estimate_pose_geometric(landmarks=pts_2d, confidence=confidence)

    # Compute rotation matrix
    r_cam, _ = cv2.Rodrigues(rvec)

    # Relative rotation of head with respect to frontal camera pose:
    # R_head = R_BASE^T * R_cam
    r_head = _R_BASE.T @ r_cam

    # Euler angle decomposition (pitch around X, yaw around Y, roll around Z)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(r_head)

    # Invert signs to match strict contract:
    #   pitch: negative = down, positive = up
    #   yaw: negative = left, positive = right
    #   roll: tilt
    pitch_deg = -float(angles[0])
    yaw_deg = float(angles[1])
    roll_deg = -float(angles[2])

    # Check for numerical validity or extreme solver divergence
    if (
        not np.isfinite([yaw_deg, pitch_deg, roll_deg]).all()
        or abs(yaw_deg) > 90.0
        or abs(pitch_deg) > 90.0
    ):
        # Fall back to geometric ratio estimation
        return estimate_pose_geometric(landmarks=pts_2d, confidence=confidence)

    # Calculate reprojection error to gauge fit quality
    proj_pts, _ = cv2.projectPoints(CANONICAL_3D_FACE_MODEL, rvec, tvec, cam_mat, dist)
    proj_pts = proj_pts.reshape(-1, 2)
    rep_err = float(np.mean(np.linalg.norm(pts_2d - proj_pts, axis=1)))
    eye_dist = float(np.linalg.norm(pts_2d[1] - pts_2d[0]))
    err_ratio = (rep_err / eye_dist) if eye_dist > 1e-4 else 1.0

    is_reliable = err_ratio <= MAX_REPROJECTION_ERROR_RATIO
    adjusted_conf = confidence * max(0.0, min(1.0, 1.0 - err_ratio * 0.8))

    category = categorize_orientation(
        yaw=yaw_deg,
        pitch=pitch_deg,
        confidence=adjusted_conf,
        is_reliable=is_reliable,
    )

    return FacePose(
        yaw=round(float(yaw_deg), 1),
        pitch=round(float(pitch_deg), 1),
        roll=round(float(roll_deg), 1),
        category=category,
        confidence=round(float(adjusted_conf), 3),
    )


def estimate_face_pose(
    landmarks: Any,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    confidence: float = 1.0,
    method: str = "auto",
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
) -> FacePose:
    """Primary entry point for facial pose estimation.

    Parameters:
      - landmarks: 5-point facial landmarks or FaceObservation instance
      - bbox: Optional bounding box (x1, y1, x2, y2)
      - image_size: Optional frame dimensions (width, height)
      - confidence: Base landmark / detection confidence
      - method: "auto", "pnp", or "geometric"
      - camera_matrix: Optional 3x3 camera matrix
      - dist_coeffs: Optional distortion coefficients

    Returns:
      FacePose instance conforming to services.face_contracts
    """
    pts, confs = parse_landmarks(landmarks)
    if confs is not None and len(confs) > 0:
        confidence = min(confidence, float(np.mean(confs)))

    is_valid, reason = validate_face_landmarks(pts, bbox=bbox)
    if not is_valid or pts is None:
        return FacePose(
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            category="uncertain",
            confidence=0.0,
        )

    if method == "geometric":
        return estimate_pose_geometric(pts, confidence=confidence)
    elif method == "pnp":
        return estimate_pose_pnp(
            pts,
            image_size=image_size,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            confidence=confidence,
        )
    else:  # "auto": Try PnP with geometric verification/fallback
        pose = estimate_pose_pnp(
            pts,
            image_size=image_size,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            confidence=confidence,
        )
        if pose.category == "uncertain" and confidence >= MIN_LANDMARK_CONFIDENCE:
            # Attempt geometric estimation before declaring uncertain
            geom_pose = estimate_pose_geometric(pts, confidence=confidence)
            if geom_pose.category != "uncertain":
                return geom_pose
        return pose


def project_reference_model(
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    camera_matrix: Optional[np.ndarray] = None,
    image_size: Tuple[int, int] = (640, 480),
    distance: float = 400.0,
) -> np.ndarray:
    """Project standard canonical 3D model into 2D camera coordinates given Euler angles.

    Useful for unit testing, synthetic ground-truth generation, and calibration.
    """
    w, h = float(image_size[0]), float(image_size[1])
    if camera_matrix is None:
        focal_length = max(w, h)
        cam_mat = np.array(
            [[focal_length, 0.0, w / 2.0], [0.0, focal_length, h / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    else:
        cam_mat = np.asarray(camera_matrix, dtype=np.float64)

    dist = np.zeros((4, 1), dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [distance]], dtype=np.float64)

    # Invert signs from contract convention to OpenCV camera rotations:
    #   pitch: negative down, positive up -> -pitch around X
    #   yaw: negative left, positive right -> +yaw around Y
    #   roll: tilt -> -roll around Z
    rx = cv2.Rodrigues(np.array([-np.radians(pitch), 0.0, 0.0], dtype=np.float64))[0]
    ry = cv2.Rodrigues(np.array([0.0, np.radians(yaw), 0.0], dtype=np.float64))[0]
    rz = cv2.Rodrigues(np.array([0.0, 0.0, -np.radians(roll)], dtype=np.float64))[0]

    r_cam = _R_BASE @ rx @ ry @ rz
    rvec, _ = cv2.Rodrigues(r_cam)

    pts2d, _ = cv2.projectPoints(CANONICAL_3D_FACE_MODEL, rvec, tvec, cam_mat, dist)
    return pts2d.reshape(-1, 2)


class FacePoseEstimator:
    """Stateful, configurable face pose estimator for pipeline integration."""

    def __init__(
        self,
        method: str = "auto",
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        min_eye_distance: float = MIN_EYE_DISTANCE_PX,
        min_confidence: float = MIN_LANDMARK_CONFIDENCE,
    ) -> None:
        self.method = method
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.min_eye_distance = min_eye_distance
        self.min_confidence = min_confidence

    def estimate(
        self,
        landmarks: Any,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        confidence: float = 1.0,
    ) -> FacePose:
        """Estimate 3D head pose from landmarks."""
        return estimate_face_pose(
            landmarks=landmarks,
            bbox=bbox,
            image_size=image_size,
            confidence=confidence,
            method=self.method,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
        )

    def estimate_observation(self, obs: FaceObservation) -> FacePose:
        """Estimate and attach pose to an existing FaceObservation instance."""
        pose = self.estimate(
            landmarks=obs.landmarks,
            bbox=obs.bbox,
            confidence=obs.confidence,
        )
        obs.pose = pose
        return pose

    def validate(
        self,
        landmarks: Any,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[bool, str]:
        """Validate landmark geometry."""
        pts, _ = parse_landmarks(landmarks)
        return validate_face_landmarks(
            pts,
            min_eye_dist=self.min_eye_distance,
            bbox=bbox,
        )

    def categorize(
        self,
        yaw: float,
        pitch: float,
        confidence: float = 1.0,
        is_reliable: bool = True,
    ) -> str:
        """Categorize orientation given yaw and pitch angles."""
        return categorize_orientation(
            yaw=yaw,
            pitch=pitch,
            confidence=confidence,
            is_reliable=is_reliable,
            min_confidence=self.min_confidence,
        )
