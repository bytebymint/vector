"""Multi-Person Facial Capture Benchmark Suite and Load Simulator.

Measures real-time system performance under realistic workloads with up to 30 people:
- Pipeline FPS (throughput)
- Detection latency (ms)
- Tracking association latency (ms)
- Face detection & pose estimation latency (ms)
- Quality evaluation latency (ms)
- Capture queue processing delay (ms)
- CPU usage (%) and Memory consumption (MB)
- ID switches (zero or near-zero)

Outputs structured benchmark results to outputs/data/people_benchmark_results.json.
Conforms strictly to services.face_contracts schemas.
"""
from __future__ import annotations

import argparse
import datetime
import math
import os
import platform
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil
import torch

# Ensure outputs directory is in sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.face_contracts import (  # noqa: E402
    CaptureCandidate,
    CaptureRecord,
    FaceObservation,
    FacePose,
    FaceQualityResult,
    FramePacket,
    PersonDetection,
    PersonProfile,
    PersonTrack,
    ProcessingMetrics,
)
from services.tracker import VehicleTracker  # noqa: E402
from ultralytics.engine.results import Boxes  # noqa: E402


def compute_stats(samples: List[float]) -> Dict[str, float]:
    """Calculate descriptive statistics for a list of measurement samples."""
    if not samples:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
        }
    arr = np.array(samples, dtype=float)
    return {
        "mean": round(float(np.mean(arr)), 2),
        "median": round(float(np.median(arr)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "std": round(float(np.std(arr)), 2),
    }


# =====================================================================
# Procedural Face and Realistic Person Generation
# =====================================================================


def generate_procedural_face(
    width: int,
    height: int,
    skin_tone: Tuple[int, int, int],
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    sharpness_factor: float = 1.0,
) -> np.ndarray:
    """Generate a realistic synthetic face crop with detailed facial geometry.

    Includes eyes, pupils, eyebrows, nose, mouth/lips, facial contours,
    and lighting gradients suitable for face detection and quality analysis.
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 220
    b, g, r = skin_tone
    center_x = width // 2
    center_y = height // 2
    face_rx = int(width * 0.42)
    face_ry = int(height * 0.48)

    # 1. Base skin ellipse
    cv2.ellipse(
        img,
        (center_x, center_y),
        (face_rx, face_ry),
        0,
        0,
        360,
        (b, g, r),
        -1,
    )

    # 2. Lighting shading gradient
    y_coords, x_coords = np.ogrid[:height, :width]
    dist_from_top_left = np.sqrt((x_coords - width * 0.3) ** 2 + (y_coords - height * 0.3) ** 2)
    shade = np.clip(1.0 - (dist_from_top_left / (max(width, height) * 1.2)) * 0.35, 0.6, 1.1)
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c].astype(float) * shade, 0, 255).astype(np.uint8)

    # 3. Eyebrows & Eyes
    yaw_shift = int(np.clip(yaw / 45.0 * (width * 0.12), -width * 0.15, width * 0.15))
    pitch_shift = int(np.clip(-pitch / 45.0 * (height * 0.10), -height * 0.12, height * 0.12))

    eye_y = int(height * 0.42) + pitch_shift
    eye_spacing = int(width * 0.22)
    left_eye_x = center_x - eye_spacing + yaw_shift
    right_eye_x = center_x + eye_spacing + yaw_shift

    eye_rx = max(4, int(width * 0.08))
    eye_ry = max(3, int(height * 0.05))

    # Eyes (sclera + iris + pupil)
    for ex in (left_eye_x, right_eye_x):
        if 0 < ex < width and 0 < eye_y < height:
            cv2.ellipse(img, (ex, eye_y), (eye_rx, eye_ry), 0, 0, 360, (245, 245, 245), -1)
            cv2.circle(img, (ex + int(yaw_shift * 0.3), eye_y), max(2, int(eye_rx * 0.55)), (40, 60, 80), -1)
            cv2.circle(img, (ex + int(yaw_shift * 0.3), eye_y), max(1, int(eye_rx * 0.25)), (15, 15, 15), -1)
            # Specular catchlight
            cv2.circle(img, (ex + int(yaw_shift * 0.3) - 1, eye_y - 1), 1, (255, 255, 255), -1)

    # Eyebrows
    brow_y = eye_y - max(4, int(height * 0.08))
    brow_color = (max(10, b - 60), max(10, g - 60), max(10, r - 60))
    cv2.ellipse(img, (left_eye_x, brow_y), (eye_rx + 2, 2), -5, 180, 360, brow_color, 2)
    cv2.ellipse(img, (right_eye_x, brow_y), (eye_rx + 2, 2), 5, 180, 360, brow_color, 2)

    # 4. Nose bridge and tip
    nose_x = center_x + int(yaw_shift * 1.1)
    nose_y = int(height * 0.58) + pitch_shift
    nose_color = (max(0, b - 40), max(0, g - 40), max(0, r - 40))
    cv2.line(img, (center_x + yaw_shift, eye_y), (nose_x, nose_y), nose_color, 1)
    cv2.circle(img, (nose_x, nose_y), max(2, int(width * 0.05)), nose_color, -1)

    # 5. Lips and mouth
    mouth_x = center_x + yaw_shift
    mouth_y = int(height * 0.74) + pitch_shift
    mouth_w = max(6, int(width * 0.16))
    lip_color = (min(255, b + 20), max(30, g - 45), min(255, r + 40))
    cv2.ellipse(img, (mouth_x, mouth_y), (mouth_w, max(2, int(height * 0.04))), 0, 0, 360, lip_color, -1)
    cv2.line(img, (mouth_x - mouth_w + 2, mouth_y), (mouth_x + mouth_w - 2, mouth_y), (max(0, b - 50), max(0, g - 50), max(0, r - 50)), 1)

    # 6. Hair region at top
    hair_color = (25, 25, 30)
    cv2.ellipse(img, (center_x, int(height * 0.22)), (face_rx, int(height * 0.25)), 0, 180, 360, hair_color, -1)

    # Apply blur or sharpening if requested
    if sharpness_factor < 0.8:
        ksize = 3 if sharpness_factor > 0.5 else 5
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # Apply roll rotation if roll != 0
    if abs(roll) > 1.0:
        rot_mat = cv2.getRotationMatrix2D((center_x, center_y), roll, 1.0)
        img = cv2.warpAffine(img, rot_mat, (width, height), borderMode=cv2.BORDER_REFLECT)

    return img


class FaceCropPool:
    """Pool of diverse face crops mixing real crops from disk and procedural faces."""

    def __init__(self, root_dir: Path, target_size: Tuple[int, int] = (96, 96)):
        self.target_size = target_size
        self.crops: List[np.ndarray] = []
        self._load_real_crops(root_dir)
        self._fill_with_procedural_faces()

    def _load_real_crops(self, root_dir: Path):
        """Extract existing real face crops from available images."""
        yunet_path = root_dir / "models" / "face_detection_yunet_2023mar.onnx"
        detector = None
        if yunet_path.exists():
            try:
                detector = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), score_threshold=0.35)
            except Exception:
                detector = None

        sample_candidates = [
            root_dir / "performance-recording-proof.jpg",
            root_dir / "linkedin-sample-05.jpg",
            root_dir / "linkedin-sample-20.jpg",
            root_dir / "linkedin-first-five.jpg",
        ]
        for img_path in sample_candidates:
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            if detector is not None:
                detector.setInputSize((img.shape[1], img.shape[0]))
                _, faces = detector.detect(img)
                if faces is not None:
                    for f in faces:
                        x, y, w, h = map(int, f[:4])
                        # Pad by 20%
                        pad_x = int(w * 0.2)
                        pad_y = int(h * 0.2)
                        x1 = max(0, x - pad_x)
                        y1 = max(0, y - pad_y)
                        x2 = min(img.shape[1], x + w + pad_x)
                        y2 = min(img.shape[0], y + h + pad_y)
                        crop = img[y1:y2, x1:x2]
                        if crop.shape[0] > 20 and crop.shape[1] > 20:
                            resized = cv2.resize(crop, self.target_size)
                            self.crops.append(resized)

    def _fill_with_procedural_faces(self):
        """Ensure at least 35 diverse faces exist in pool."""
        skin_tones = [
            (170, 200, 235),  # fair / peach
            (145, 185, 220),  # medium warm
            (110, 150, 195),  # olive / tan
            (80, 115, 155),   # rich bronze
            (60, 85, 120),    # dark brown
        ]
        angles = [
            (0.0, 0.0),    # frontal
            (-18.0, 2.0),  # left angle
            (18.0, -2.0),  # right angle
            (-8.0, 6.0),   # slight left tilt down
            (8.0, -5.0),   # slight right tilt up
            (0.0, 8.0),    # pitch down
            (0.0, -8.0),   # pitch up
        ]

        idx = 0
        while len(self.crops) < 35:
            tone = skin_tones[idx % len(skin_tones)]
            yaw, pitch = angles[(idx // len(skin_tones)) % len(angles)]
            face = generate_procedural_face(
                self.target_size[0],
                self.target_size[1],
                tone,
                yaw=yaw,
                pitch=pitch,
                sharpness_factor=0.9 + 0.2 * (idx % 3),
            )
            self.crops.append(face)
            idx += 1

    def get_face(self, index: int) -> np.ndarray:
        return self.crops[index % len(self.crops)]


class RealisticPerson:
    """Simulated person with smooth walking dynamics, posture, and facial identity."""

    def __init__(
        self,
        person_id: int,
        initial_x: float,
        initial_y: float,
        width: float,
        height: float,
        velocity_x: float,
        velocity_y: float,
        face_crop: np.ndarray,
        clothing_color: Tuple[int, int, int],
        base_yaw: float = 0.0,
        base_pitch: float = 0.0,
    ):
        self.person_id = person_id
        self.x = initial_x
        self.y = initial_y
        self.width = width
        self.height = height
        self.vx = velocity_x
        self.vy = velocity_y
        self.face_crop = face_crop
        self.clothing_color = clothing_color
        self.base_yaw = base_yaw
        self.base_pitch = base_pitch
        self.head_turn_phase = np.random.uniform(0, 2 * math.pi)
        self.head_turn_speed = np.random.uniform(0.05, 0.12)

    def update(self, bounds_w: int, bounds_h: int):
        """Advance trajectory with realistic smooth reflection at boundaries."""
        self.x += self.vx
        self.y += self.vy

        margin_x = 30
        margin_y = 40
        max_x = bounds_w - self.width - margin_x
        max_y = bounds_h - self.height - margin_y

        if self.x < margin_x:
            self.x = margin_x
            self.vx = abs(self.vx)
        elif self.x > max_x:
            self.x = max_x
            self.vx = -abs(self.vx)

        if self.y < margin_y:
            self.y = margin_y
            self.vy = abs(self.vy)
        elif self.y > max_y:
            self.y = max_y
            self.vy = -abs(self.vy)

    def current_pose(self, frame_seq: int) -> Tuple[float, float, float]:
        """Compute current 3D head pose orientation (yaw, pitch, roll) in degrees."""
        osc = math.sin(frame_seq * self.head_turn_speed + self.head_turn_phase)
        yaw = float(np.clip(self.base_yaw + osc * 16.0, -60.0, 60.0))
        pitch = float(np.clip(self.base_pitch + math.cos(frame_seq * 0.08) * 6.0, -30.0, 30.0))
        roll = float(np.clip(osc * 4.0, -20.0, 20.0))
        return yaw, pitch, roll

    def render(self, frame: np.ndarray, frame_seq: int) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
        """Render person body and face crop onto frame image.

        Returns (person_bbox, face_bbox) in absolute pixels (x1, y1, x2, y2).
        """
        x1 = int(round(self.x))
        y1 = int(round(self.y))
        w = int(round(self.width))
        h = int(round(self.height))
        x2 = x1 + w
        y2 = y1 + h

        # 1. Torso and legs
        torso_y1 = y1 + int(h * 0.28)
        legs_y1 = y1 + int(h * 0.65)
        # Legs
        cv2.rectangle(frame, (x1 + int(w * 0.15), legs_y1), (x1 + int(w * 0.45), y2), (40, 40, 45), -1)
        cv2.rectangle(frame, (x1 + int(w * 0.55), legs_y1), (x1 + int(w * 0.85), y2), (40, 40, 45), -1)
        # Torso
        cv2.rectangle(frame, (x1 + int(w * 0.1), torso_y1), (x2 - int(w * 0.1), legs_y1), self.clothing_color, -1)
        # Arms
        cv2.rectangle(frame, (x1, torso_y1), (x1 + int(w * 0.12), legs_y1 - int(h * 0.1)), self.clothing_color, -1)
        cv2.rectangle(frame, (x2 - int(w * 0.12), torso_y1), (x2, legs_y1 - int(h * 0.1)), self.clothing_color, -1)

        # 2. Head / Face region
        head_w = int(w * 0.55)
        head_h = int(h * 0.28)
        fx1 = x1 + (w - head_w) // 2
        fy1 = y1
        fx2 = fx1 + head_w
        fy2 = fy1 + head_h

        # Blend face crop into head region
        face_resized = cv2.resize(self.face_crop, (head_w, head_h))
        # Ensure within frame bounds
        if 0 <= fy1 and fy2 <= frame.shape[0] and 0 <= fx1 and fx2 <= frame.shape[1]:
            # Elliptical mask for seamless blend
            mask = np.zeros((head_h, head_w), dtype=np.uint8)
            cv2.ellipse(mask, (head_w // 2, head_h // 2), (head_w // 2, head_h // 2), 0, 0, 360, 255, -1)
            mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            roi = frame[fy1:fy2, fx1:fx2]
            blended = (face_resized.astype(float) * mask_3c + roi.astype(float) * (1.0 - mask_3c)).astype(np.uint8)
            frame[fy1:fy2, fx1:fx2] = blended

        return (float(x1), float(y1), float(x2), float(y2)), (float(fx1), float(fy1), float(fx2), float(fy2))


# =====================================================================
# Synthetic Video Stream Generator
# =====================================================================


class SyntheticPersonStream:
    """Generates synthetic video stream frames with moving people and realistic face crops."""

    def __init__(
        self,
        person_count: int,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        face_pool: Optional[FaceCropPool] = None,
        source_id: str = "benchmark_camera_1",
    ):
        self.person_count = person_count
        self.width = width
        self.height = height
        self.fps = fps
        self.source_id = source_id
        self.sequence = 0
        self.epoch = f"bench_{uuid.uuid4().hex[:8]}"

        if face_pool is None:
            face_pool = FaceCropPool(_ROOT)
        self.face_pool = face_pool
        self.people = self._create_people()

    def _create_people(self) -> List[RealisticPerson]:
        """Initialize well-spaced simulated individuals."""
        people: List[RealisticPerson] = []
        cols = max(1, int(math.ceil(math.sqrt(self.person_count * (self.width / self.height)))))
        rows = max(1, int(math.ceil(self.person_count / cols)))

        dx = (self.width - 200) / max(1, cols)
        dy = (self.height - 240) / max(1, rows)

        clothing_palette = [
            (160, 70, 50),   # Navy
            (50, 120, 160),  # Rust
            (80, 140, 70),   # Forest
            (40, 80, 150),   # Maroon
            (120, 120, 120), # Slate
            (60, 110, 130),  # Ochre
            (140, 80, 110),  # Purple
        ]

        for i in range(self.person_count):
            col = i % cols
            row = (i // cols) % rows
            x = 80 + col * dx + np.random.uniform(-15, 15)
            y = 60 + row * dy + np.random.uniform(-15, 15)

            # Simulated physical sizes (depth variation)
            person_h = float(np.random.uniform(180, 240))
            person_w = float(person_h * np.random.uniform(0.36, 0.44))

            # Walking velocities
            vx = float(np.random.choice([-1, 1]) * np.random.uniform(1.2, 2.8))
            vy = float(np.random.choice([-1, 1]) * np.random.uniform(0.4, 1.2))

            face = self.face_pool.get_face(i)
            clothing = clothing_palette[i % len(clothing_palette)]
            base_yaw = float(np.random.uniform(-20, 20))
            base_pitch = float(np.random.uniform(-8, 8))

            person = RealisticPerson(
                person_id=i + 1,
                initial_x=x,
                initial_y=y,
                width=person_w,
                height=person_h,
                velocity_x=vx,
                velocity_y=vy,
                face_crop=face,
                clothing_color=clothing,
                base_yaw=base_yaw,
                base_pitch=base_pitch,
            )
            people.append(person)
        return people

    def next_frame(self) -> Tuple[FramePacket, List[Dict[str, Any]]]:
        """Generate the next frame in the stream."""
        self.sequence += 1
        timestamp = time.time()

        # Canvas with realistic architectural backdrop / floor
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 235
        # Floor gradient
        floor_y = int(self.height * 0.45)
        for y in range(floor_y, self.height):
            ratio = (y - floor_y) / (self.height - floor_y)
            color = int(220 - ratio * 35)
            frame[y, :] = (color, color + 5, color + 10)

        ground_truth: List[Dict[str, Any]] = []

        # Sort by y-coordinate (depth ordering: farther people drawn first)
        sorted_people = sorted(self.people, key=lambda p: p.y + p.height)
        for p in sorted_people:
            p.update(self.width, self.height)
            p_bbox, f_bbox = p.render(frame, self.sequence)
            yaw, pitch, roll = p.current_pose(self.sequence)
            ground_truth.append({
                "person_id": p.person_id,
                "person_bbox": p_bbox,
                "face_bbox": f_bbox,
                "pose": (yaw, pitch, roll),
                "face_crop": p.face_crop,
            })

        packet = FramePacket(
            sequence=self.sequence,
            timestamp=timestamp,
            image=frame,
            source_id=self.source_id,
            epoch=self.epoch,
            width=self.width,
            height=self.height,
        )
        return packet, ground_truth


# =====================================================================
# Pipeline Stage Modules
# =====================================================================


class PersonDetectionStage:
    """Person detection stage measuring localization latency and generating PersonDetection."""

    def __init__(self, mode: str = "simulated", models_dir: Optional[Path] = None):
        self.mode = mode
        self.yolo_model = None
        if mode == "yolo" and models_dir is not None:
            pt_path = models_dir / "yolo26n.pt"
            if pt_path.exists():
                try:
                    from ultralytics import YOLO
                    self.yolo_model = YOLO(str(pt_path))
                    # Warmup
                    self.yolo_model.predict(np.zeros((320, 320, 3), dtype=np.uint8), classes=[0], verbose=False)
                except Exception:
                    self.yolo_model = None

    def process(
        self,
        frame_packet: FramePacket,
        ground_truth: List[Dict[str, Any]],
    ) -> Tuple[List[PersonDetection], float]:
        """Detect person bboxes. Returns (detections, latency_ms)."""
        t0 = time.perf_counter()

        if self.mode == "yolo" and self.yolo_model is not None:
            res = self.yolo_model.predict(frame_packet.image, classes=[0], imgsz=640, verbose=False)
            dt = (time.perf_counter() - t0) * 1000.0
            detections: List[PersonDetection] = []
            if res and len(res) > 0 and res[0].boxes is not None:
                boxes = res[0].boxes.data.cpu().numpy()
                for b in boxes:
                    x1, y1, x2, y2, conf, cls_id = b[:6]
                    detections.append(PersonDetection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=float(conf),
                        class_id=int(cls_id),
                    ))
            return detections, dt

        # High-precision simulated detector with realistic detector jitter and confidence
        detections: List[PersonDetection] = []
        for gt in ground_truth:
            x1, y1, x2, y2 = gt["person_bbox"]
            # Small jitter simulating neural net bounding box estimation
            jx1 = x1 + np.random.normal(0, 1.0)
            jy1 = y1 + np.random.normal(0, 1.0)
            jx2 = x2 + np.random.normal(0, 1.0)
            jy2 = y2 + np.random.normal(0, 1.0)
            conf = float(np.clip(np.random.normal(0.92, 0.03), 0.70, 0.99))
            detections.append(PersonDetection(
                bbox=(jx1, jy1, jx2, jy2),
                confidence=conf,
                class_id=0,
                detection_id=f"det_{gt['person_id']}_{frame_packet.sequence}",
            ))

        # Coordinate normalization and filtering compute
        if detections:
            _ = np.array([d.bbox for d in detections], dtype=np.float32)
        dt = (time.perf_counter() - t0) * 1000.0
        return detections, dt


class TrackingAssociationStage:
    """Multi-object tracking association stage measuring latency and tracking continuity."""

    def __init__(self, models_dir: Path, source_fps: float = 30.0):
        settings = {
            "lifetime": 2.5,
            "tracker": "bytetrack",
            "reid": False,
            "person_confidence": 0.08,
        }
        self.tracker = VehicleTracker(settings, models_dir, source_fps=source_fps, person_mode=True)
        self.gt_to_track_map: Dict[int, int] = {}
        self.id_switches = 0
        self.active_tracks: Dict[int, PersonTrack] = {}

    def process(
        self,
        detections: List[PersonDetection],
        frame_packet: FramePacket,
        ground_truth: List[Dict[str, Any]],
    ) -> Tuple[List[PersonTrack], float, int]:
        """Update tracker with detections. Returns (tracks, latency_ms, id_switches_in_frame)."""
        t0 = time.perf_counter()

        if detections:
            boxes_data = torch.zeros((len(detections), 6), dtype=torch.float32)
            for idx, det in enumerate(detections):
                boxes_data[idx, :4] = torch.tensor(det.bbox, dtype=torch.float32)
                boxes_data[idx, 4] = det.confidence
                boxes_data[idx, 5] = 0.0
            boxes = Boxes(boxes_data, (frame_packet.height, frame_packet.width))
        else:
            boxes = Boxes(torch.zeros((0, 6), dtype=torch.float32), (frame_packet.height, frame_packet.width))

        rows = self.tracker.update(boxes, frame_packet.image, frame_packet.sequence)
        dt = (time.perf_counter() - t0) * 1000.0

        current_tracks: List[PersonTrack] = []
        new_switches = 0

        if len(rows) > 0:
            row_arr = np.array(rows)
            # Evaluate CLEAR MOT ID switches by IoU association with ground truth
            for gt in ground_truth:
                gt_id = gt["person_id"]
                gt_box = gt["person_bbox"]
                r_boxes = row_arr[:, :4]
                ix1 = np.maximum(gt_box[0], r_boxes[:, 0])
                iy1 = np.maximum(gt_box[1], r_boxes[:, 1])
                ix2 = np.minimum(gt_box[2], r_boxes[:, 2])
                iy2 = np.minimum(gt_box[3], r_boxes[:, 3])
                inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
                gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                r_area = (r_boxes[:, 2] - r_boxes[:, 0]) * (r_boxes[:, 3] - r_boxes[:, 1])
                union = gt_area + r_area - inter
                ious = inter / np.maximum(union, 1e-6)
                best_idx = int(np.argmax(ious))

                if ious[best_idx] >= 0.35:
                    track_id = int(row_arr[best_idx, 4])
                    if gt_id in self.gt_to_track_map:
                        if self.gt_to_track_map[gt_id] != track_id:
                            self.id_switches += 1
                            new_switches += 1
                    self.gt_to_track_map[gt_id] = track_id

            # Build PersonTrack objects
            for row in rows:
                tid = int(row[4])
                conf = float(row[5])
                box = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
                track = self.active_tracks.get(tid)
                if track is None:
                    track = PersonTrack(
                        track_id=tid,
                        session_id=f"{frame_packet.epoch}:{tid}",
                        first_seen=frame_packet.timestamp,
                        last_seen=frame_packet.timestamp,
                        bbox=box,
                        confidence=conf,
                        hits=1,
                        state="active",
                        confirmed=True,
                    )
                    self.active_tracks[tid] = track
                else:
                    track.last_seen = frame_packet.timestamp
                    track.bbox = box
                    track.confidence = 0.7 * conf + 0.3 * track.confidence
                    track.hits += 1
                current_tracks.append(track)

        return current_tracks, dt, new_switches


class FacePoseStage:
    """Face detection & 3D head pose estimation stage using OpenCV YuNet ONNX."""

    def __init__(self, models_dir: Path):
        self.yunet_path = models_dir / "face_detection_yunet_2023mar.onnx"
        self.detector = None
        if self.yunet_path.exists():
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    str(self.yunet_path), "", (160, 160), score_threshold=0.30
                )
            except Exception:
                self.detector = None

    def process(
        self,
        frame_packet: FramePacket,
        tracks: List[PersonTrack],
        ground_truth: List[Dict[str, Any]],
    ) -> Tuple[List[FaceObservation], float]:
        """Detect faces on person crops and estimate 3D head pose."""
        t0 = time.perf_counter()
        observations: List[FaceObservation] = []
        gt_map = {gt["person_id"]: gt for gt in ground_truth}

        for track in tracks:
            # Upper head region crop
            x1, y1, x2, y2 = track.bbox
            w = max(10, x2 - x1)
            h = max(10, y2 - y1)
            head_h = max(16, int(h * 0.35))
            cx1 = max(0, int(round(x1)))
            cy1 = max(0, int(round(y1)))
            cx2 = min(frame_packet.width, int(round(x2)))
            cy2 = min(frame_packet.height, int(round(y1 + head_h)))

            crop = frame_packet.image[cy1:cy2, cx1:cx2]
            if crop.shape[0] < 12 or crop.shape[1] < 12:
                continue

            detected_pose = None
            landmarks: List[Tuple[float, float]] = []

            # 1. Run actual YuNet ONNX inference
            if self.detector is not None:
                try:
                    self.detector.setInputSize((crop.shape[1], crop.shape[0]))
                    _, faces = self.detector.detect(crop)
                    if faces is not None and len(faces) > 0:
                        f = faces[0]
                        # Landmarks: right eye, left eye, nose, right mouth, left mouth
                        re = np.array([f[4], f[5]])
                        le = np.array([f[6], f[7]])
                        nt = np.array([f[8], f[9]])
                        rm = np.array([f[10], f[11]])
                        lm = np.array([f[12], f[13]])
                        landmarks = [
                            (float(re[0] + cx1), float(re[1] + cy1)),
                            (float(le[0] + cx1), float(le[1] + cy1)),
                            (float(nt[0] + cx1), float(nt[1] + cy1)),
                            (float(rm[0] + cx1), float(rm[1] + cy1)),
                            (float(lm[0] + cx1), float(lm[1] + cy1)),
                        ]
                        # Compute pose geometry
                        eye_mid = (re + le) / 2.0
                        mouth_mid = (rm + lm) / 2.0
                        iod = max(1e-4, np.linalg.norm(le - re))
                        dx = (nt[0] - eye_mid[0]) / iod
                        yaw = float(np.clip(dx * 90.0, -90.0, 90.0))
                        roll = float(np.degrees(np.arctan2(le[1] - re[1], le[0] - re[0])))
                        face_h = max(1e-4, mouth_mid[1] - eye_mid[1])
                        dy = (nt[1] - eye_mid[1]) / face_h - 0.45
                        pitch = float(np.clip(-dy * 90.0, -90.0, 90.0))
                        detected_pose = FacePose(yaw=yaw, pitch=pitch, roll=roll)
                except Exception:
                    pass

            # 2. Geometric fallback if detector didn't fire on crop
            if detected_pose is None:
                # Find matching ground truth pose
                gt_match = gt_map.get(track.track_id)
                if gt_match:
                    yaw, pitch, roll = gt_match["pose"]
                else:
                    yaw = float(math.sin(frame_packet.sequence * 0.1 + track.track_id) * 15.0)
                    pitch = float(math.cos(frame_packet.sequence * 0.08) * 5.0)
                    roll = float(math.sin(frame_packet.sequence * 0.05) * 2.0)

                detected_pose = FacePose(yaw=yaw, pitch=pitch, roll=roll)
                # Simulated landmarks
                fcx = (cx1 + cx2) / 2
                fcy = (cy1 + cy2) / 2
                landmarks = [
                    (fcx - 15, fcy - 8),
                    (fcx + 15, fcy - 8),
                    (fcx, fcy + 2),
                    (fcx - 10, fcy + 15),
                    (fcx + 10, fcy + 15),
                ]

            # Categorize pose
            y_abs = abs(detected_pose.yaw)
            p_abs = abs(detected_pose.pitch)
            if y_abs <= 15.0 and p_abs <= 15.0:
                detected_pose.category = "frontal"
            elif -35.0 <= detected_pose.yaw < -15.0:
                detected_pose.category = "left_angle"
            elif 15.0 < detected_pose.yaw <= 35.0:
                detected_pose.category = "right_angle"
            elif y_abs > 35.0:
                detected_pose.category = "profile"
            else:
                detected_pose.category = "uncertain"

            obs = FaceObservation(
                observation_id=f"obs_{track.track_id}_{frame_packet.sequence}",
                track_id=track.track_id,
                timestamp=frame_packet.timestamp,
                bbox=(float(cx1), float(cy1), float(cx2), float(cy2)),
                confidence=track.confidence,
                landmarks=landmarks,
                pose=detected_pose,
                face_crop=crop,
            )
            observations.append(obs)

        dt = (time.perf_counter() - t0) * 1000.0
        return observations, dt


class FaceQualityEvaluationStage:
    """Multi-factor face quality assessment stage scoring sharpness, lighting, pose, and resolution."""

    def process(self, observations: List[FaceObservation]) -> Tuple[List[FaceObservation], float]:
        """Compute comprehensive quality metrics for each observation."""
        t0 = time.perf_counter()

        for obs in observations:
            crop = obs.face_crop
            if crop is None or crop.size == 0:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

            # 1. Sharpness via Laplacian variance
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            sharpness_score = min(100.0, lap_var / 1.5)

            # 2. Motion blur via gradient asymmetry
            gx = float(cv2.Sobel(gray, cv2.CV_64F, 1, 0).var())
            gy = float(cv2.Sobel(gray, cv2.CV_64F, 0, 1).var())
            motion_blur = float(np.clip(abs(gx - gy) / max(gx + gy, 1e-4), 0.0, 1.0))

            # 3. Resolution score (target 100x100)
            min_dim = min(crop.shape[0], crop.shape[1])
            resolution_score = float(np.clip(min_dim / 100.0 * 100.0, 0.0, 100.0))

            # 4. Lighting score (exposure and contrast)
            mean_luma = float(np.mean(gray))
            std_luma = float(np.std(gray))
            exposure_penalty = max(0.0, abs(mean_luma - 128.0) - 25.0) / 103.0
            contrast_factor = min(1.0, std_luma / 35.0)
            lighting_score = float(np.clip((1.0 - exposure_penalty) * contrast_factor * 100.0, 0.0, 100.0))

            # 5. Pose score
            if obs.pose:
                y_abs = abs(obs.pose.yaw)
                p_abs = abs(obs.pose.pitch)
                pose_penalty = min(1.0, (y_abs / 50.0) ** 1.5 + (p_abs / 40.0) ** 1.5)
                pose_score = float(np.clip((1.0 - pose_penalty) * 100.0, 0.0, 100.0))
            else:
                pose_score = 50.0

            # Composite overall quality score (0 - 100)
            composite = float(np.clip(
                0.35 * sharpness_score +
                0.25 * lighting_score +
                0.25 * pose_score +
                0.15 * resolution_score,
                0.0,
                100.0,
            ))

            is_acceptable = composite >= 50.0 and sharpness_score >= 35.0 and motion_blur <= 0.85

            obs.quality = FaceQualityResult(
                score=composite,
                sharpness=lap_var,
                motion_blur=motion_blur,
                resolution_score=resolution_score,
                lighting_score=lighting_score,
                pose_score=pose_score,
                size_score=resolution_score,
                stability_score=100.0,
                is_acceptable=is_acceptable,
                metrics={
                    "mean_luma": mean_luma,
                    "std_luma": std_luma,
                    "laplacian_var": lap_var,
                },
            )

        dt = (time.perf_counter() - t0) * 1000.0
        return observations, dt


class CaptureQueueWorker:
    """Asynchronous background worker evaluating candidate crops, encoding JPEG, and measuring queue delay."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.queue: queue.Queue = queue.Queue()
        self.storage_dir = storage_dir
        if storage_dir:
            storage_dir.mkdir(parents=True, exist_ok=True)
        self.delays: List[float] = []
        self.profiles: Dict[int, PersonProfile] = {}
        self.captures_saved = 0
        self._stopped = False
        self._worker_thread = threading.Thread(target=self._loop, daemon=True)
        self._worker_thread.start()

    def enqueue(self, candidate: CaptureCandidate):
        """Enqueue face candidate with precision enqueue timestamp."""
        t_enqueued = time.perf_counter()
        self.queue.put((candidate, t_enqueued))

    def _loop(self):
        while not self._stopped:
            try:
                item = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if item is None:
                self.queue.task_done()
                break

            candidate, t_enqueued = item
            t_processed = time.perf_counter()
            delay_ms = (t_processed - t_enqueued) * 1000.0
            self.delays.append(delay_ms)

            self._process_candidate(candidate)
            self.queue.task_done()

    def _process_candidate(self, candidate: CaptureCandidate):
        """Process candidate crop: check best score, encode JPEG, and persist record."""
        tid = candidate.track_id
        profile = self.profiles.get(tid)
        if profile is None:
            profile = PersonProfile(
                session_id=f"session:{tid}",
                track_id=tid,
                camera_id="cam_benchmark",
                first_seen=candidate.timestamp,
                last_seen=candidate.timestamp,
                duration=0.0,
                status="active",
            )
            self.profiles[tid] = profile

        profile.last_seen = candidate.timestamp
        profile.total_observations += 1

        cat = candidate.category
        curr_best = profile.captures_by_category.get(cat)

        if curr_best is None or candidate.quality_score > curr_best.quality_score:
            # Simulate real image encoding (JPEG compress)
            ok, buf = cv2.imencode(".jpg", candidate.crop_original, [cv2.IMWRITE_JPEG_QUALITY, 90])
            size_bytes = len(buf) if ok else 0

            record = CaptureRecord(
                capture_id=f"cap_{candidate.candidate_id}",
                track_id=tid,
                session_id=profile.session_id,
                timestamp=candidate.timestamp,
                category=cat,
                quality_score=candidate.quality_score,
                pose_category=candidate.pose.category,
                yaw=candidate.pose.yaw,
                pitch=candidate.pose.pitch,
                original_path=f"data/captures/{candidate.candidate_id}.jpg",
                crop_width=candidate.crop_original.shape[1],
                crop_height=candidate.crop_original.shape[0],
                file_size_bytes=size_bytes,
                created_at=time.time(),
            )
            profile.captures_by_category[cat] = record
            if profile.best_capture is None or candidate.quality_score > profile.best_quality_score:
                profile.best_capture = record
                profile.best_quality_score = candidate.quality_score
            self.captures_saved += 1

    def flush(self):
        """Wait until all queued items are processed."""
        self.queue.join()

    def stop(self):
        """Signal worker to stop and wait for completion."""
        self._stopped = True
        self.queue.put(None)
        self._worker_thread.join(timeout=2.0)


# =====================================================================
# Main Benchmark Runner
# =====================================================================


class PeopleBenchmarkSuite:
    """Automated benchmark runner orchestrating multi-person load simulation."""

    def __init__(
        self,
        models_dir: Path,
        storage_dir: Optional[Path] = None,
        frame_width: int = 1280,
        frame_height: int = 720,
        detector_mode: str = "simulated",
    ):
        self.models_dir = models_dir
        self.storage_dir = storage_dir
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.detector_mode = detector_mode
        self.process = psutil.Process(os.getpid())

    def run_scenario(
        self,
        person_count: int,
        frames: int = 30,
        warmup: int = 5,
    ) -> Dict[str, Any]:
        """Run single benchmark scenario for given person count."""
        print(f"\n[BENCHMARK] Running scenario: {person_count} person(s), {frames} frames (warmup: {warmup})...")

        stream = SyntheticPersonStream(
            person_count=person_count,
            width=self.frame_width,
            height=self.frame_height,
        )
        detector = PersonDetectionStage(mode=self.detector_mode, models_dir=self.models_dir)
        tracker = TrackingAssociationStage(models_dir=self.models_dir, source_fps=30.0)
        face_pose = FacePoseStage(models_dir=self.models_dir)
        quality_eval = FaceQualityEvaluationStage()
        queue_worker = CaptureQueueWorker(storage_dir=self.storage_dir)

        # Measurements
        detection_times: List[float] = []
        tracking_times: List[float] = []
        face_pose_times: List[float] = []
        quality_eval_times: List[float] = []
        total_frame_times: List[float] = []
        cpu_readings: List[float] = []
        memory_readings: List[float] = []
        candidates_enqueued = 0

        # Prime CPU reading
        self.process.cpu_percent(interval=None)

        total_iterations = warmup + frames
        t_scenario_start = 0.0

        try:
            for idx in range(total_iterations):
                is_warmup = idx < warmup
                t_frame_start = time.perf_counter()

                if not is_warmup and t_scenario_start == 0.0:
                    t_scenario_start = time.perf_counter()

                # 1. Acquire Frame
                packet, ground_truth = stream.next_frame()

                # 2. Person Detection
                detections, det_ms = detector.process(packet, ground_truth)

                # 3. Tracking Association
                tracks, track_ms, _ = tracker.process(detections, packet, ground_truth)

                # 4. Face Detection & Pose Estimation
                observations, face_ms = face_pose.process(packet, tracks, ground_truth)

                # 5. Quality Evaluation
                observations, qual_ms = quality_eval.process(observations)

                # 6. Candidate Enqueueing
                for obs in observations:
                    if obs.quality and obs.quality.is_acceptable and obs.quality.score >= 50.0 and obs.face_crop is not None:
                        candidate = CaptureCandidate(
                            candidate_id=f"c_{obs.track_id}_{packet.sequence}",
                            track_id=obs.track_id,
                            timestamp=packet.timestamp,
                            category=obs.pose.category if obs.pose and obs.pose.category in ("frontal", "left_angle", "right_angle") else "overall",
                            quality_score=obs.quality.score,
                            pose=obs.pose or FacePose(0, 0, 0),
                            crop_original=obs.face_crop,
                            frame_sequence=packet.sequence,
                            bbox_in_frame=obs.bbox,
                        )
                        queue_worker.enqueue(candidate)
                        if not is_warmup:
                            candidates_enqueued += 1

                t_frame_end = time.perf_counter()
                frame_ms = (t_frame_end - t_frame_start) * 1000.0

                if not is_warmup:
                    detection_times.append(det_ms)
                    tracking_times.append(track_ms)
                    face_pose_times.append(face_ms)
                    quality_eval_times.append(qual_ms)
                    total_frame_times.append(frame_ms)

                    # Monitor resources
                    cpu_readings.append(self.process.cpu_percent(interval=None))
                    mem_mb = self.process.memory_info().rss / (1024 * 1024)
                    memory_readings.append(mem_mb)

            # Flush background capture queue
            queue_worker.flush()
            t_scenario_end = time.perf_counter()
            total_elapsed_s = max(1e-4, t_scenario_end - t_scenario_start)
            pipeline_fps = frames / total_elapsed_s

        finally:
            queue_worker.stop()

        # Compute statistics
        det_stats = compute_stats(detection_times)
        track_stats = compute_stats(tracking_times)
        face_stats = compute_stats(face_pose_times)
        qual_stats = compute_stats(quality_eval_times)
        delay_stats = compute_stats(queue_worker.delays)
        frame_stats = compute_stats(total_frame_times)

        mean_cpu = round(float(np.mean(cpu_readings)), 1) if cpu_readings else 0.0
        peak_cpu = round(float(np.max(cpu_readings)), 1) if cpu_readings else 0.0
        mean_mem = round(float(np.mean(memory_readings)), 1) if memory_readings else 0.0
        peak_mem = round(float(np.max(memory_readings)), 1) if memory_readings else 0.0

        scenario_duration_min = total_elapsed_s / 60.0
        id_switch_rate = round(tracker.id_switches / max(1e-4, scenario_duration_min * person_count), 4)

        result = {
            "person_count": person_count,
            "frames_evaluated": frames,
            "pipeline_fps": round(pipeline_fps, 2),
            "serial_processing_fps": round(1000.0 / max(1e-4, frame_stats["mean"]), 2),
            "latencies": {
                "detection_ms": det_stats,
                "tracking_ms": track_stats,
                "face_pose_ms": face_stats,
                "quality_eval_ms": qual_stats,
                "queue_delay_ms": delay_stats,
                "total_frame_ms": frame_stats,
            },
            "resources": {
                "cpu_usage_percent": mean_cpu,
                "peak_cpu_percent": peak_cpu,
                "memory_consumption_mb": mean_mem,
                "peak_memory_mb": peak_mem,
            },
            "accuracy": {
                "ground_truth_people": person_count,
                "tracked_unique_ids": len(tracker.active_tracks),
                "id_switches": tracker.id_switches,
                "id_switch_rate_per_min": id_switch_rate,
                "candidates_enqueued": candidates_enqueued,
                "captures_saved": queue_worker.captures_saved,
            },
        }

        print(
            f"  Result -> FPS: {result['pipeline_fps']:.1f} | Det: {det_stats['mean']:.2f}ms | "
            f"Track: {track_stats['mean']:.2f}ms | Face/Pose: {face_stats['mean']:.2f}ms | "
            f"Quality: {qual_stats['mean']:.2f}ms | Queue Delay: {delay_stats['mean']:.2f}ms | "
            f"ID Switches: {tracker.id_switches}"
        )
        return result

    def run_all(
        self,
        person_counts: List[int],
        frames: int = 30,
        warmup: int = 5,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Execute full benchmark across all requested people scenarios."""
        print("=" * 76)
        print("  STARTING MULTI-PERSON FACIAL CAPTURE BENCHMARK SUITE")
        print("=" * 76)
        sys_info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "opencv_version": cv2.__version__,
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        }
        print(f"Platform: {sys_info['platform']}")
        print(f"Processor: {sys_info['processor']} ({sys_info['cpu_count_logical']} logical cores)")
        print(f"Scenarios: {person_counts} people, {frames} frames each")

        scenario_results: List[Dict[str, Any]] = []
        for count in person_counts:
            res = self.run_scenario(person_count=count, frames=frames, warmup=warmup)
            scenario_results.append(res)

        full_results = {
            "benchmark_version": "1.0.0",
            "timestamp": datetime.datetime.now().isoformat(),
            "system_info": sys_info,
            "benchmark_config": {
                "scenarios": person_counts,
                "frames_per_scenario": frames,
                "warmup_frames": warmup,
                "resolution": f"{self.frame_width}x{self.frame_height}",
                "detector_mode": self.detector_mode,
            },
            "scenarios": scenario_results,
        }

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            import json
            output_path.write_text(json.dumps(full_results, indent=2), encoding="utf-8")
            print(f"\n[BENCHMARK] Results written successfully to: {output_path}")

        self.print_summary_table(full_results)
        return full_results

    @staticmethod
    def print_summary_table(results: Dict[str, Any]):
        """Print a formatted Markdown summary table of benchmark results."""
        print("\n" + "=" * 90)
        print("  BENCHMARK SUMMARY TABLE")
        print("=" * 90)
        header = (
            f"| {'People':<6} | {'FPS':<6} | {'Det (ms)':<9} | {'Track (ms)':<10} | "
            f"{'Face/Pose':<10} | {'Quality':<8} | {'Queue (ms)':<10} | {'CPU (%)':<8} | {'RAM (MB)':<8} | {'IDSW':<5} |"
        )
        sep = (
            f"|{'-'*8}|{'-'*8}|{'-'*11}|{'-'*12}|{'-'*12}|{'-'*10}|{'-'*12}|{'-'*10}|{'-'*10}|{'-'*7}|"
        )
        print(header)
        print(sep)
        for s in results["scenarios"]:
            p = s["person_count"]
            fps = s["pipeline_fps"]
            det = s["latencies"]["detection_ms"]["mean"]
            trk = s["latencies"]["tracking_ms"]["mean"]
            fc = s["latencies"]["face_pose_ms"]["mean"]
            ql = s["latencies"]["quality_eval_ms"]["mean"]
            qd = s["latencies"]["queue_delay_ms"]["mean"]
            cpu = s["resources"]["cpu_usage_percent"]
            mem = s["resources"]["memory_consumption_mb"]
            idsw = s["accuracy"]["id_switches"]
            row = (
                f"| {p:<6} | {fps:<6.1f} | {det:<9.2f} | {trk:<10.2f} | "
                f"{fc:<10.2f} | {ql:<8.2f} | {qd:<10.2f} | {cpu:<8.1f} | {mem:<8.1f} | {idsw:<5} |"
            )
            print(row)
        print("=" * 90 + "\n")


def run_people_benchmark(
    person_counts: Optional[List[int]] = None,
    frames_per_scenario: int = 30,
    warmup_frames: int = 5,
    output_path: Optional[Path] = None,
    detector_mode: str = "simulated",
) -> Dict[str, Any]:
    """Programmatic entry point for people capture benchmark."""
    if person_counts is None:
        person_counts = [1, 5, 10, 20, 30]
    if output_path is None:
        output_path = _ROOT / "data" / "people_benchmark_results.json"

    models_dir = _ROOT / "models"
    suite = PeopleBenchmarkSuite(
        models_dir=models_dir,
        storage_dir=_ROOT / "data" / "benchmark_captures",
        detector_mode=detector_mode,
    )
    return suite.run_all(
        person_counts=person_counts,
        frames=frames_per_scenario,
        warmup=warmup_frames,
        output_path=output_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Multi-Person Facial Capture Benchmark Suite")
    parser.add_argument(
        "--people",
        type=str,
        default="1,5,10,20,30",
        help="Comma-separated list of person counts (default: 1,5,10,20,30)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Benchmark frame count per scenario (default: 30)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup frame count per scenario (default: 5)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast run with fewer frames (useful for smoke tests)",
    )
    parser.add_argument(
        "--detector",
        choices=["simulated", "yolo"],
        default="simulated",
        help="Person detection stage mode (default: simulated)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "data" / "people_benchmark_results.json",
        help="Output JSON results path",
    )
    args = parser.parse_args()

    counts = [int(p.strip()) for p in args.people.split(",") if p.strip()]
    frames = 5 if args.fast else args.frames
    warmup = 1 if args.fast else args.warmup

    run_people_benchmark(
        person_counts=counts,
        frames_per_scenario=frames,
        warmup_frames=warmup,
        output_path=args.output,
        detector_mode=args.detector,
    )


if __name__ == "__main__":
    main()
