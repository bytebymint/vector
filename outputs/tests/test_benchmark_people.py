"""Unit tests verifying the Multi-Person Facial Capture Benchmark Suite and Load Simulator."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure outputs directory is in sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark_people import (  # noqa: E402
    CaptureQueueWorker,
    FaceCropPool,
    FacePoseStage,
    FaceQualityEvaluationStage,
    PersonDetectionStage,
    RealisticPerson,
    SyntheticPersonStream,
    TrackingAssociationStage,
    compute_stats,
    generate_procedural_face,
    run_people_benchmark,
)
from services.face_contracts import (  # noqa: E402
    CaptureCandidate,
    FaceObservation,
    FacePose,
    FramePacket,
    PersonDetection,
    PersonTrack,
)


def test_compute_stats_empty_and_valid():
    """Verify statistics calculation handles both empty and valid inputs."""
    empty_stats = compute_stats([])
    assert empty_stats["mean"] == 0.0
    assert empty_stats["p95"] == 0.0

    samples = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = compute_stats(samples)
    assert stats["mean"] == 30.0
    assert stats["median"] == 30.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["p95"] >= 45.0


def test_generate_procedural_face():
    """Verify procedural face generation creates valid images with varied poses."""
    face = generate_procedural_face(
        width=96,
        height=96,
        skin_tone=(150, 190, 225),
        yaw=15.0,
        pitch=-5.0,
        roll=2.0,
    )
    assert isinstance(face, np.ndarray)
    assert face.shape == (96, 96, 3)
    assert face.dtype == np.uint8
    # Image should not be blank
    assert float(np.mean(face)) > 30.0
    assert float(np.std(face)) > 15.0


def test_face_crop_pool():
    """Verify face crop pool initializes and returns face images."""
    pool = FaceCropPool(_ROOT, target_size=(80, 80))
    assert len(pool.crops) >= 30

    face0 = pool.get_face(0)
    face5 = pool.get_face(5)
    assert face0.shape == (80, 80, 3)
    assert face5.shape == (80, 80, 3)


def test_synthetic_person_stream_dimensions_and_packets():
    """Verify synthetic stream produces valid FramePackets and moving person ground truth."""
    stream = SyntheticPersonStream(person_count=5, width=640, height=480)
    packet, gt = stream.next_frame()

    assert isinstance(packet, FramePacket)
    assert packet.width == 640
    assert packet.height == 480
    assert packet.image.shape == (480, 640, 3)
    assert packet.sequence == 1
    assert len(gt) == 5

    for person_info in gt:
        assert "person_id" in person_info
        assert len(person_info["person_bbox"]) == 4
        assert len(person_info["face_bbox"]) == 4
        assert len(person_info["pose"]) == 3
        # Check coordinates within reasonable range
        x1, y1, x2, y2 = person_info["person_bbox"]
        assert x2 > x1 and y2 > y1

    # Second frame advances trajectory
    packet2, gt2 = stream.next_frame()
    assert packet2.sequence == 2


def test_person_detection_stage():
    """Verify person detection stage accurately produces PersonDetection contracts."""
    stream = SyntheticPersonStream(person_count=3, width=640, height=480)
    packet, gt = stream.next_frame()

    stage = PersonDetectionStage(mode="simulated")
    detections, det_ms = stage.process(packet, gt)

    assert isinstance(detections, list)
    assert len(detections) == 3
    assert det_ms >= 0.0
    for det in detections:
        assert isinstance(det, PersonDetection)
        assert len(det.bbox) == 4
        assert 0.0 <= det.confidence <= 1.0
        assert det.class_id == 0


def test_tracking_association_stage_continuity():
    """Verify tracking association preserves IDs across successive frames without ID switches."""
    models_dir = _ROOT / "models"
    tracker_stage = TrackingAssociationStage(models_dir=models_dir, source_fps=30.0)
    stream = SyntheticPersonStream(person_count=3, width=640, height=480)
    det_stage = PersonDetectionStage(mode="simulated")

    # Run for 10 frames
    for _ in range(10):
        packet, gt = stream.next_frame()
        detections, _ = det_stage.process(packet, gt)
        tracks, track_ms, idsw = tracker_stage.process(detections, packet, gt)
        assert track_ms >= 0.0
        assert isinstance(tracks, list)

    assert len(tracker_stage.active_tracks) >= 3
    # ID switches should be zero or near zero on clean non-colliding paths
    assert tracker_stage.id_switches == 0


def test_face_pose_stage():
    """Verify face detection and 3D head pose estimation generates valid FaceObservations."""
    models_dir = _ROOT / "models"
    stream = SyntheticPersonStream(person_count=2, width=640, height=480)
    packet, gt = stream.next_frame()

    det_stage = PersonDetectionStage(mode="simulated")
    tracker_stage = TrackingAssociationStage(models_dir=models_dir)
    face_stage = FacePoseStage(models_dir=models_dir)

    detections, _ = det_stage.process(packet, gt)
    tracks, _, _ = tracker_stage.process(detections, packet, gt)
    observations, face_ms = face_stage.process(packet, tracks, gt)

    assert isinstance(observations, list)
    assert len(observations) == len(tracks)
    assert face_ms >= 0.0

    for obs in observations:
        assert isinstance(obs, FaceObservation)
        assert obs.pose is not None
        assert isinstance(obs.pose, FacePose)
        assert obs.pose.category in ("frontal", "left_angle", "right_angle", "profile", "uncertain")
        assert len(obs.landmarks) == 5
        assert obs.face_crop is not None


def test_quality_eval_stage():
    """Verify multi-factor face quality assessment scores within expected [0, 100] bounds."""
    models_dir = _ROOT / "models"
    stream = SyntheticPersonStream(person_count=2, width=640, height=480)
    packet, gt = stream.next_frame()

    det_stage = PersonDetectionStage(mode="simulated")
    tracker_stage = TrackingAssociationStage(models_dir=models_dir)
    face_stage = FacePoseStage(models_dir=models_dir)
    quality_stage = FaceQualityEvaluationStage()

    detections, _ = det_stage.process(packet, gt)
    tracks, _, _ = tracker_stage.process(detections, packet, gt)
    observations, _ = face_stage.process(packet, tracks, gt)
    observations, qual_ms = quality_stage.process(observations)

    assert qual_ms >= 0.0
    for obs in observations:
        assert obs.quality is not None
        assert 0.0 <= obs.quality.score <= 100.0
        assert obs.quality.sharpness >= 0.0
        assert 0.0 <= obs.quality.motion_blur <= 1.0
        assert 0.0 <= obs.quality.resolution_score <= 100.0
        assert 0.0 <= obs.quality.lighting_score <= 100.0
        assert isinstance(obs.quality.is_acceptable, bool)


def test_capture_queue_worker():
    """Verify asynchronous capture queue processes candidates and records processing delay."""
    worker = CaptureQueueWorker()
    crop = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
    pose = FacePose(yaw=2.0, pitch=1.0, roll=0.0, category="frontal")

    candidate = CaptureCandidate(
        candidate_id="cand_test_01",
        track_id=1,
        timestamp=time.time(),
        category="frontal",
        quality_score=85.0,
        pose=pose,
        crop_original=crop,
        frame_sequence=1,
        bbox_in_frame=(10.0, 10.0, 74.0, 74.0),
    )

    try:
        worker.enqueue(candidate)
        worker.flush()

        assert len(worker.delays) >= 1
        assert worker.delays[0] >= 0.0
        assert worker.captures_saved >= 1
        assert 1 in worker.profiles
        profile = worker.profiles[1]
        assert profile.best_capture is not None
        assert profile.best_quality_score == 85.0
    finally:
        worker.stop()


def test_run_people_benchmark_smoke(tmp_path):
    """Verify end-to-end benchmark execution across small scenarios produces valid JSON output."""
    output_file = tmp_path / "test_benchmark_results.json"
    results = run_people_benchmark(
        person_counts=[1, 3],
        frames_per_scenario=4,
        warmup_frames=1,
        output_path=output_file,
        detector_mode="simulated",
    )

    assert "benchmark_version" in results
    assert "system_info" in results
    assert "scenarios" in results
    assert len(results["scenarios"]) == 2

    # Check required metrics for each scenario
    for sc in results["scenarios"]:
        assert "pipeline_fps" in sc
        assert sc["pipeline_fps"] > 0.0
        assert "latencies" in sc
        lat = sc["latencies"]
        for key in ("detection_ms", "tracking_ms", "face_pose_ms", "quality_eval_ms", "queue_delay_ms", "total_frame_ms"):
            assert key in lat
            assert "mean" in lat[key]
            assert "p95" in lat[key]
            assert lat[key]["mean"] >= 0.0

        assert "resources" in sc
        res = sc["resources"]
        assert "cpu_usage_percent" in res
        assert "memory_consumption_mb" in res
        assert res["memory_consumption_mb"] > 0.0

        assert "accuracy" in sc
        acc = sc["accuracy"]
        assert "id_switches" in acc
        assert acc["id_switches"] >= 0
        assert acc["ground_truth_people"] in (1, 3)

    # Check file was saved correctly and is valid JSON
    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(data["scenarios"]) == 2


def test_benchmark_cli():
    """Verify CLI interface runs correctly with arguments."""
    script_path = _ROOT / "benchmark_people.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--people", "1",
        "--frames", "2",
        "--warmup", "1",
        "--fast",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_ROOT))
    assert res.returncode == 0
    assert "BENCHMARK SUMMARY TABLE" in res.stdout
