"""Unit tests for PeopleSystemOrchestrator."""
import time
from pathlib import Path
import numpy as np
import pytest

from services.face_contracts import (
    AnalysisState,
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
from services.people_orchestrator import PeopleSystemOrchestrator
from services.profile_repository import ProfileRepository
from services.people_events import PeopleEventEmitter
from services.video_pipeline import VideoPipeline


def create_synthetic_frame(width=640, height=480, sequence=1, timestamp=100.0):
    """Generate a valid BGR test image with synthetic person and face."""
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    # Draw simple body and head
    cv2_person = (int(width * 0.2), int(height * 0.2), int(width * 0.5), int(height * 0.8))
    # Fill person area with slightly darker color
    img[cv2_person[1]:cv2_person[3], cv2_person[0]:cv2_person[2]] = 90
    # Fill face area
    face_y1, face_y2 = cv2_person[1], cv2_person[1] + int((cv2_person[3] - cv2_person[1]) * 0.3)
    img[face_y1:face_y2, cv2_person[0]:cv2_person[2]] = 180

    return FramePacket(
        sequence=sequence,
        timestamp=timestamp,
        image=img,
        source_id="test_cam",
        epoch="test_epoch_1",
        width=width,
        height=height,
    )


def test_orchestrator_initialization(tmp_path: Path):
    """Verify clean initialization of all subsystems and defaults."""
    db_path = tmp_path / "people_test.sqlite3"
    storage_dir = tmp_path / "captures"
    storage_dir.mkdir(parents=True, exist_ok=True)

    repo = ProfileRepository(db_path=db_path, storage_dir=storage_dir)
    emitter = PeopleEventEmitter()

    orch = PeopleSystemOrchestrator(
        profile_repository=repo,
        people_event_emitter=emitter,
        data_dir=tmp_path,
        auto_start=False,
    )

    assert orch.is_running() is False
    assert orch.repository is repo
    assert orch.face_detector is not None
    assert orch.person_detector is not None
    assert orch.person_tracker is not None
    assert orch.face_pose_estimator is not None
    assert orch.face_quality_evaluator is not None
    assert orch.capture_selector is not None
    assert orch.face_enhancer is not None
    assert orch.pipeline_scheduler is not None
    assert orch.privacy_governance is not None

    state = orch.get_state()
    assert isinstance(state, AnalysisState)
    assert state.is_running is False

    metrics = orch.get_metrics()
    assert isinstance(metrics, ProcessingMetrics)


def test_orchestrator_lifecycle_transitions(tmp_path: Path):
    """Verify start, stop, and idempotent state transitions."""
    orch = PeopleSystemOrchestrator(
        data_dir=tmp_path,
        auto_start=False,
    )

    assert orch.is_running() is False
    started = orch.start(epoch_id="session_test_01")
    assert started is True
    assert orch.is_running() is True
    assert "session_test_01" in orch.current_epoch

    # Idempotent start
    assert orch.start() is True

    # Stop
    stopped = orch.stop(drain_timeout=1.0)
    assert stopped is True
    assert orch.is_running() is False

    # Idempotent stop
    assert orch.stop() is True


def test_orchestrator_configuration_updates(tmp_path: Path):
    """Verify dynamic configuration get and set."""
    orch = PeopleSystemOrchestrator(
        data_dir=tmp_path,
        auto_start=False,
    )

    cfg = orch.get_config()
    assert "person_confidence_threshold" in cfg

    updated = orch.configure({
        "person_confidence_threshold": 0.65,
        "quality_threshold": 55.0,
        "enhancement_scale": 4,
    })

    assert updated["person_confidence_threshold"] == 0.65
    assert updated["quality_threshold"] == 55.0
    assert updated["enhancement_scale"] == 4
    assert orch.person_detector.config.confidence_threshold == 0.65

    with pytest.raises(ValueError):
        orch.configure("invalid_string_not_dict")  # type: ignore


def test_orchestrator_process_frame_end_to_end(tmp_path: Path):
    """Verify process_frame executes full pipeline stages and updates snapshot."""
    db_path = tmp_path / "people_e2e.sqlite3"
    storage_dir = tmp_path / "captures_e2e"
    storage_dir.mkdir(parents=True, exist_ok=True)

    repo = ProfileRepository(db_path=db_path, storage_dir=storage_dir)
    emitter = PeopleEventEmitter()

    emitted_frames = []
    emitter.add_frame_callback = lambda cb: None  # mock
    orch = PeopleSystemOrchestrator(
        profile_repository=repo,
        people_event_emitter=emitter,
        data_dir=tmp_path,
        auto_start=False,
    )
    orch.start(epoch_id="e2e_epoch")

    packet = create_synthetic_frame(sequence=1, timestamp=10.0)
    result = orch.process_frame(packet)

    assert "sequence" in result
    assert result["sequence"] == 1
    assert "timestamp" in result
    assert "tracks" in result
    assert "faces" in result
    assert "total_latency_ms" in result

    # Check snapshot
    snap = orch.snapshot
    assert snap["sequence"] == 1
    assert snap["timestamp"] == 10.0

    metrics = orch.get_metrics()
    assert metrics.frame_sequence == 1
    assert metrics.acquisition_fps >= 0.0

    orch.stop()


def test_orchestrator_handles_empty_and_corrupt_frames(tmp_path: Path):
    """Verify graceful error recovery when invalid frame inputs are passed."""
    orch = PeopleSystemOrchestrator(
        data_dir=tmp_path,
        auto_start=False,
    )
    orch.start()

    # 1. None packet
    res_none = orch.process_frame(None)
    assert "error" in res_none

    # 2. Raw numpy array
    raw_arr = np.zeros((100, 100, 3), dtype=np.uint8)
    res_raw = orch.process_frame(raw_arr)
    assert res_raw.get("status") != "error"

    # 3. Stopped state
    orch.stop()
    res_stopped = orch.process_frame(create_synthetic_frame(sequence=2))
    assert res_stopped["status"] == "stopped"
