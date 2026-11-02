"""Unit tests for PipelineScheduler.

Verifies:
1. Priority scheduling execution order (Tiers 1-5) and FIFO sequence ordering
2. Round-robin track selection across 30 active people
3. Fast track-skipping for stationary and low-confidence tracks
4. Backpressure dropping under queue saturation (bounded memory)
5. Adaptive frame skipping under high latency budget
6. Thread pool lifecycle, bounded workers, shutdown and cleanup
7. Stale track buffer memory leak prevention
"""
import time
import numpy as np
import pytest

from services.face_contracts import FramePacket, PersonTrack, ProcessingMetrics
from services.pipeline_scheduler import (
    PipelineScheduler,
    PipelineSchedulerConfig,
    TaskPriority,
    PRIORITY_STREAM,
    PRIORITY_TRACKING,
    PRIORITY_FACE,
    PRIORITY_DISK,
    PRIORITY_ENHANCEMENT,
)


def test_priority_scheduling_execution_order():
    """Verify tasks execute strictly in priority order (1 -> 2 -> 3 -> 4 -> 5)."""
    executed_order = []

    # Configure single worker and pause before submitting to build backlog
    scheduler = PipelineScheduler(worker_threads=1, auto_start=True)
    scheduler.pause()

    def make_task(priority_val, label):
        def task():
            executed_order.append((priority_val, label))
            return label
        return task

    # Submit tasks in reverse / scrambled order
    f_enhancement = scheduler.submit_enhancement_task(make_task(5, "enhancement"))
    f_disk = scheduler.submit_disk_task(make_task(4, "disk"))
    f_stream = scheduler.submit_stream_task(make_task(1, "stream"))
    f_face = scheduler.submit_face_task(make_task(3, "face"))
    f_tracking = scheduler.submit_tracking_task(make_task(2, "tracking"))

    # Also submit two tasks of identical priority to verify FIFO sequence ordering
    f_face2 = scheduler.submit_face_task(make_task(3, "face_second"))

    # Resume worker to process backlog
    scheduler.resume()

    # Wait for all tasks to complete
    assert f_stream.result(timeout=2.0) == "stream"
    assert f_tracking.result(timeout=2.0) == "tracking"
    assert f_face.result(timeout=2.0) == "face"
    assert f_face2.result(timeout=2.0) == "face_second"
    assert f_disk.result(timeout=2.0) == "disk"
    assert f_enhancement.result(timeout=2.0) == "enhancement"

    expected = [
        (1, "stream"),
        (2, "tracking"),
        (3, "face"),
        (3, "face_second"),
        (4, "disk"),
        (5, "enhancement"),
    ]
    assert executed_order == expected

    scheduler.shutdown(wait=True)


def test_round_robin_track_selection_30_tracks():
    """Verify fair round-robin evaluation across 30 simultaneous people."""
    scheduler = PipelineScheduler(worker_threads=1, max_faces_per_frame=5, auto_start=False)

    # Generate 30 distinct person tracks
    tracks = [
        PersonTrack(
            track_id=i,
            session_id=f"session:{i}",
            first_seen=100.0,
            last_seen=100.0,
            bbox=(float(i * 10), 50.0, float(i * 10 + 40), 150.0),
            confidence=0.85,
            confirmed=True,
            speed_px_s=12.0  # Moving, not stationary
        )
        for i in range(1, 31)
    ]

    selected_ids_by_batch = []
    # 6 frames of 5 tracks each should cover all 30 unique tracks
    for frame_idx in range(6):
        batch = scheduler.select_tracks_for_face_analysis(
            tracks,
            max_evaluations=5,
            timestamp=100.0 + frame_idx * 0.05
        )
        assert len(batch) == 5
        ids = [t.track_id for t in batch]
        selected_ids_by_batch.append(ids)

    # Verify batches
    assert selected_ids_by_batch[0] == [1, 2, 3, 4, 5]
    assert selected_ids_by_batch[1] == [6, 7, 8, 9, 10]
    assert selected_ids_by_batch[2] == [11, 12, 13, 14, 15]
    assert selected_ids_by_batch[3] == [16, 17, 18, 19, 20]
    assert selected_ids_by_batch[4] == [21, 22, 23, 24, 25]
    assert selected_ids_by_batch[5] == [26, 27, 28, 29, 30]

    # Verify 100% complete coverage across all 30 tracks
    all_evaluated = [tid for batch in selected_ids_by_batch for tid in batch]
    assert len(set(all_evaluated)) == 30

    # 7th frame wraps around to the beginning
    wrap_batch = scheduler.select_tracks_for_face_analysis(
        tracks,
        max_evaluations=5,
        timestamp=100.0 + 6 * 0.05
    )
    assert [t.track_id for t in wrap_batch] == [1, 2, 3, 4, 5]

    scheduler.shutdown(wait=True)


def test_fast_track_skipping_stationary():
    """Verify stationary tracks skip face evaluation until recheck interval passes."""
    scheduler = PipelineScheduler(
        stationary_recheck_interval_s=2.0,
        min_speed_threshold_px_s=5.0,
        stationary_displacement_threshold_px=10.0,
        auto_start=False
    )

    t0 = 100.0
    stationary_track = PersonTrack(
        track_id=1,
        session_id="s1",
        first_seen=t0,
        last_seen=t0,
        bbox=(50.0, 50.0, 100.0, 150.0),
        confidence=0.90,
        confirmed=True,
        speed_px_s=0.0  # Stationary
    )

    # Frame 1: New track must be evaluated initially
    selected = scheduler.select_tracks_for_face_analysis([stationary_track], timestamp=t0)
    assert len(selected) == 1
    assert selected[0].track_id == 1

    # Frame 2 (0.1s later): Track is stationary and recently evaluated -> skipped!
    selected = scheduler.select_tracks_for_face_analysis([stationary_track], timestamp=t0 + 0.1)
    assert len(selected) == 0

    # Frame 3 (2.1s later): Recheck interval elapsed -> re-evaluated
    selected = scheduler.select_tracks_for_face_analysis([stationary_track], timestamp=t0 + 2.1)
    assert len(selected) == 1
    assert selected[0].track_id == 1

    # Frame 4 (2.2s later): Track begins moving (speed > 5 px/s) -> evaluated immediately
    moving_track = PersonTrack(
        track_id=1,
        session_id="s1",
        first_seen=t0,
        last_seen=t0 + 2.2,
        bbox=(80.0, 50.0, 130.0, 150.0),  # Moved 30px
        confidence=0.90,
        confirmed=True,
        speed_px_s=15.0
    )
    selected = scheduler.select_tracks_for_face_analysis([moving_track], timestamp=t0 + 2.2)
    assert len(selected) == 1
    assert selected[0].track_id == 1

    scheduler.shutdown(wait=True)


def test_fast_track_skipping_low_confidence_and_state():
    """Verify tracks with low confidence or inactive states are filtered out."""
    scheduler = PipelineScheduler(
        min_track_confidence=0.40,
        require_confirmed=True,
        auto_start=False
    )

    good_track = PersonTrack(track_id=1, session_id="s1", first_seen=1.0, last_seen=1.0,
                             bbox=(10, 10, 40, 40), confidence=0.85, confirmed=True, state="active")
    low_conf_track = PersonTrack(track_id=2, session_id="s1", first_seen=1.0, last_seen=1.0,
                                 bbox=(10, 10, 40, 40), confidence=0.25, confirmed=True, state="active")
    lost_track = PersonTrack(track_id=3, session_id="s1", first_seen=1.0, last_seen=1.0,
                             bbox=(10, 10, 40, 40), confidence=0.90, confirmed=True, state="lost")
    unconfirmed_track = PersonTrack(track_id=4, session_id="s1", first_seen=1.0, last_seen=1.0,
                                   bbox=(10, 10, 40, 40), confidence=0.90, confirmed=False, state="active")

    tracks = [good_track, low_conf_track, lost_track, unconfirmed_track]
    selected = scheduler.select_tracks_for_face_analysis(tracks)
    assert len(selected) == 1
    assert selected[0].track_id == 1

    scheduler.shutdown(wait=True)


def test_backpressure_analysis_queue_saturation():
    """Verify analysis queue enforces bounded capacity and drops frames under saturation."""
    scheduler = PipelineScheduler(max_analysis_queue_size=5, auto_start=False)

    def dummy_packet(seq):
        return FramePacket(
            sequence=seq,
            timestamp=100.0 + seq * 0.04,
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            source_id="cam1",
            epoch="s1",
            width=10,
            height=10
        )

    # Enqueue up to capacity
    for i in range(5):
        assert scheduler.enqueue_analysis_frame(dummy_packet(i)) is True

    assert scheduler.analysis_queue_depth == 5
    assert scheduler.get_metrics()["dropped_frames"] == 0

    # Enqueue 3 extra frames with 'oldest' drop strategy
    for i in range(5, 8):
        # Returns False indicating backpressure drop occurred
        assert scheduler.enqueue_analysis_frame(dummy_packet(i), drop_strategy="oldest") is False

    # Queue remains strictly bounded at 5
    assert scheduler.analysis_queue_depth == 5
    metrics = scheduler.get_metrics()
    assert metrics["dropped_frames"] == 3
    assert metrics["dropped_analysis_frames"] == 3

    # Oldest frames (0, 1, 2) were dropped; queue contains 3, 4, 5, 6, 7
    remaining_seqs = []
    while scheduler.analysis_queue_depth > 0:
        remaining_seqs.append(scheduler.get_analysis_frame().sequence)
    assert remaining_seqs == [3, 4, 5, 6, 7]

    # Test 'newest' drop strategy
    for i in range(5):
        scheduler.enqueue_analysis_frame(dummy_packet(i))
    assert scheduler.enqueue_analysis_frame(dummy_packet(99), drop_strategy="newest") is False
    assert scheduler.get_metrics()["dropped_frames"] == 4

    scheduler.shutdown(wait=True)


def test_backpressure_disk_and_enhancement_queues():
    """Verify disk and super-resolution queues respect capacity limits."""
    scheduler = PipelineScheduler(
        max_disk_queue_size=20,
        max_enhancement_queue_size=10,
        auto_start=False
    )

    # Fill disk queue
    for i in range(20):
        assert scheduler.enqueue_disk_capture(f"capture_{i}") is True
    assert scheduler.disk_queue_depth == 20

    # Overflow disk queue by 5 items
    for i in range(20, 25):
        assert scheduler.enqueue_disk_capture(f"capture_{i}") is False
    assert scheduler.disk_queue_depth == 20
    assert scheduler.get_metrics()["dropped_disk_captures"] == 5

    # Fill enhancement queue
    for i in range(10):
        assert scheduler.enqueue_enhancement(f"enhance_{i}") is True
    assert scheduler.enhancement_queue_depth == 10

    # Overflow enhancement queue by 3 items
    for i in range(10, 13):
        assert scheduler.enqueue_enhancement(f"enhance_{i}") is False
    assert scheduler.enhancement_queue_depth == 10
    assert scheduler.get_metrics()["dropped_enhancement_tasks"] == 3

    scheduler.shutdown(wait=True)


def test_adaptive_frame_skipping_high_latency():
    """Verify adaptive analysis frame dropping under high latency while preserving streaming."""
    scheduler = PipelineScheduler(target_latency_budget_ms=100.0, auto_start=False)

    # Initial low latency within budget (45ms < 100ms)
    scheduler.record_frame_latency(45.0)
    assert scheduler.should_skip_analysis_frame() is False
    assert scheduler.get_metrics()["dropped_frames"] == 0

    # Live streaming frame published (Tier 1: 0 drops)
    scheduler.publish_stream_frame("frame_data")
    assert scheduler.get_latest_stream_frame() == "frame_data"
    assert scheduler.get_metrics()["streaming_frames_published"] == 1

    # Latency spike exceeds budget (160ms > 100ms)
    for _ in range(5):
        scheduler.record_frame_latency(160.0)
    assert scheduler.current_latency_ms > 100.0
    assert scheduler.last_latency_ms == 160.0

    # Analysis frame is dropped to relieve CPU/GPU pressure
    assert scheduler.should_skip_analysis_frame() is True
    assert scheduler.get_metrics()["dropped_frames"] == 1
    assert scheduler.get_metrics()["latency_skips"] == 1

    # Live streaming continues with 0 dropped frames
    scheduler.publish_stream_frame("fresh_frame_data")
    assert scheduler.get_latest_stream_frame() == "fresh_frame_data"
    assert scheduler.get_metrics()["streaming_frames_published"] == 2

    # Latency recovers (30ms)
    for _ in range(15):
        scheduler.record_frame_latency(30.0)
    assert scheduler.current_latency_ms < 100.0
    assert scheduler.should_skip_analysis_frame() is False

    scheduler.shutdown(wait=True)


def test_stale_track_buffer_cleanup_and_memory_safeguard():
    """Verify stale track buffers are automatically pruned to prevent memory leaks."""
    scheduler = PipelineScheduler(
        stale_track_timeout_s=5.0,
        max_tracked_history=10,
        auto_start=False
    )

    t0 = 100.0
    tracks = [
        PersonTrack(
            track_id=i, session_id=f"s:{i}", first_seen=t0, last_seen=t0,
            bbox=(10, 10, 50, 50), confidence=0.9, confirmed=True, speed_px_s=10.0
        )
        for i in range(1, 15)
    ]

    # Evaluate tracks initially
    scheduler.select_tracks_for_face_analysis(tracks, max_evaluations=14, timestamp=t0)
    assert scheduler.active_track_count <= 10  # Capped by max_tracked_history

    # Advance time beyond timeout and simulate tracks 1-5 active, others left
    active_ids = {1, 2, 3, 4, 5}
    t_later = t0 + 10.0
    pruned = scheduler.prune_stale_tracks(active_ids, current_time=t_later)
    assert pruned > 0
    assert scheduler.active_track_count <= 5

    scheduler.shutdown(wait=True)


def test_thread_pool_shutdown_and_cleanup():
    """Verify worker thread pool lifecycle and clean termination."""
    scheduler = PipelineScheduler(worker_threads=2, auto_start=True)
    assert scheduler.is_running is True
    assert scheduler.get_metrics()["worker_threads"] == 2
    assert scheduler.get_metrics()["active_workers"] == 2

    # Execute a background task
    f = scheduler.submit_disk_task(lambda x: x * 2, 21)
    assert f.result(timeout=2.0) == 42
    assert scheduler.get_metrics()["tasks_completed"] == 1

    # Shutdown
    scheduler.shutdown(wait=True)
    assert scheduler.is_running is False
    assert scheduler.get_metrics()["active_workers"] == 0

    # Submitting to shut down scheduler raises RuntimeError
    with pytest.raises(RuntimeError):
        scheduler.submit_disk_task(lambda: 1)


def test_context_manager():
    """Verify context manager semantics."""
    with PipelineScheduler(worker_threads=2) as scheduler:
        assert scheduler.is_running is True
        f = scheduler.submit_face_task(lambda: "processed")
        assert f.result(timeout=2.0) == "processed"

    assert scheduler.is_running is False


def test_processing_metrics_compatibility():
    """Verify get_processing_metrics returns standard contract."""
    scheduler = PipelineScheduler(auto_start=False)
    scheduler.record_stage_latency("detection", 18.2)
    scheduler.record_stage_latency("tracking", 4.5)
    scheduler.record_stage_latency("face_detection", 12.1)
    scheduler.record_stage_latency("quality_eval", 7.8)
    scheduler.record_stage_latency("enhancement", 25.0)

    pm = scheduler.get_processing_metrics(frame_sequence=42)
    assert isinstance(pm, ProcessingMetrics)
    assert pm.frame_sequence == 42
    assert pm.detection_ms == 18.2
    assert pm.tracking_ms == 4.5
    assert pm.face_detection_ms == 12.1
    assert pm.quality_eval_ms == 7.8
    assert pm.enhancement_ms == 25.0

    d = pm.to_dict()
    assert d["frame_sequence"] == 42
    assert d["detection_ms"] == 18.2


def test_task_exception_resilience():
    """Verify that a failing task sets exception on future without terminating the worker."""
    with PipelineScheduler(worker_threads=1) as scheduler:
        def bad_task():
            raise ValueError("simulated computational failure")

        f_bad = scheduler.submit_tracking_task(bad_task)
        with pytest.raises(ValueError, match="simulated computational failure"):
            f_bad.result(timeout=2.0)

        # Subsequent tasks succeed normally
        f_good = scheduler.submit_tracking_task(lambda: "healthy")
        assert f_good.result(timeout=2.0) == "healthy"
        assert scheduler.get_metrics()["tasks_failed"] == 1
        assert scheduler.get_metrics()["tasks_completed"] == 1


def test_duck_typing_tracks():
    """Verify scheduler seamlessly processes dict tracks and alternate schemas."""
    scheduler = PipelineScheduler(auto_start=False)

    dict_tracks = [
        {"id": 101, "confidence": 0.88, "bbox": [10, 20, 50, 60], "speed_px_s": 8.0, "state": "active"},
        {"track_id": 102, "confidence": 0.92, "bbox": [100, 120, 150, 160], "speed_px_s": 14.0, "state": "active"},
    ]

    selected = scheduler.select_tracks_for_face_analysis(dict_tracks, max_evaluations=2)
    assert len(selected) == 2
    assert [t.get("id") or t.get("track_id") for t in selected] == [101, 102]


def test_empty_and_edge_case_track_inputs():
    """Verify scheduler handles empty or edge-case track inputs safely."""
    scheduler = PipelineScheduler(auto_start=False)
    assert scheduler.select_tracks_for_face_analysis([]) == []
    assert scheduler.select_tracks_for_face_analysis([], max_evaluations=0) == []


def test_reset_metrics():
    """Verify performance metrics reset."""
    scheduler = PipelineScheduler(auto_start=False)
    scheduler.record_frame_latency(88.0)
    scheduler.record_stage_latency("detection", 20.0)
    assert scheduler.get_metrics()["last_latency_ms"] == 88.0

    scheduler.reset_metrics()
    assert scheduler.get_metrics()["last_latency_ms"] == 0.0
    assert scheduler.current_latency_ms == 0.0

