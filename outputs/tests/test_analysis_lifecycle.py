"""Unit tests verifying thread-safe operational lifecycle management for People Analysis."""
import queue
import threading
import time
from unittest.mock import MagicMock
import pytest

from services.face_contracts import AnalysisState
from services.analysis_lifecycle import (
    AnalysisLifecycleController,
    AnalysisLifecycleState,
    LifecycleState,
)


def test_initial_state():
    """Verify initial state of controller is STOPPED and inactive."""
    controller = AnalysisLifecycleController()
    assert controller.lifecycle_state == AnalysisLifecycleState.STOPPED
    assert controller.lifecycle_state == LifecycleState.STOPPED
    assert controller.is_running is False
    assert controller.is_stopping is False
    assert controller.current_epoch is None
    assert controller.is_detection_active is False
    assert controller.is_tracking_active is False
    assert controller.is_face_eval_active is False
    assert controller.is_capture_selection_active is False
    assert controller.stop_event.is_set() is True
    assert controller.running_event.is_set() is False

    state = controller.get_state()
    assert isinstance(state, AnalysisState)
    assert state.is_running is False
    assert state.status_message == "Stopped"
    assert state.error is None


def test_clean_transitions_stopped_running_stopped():
    """Verify clean transitions from STOPPED -> STARTING -> RUNNING -> STOPPING -> STOPPED."""
    recorded_states = []

    def on_change(life_state, ana_state):
        recorded_states.append((life_state, ana_state.is_running, ana_state.status_message))

    controller = AnalysisLifecycleController(on_state_change=on_change)

    # 1. Start analysis
    success = controller.start_analysis(epoch_id="test_epoch_001")
    assert success is True
    assert controller.lifecycle_state == AnalysisLifecycleState.RUNNING
    assert controller.is_running is True
    assert controller.current_epoch == "test_epoch_001"
    assert controller.is_detection_active is True
    assert controller.is_tracking_active is True
    assert controller.is_face_eval_active is True
    assert controller.is_capture_selection_active is True
    assert controller.stop_event.is_set() is False
    assert controller.running_event.is_set() is True

    running_state = controller.get_state()
    assert running_state.is_running is True
    assert running_state.started_at is not None
    assert running_state.stopped_at is None
    assert running_state.error is None

    # 2. Stop analysis
    stopped = controller.stop_analysis(drain_timeout=1.0)
    assert stopped is True
    assert controller.lifecycle_state == AnalysisLifecycleState.STOPPED
    assert controller.is_running is False
    assert controller.is_detection_active is False
    assert controller.is_tracking_active is False
    assert controller.is_face_eval_active is False
    assert controller.is_capture_selection_active is False
    assert controller.stop_event.is_set() is True
    assert controller.running_event.is_set() is False

    stopped_state = controller.get_state()
    assert stopped_state.is_running is False
    assert stopped_state.stopped_at is not None

    # Check observed transitions
    observed_enums = [s[0] for s in recorded_states]
    assert AnalysisLifecycleState.STARTING in observed_enums
    assert AnalysisLifecycleState.RUNNING in observed_enums
    assert AnalysisLifecycleState.STOPPING in observed_enums
    assert AnalysisLifecycleState.STOPPED in observed_enums


def test_idempotency_consecutive_start_and_stop():
    """Verify consecutive start() and stop() calls are safe, idempotent, and non-destructive."""
    controller = AnalysisLifecycleController()

    # First start: succeeds
    assert controller.start_analysis(epoch_id="epoch_alpha") is True
    first_epoch = controller.current_epoch
    first_started_at = controller.get_state().started_at

    # Second consecutive start: idempotent no-op, returns False, state remains RUNNING
    assert controller.start_analysis(epoch_id="epoch_beta") is False
    assert controller.lifecycle_state == AnalysisLifecycleState.RUNNING
    assert controller.current_epoch == first_epoch  # Epoch was not mutated
    assert controller.get_state().started_at == first_started_at

    # Third consecutive start: still idempotent
    assert controller.start_analysis() is False
    assert controller.lifecycle_state == AnalysisLifecycleState.RUNNING

    # First stop: succeeds
    assert controller.stop_analysis() is True
    assert controller.lifecycle_state == AnalysisLifecycleState.STOPPED
    first_stopped_at = controller.get_state().stopped_at

    # Second consecutive stop: idempotent no-op, returns False, state remains STOPPED
    assert controller.stop_analysis() is False
    assert controller.lifecycle_state == AnalysisLifecycleState.STOPPED
    assert controller.get_state().stopped_at == first_stopped_at

    # Third consecutive stop: still idempotent
    assert controller.stop_analysis() is False
    assert controller.lifecycle_state == AnalysisLifecycleState.STOPPED


def test_restart_analysis_atomic():
    """Verify restart_analysis performs clean stop and start with a new epoch."""
    controller = AnalysisLifecycleController()

    assert controller.start_analysis(epoch_id="epoch_1") is True
    assert controller.current_epoch == "epoch_1"
    assert controller.is_running is True

    # Atomic restart with explicit new epoch
    assert controller.restart_analysis(epoch_id="epoch_2") is True
    assert controller.current_epoch == "epoch_2"
    assert controller.is_running is True
    assert controller.lifecycle_state == AnalysisLifecycleState.RUNNING

    # Restart without explicit epoch generates a new unique epoch
    assert controller.restart_analysis() is True
    assert controller.current_epoch != "epoch_2"
    assert controller.current_epoch.startswith("epoch_")
    assert controller.is_running is True

    controller.stop_analysis()
    assert controller.is_running is False


def test_draining_capture_writes_on_stop():
    """Verify pending capture disk writes are safely drained before stopping."""
    written_captures = []

    def mock_writer(item):
        written_captures.append(item)

    controller = AnalysisLifecycleController(capture_writer=mock_writer)
    assert controller.start_analysis() is True

    # Enqueue multiple captures to disk queue
    captures = [
        {"capture_id": f"cap_{i}", "track_id": i, "score": 85.0 + i}
        for i in range(5)
    ]
    for cap in captures:
        assert controller.enqueue_capture_write(cap) is True

    assert controller.capture_queue.qsize() == 5

    # Stopping must safely drain pending writes
    controller.stop_analysis(drain_timeout=2.0)

    assert controller.lifecycle_state == AnalysisLifecycleState.STOPPED
    assert controller.capture_queue.empty() is True
    assert len(written_captures) == 5
    assert written_captures == captures
    assert controller.get_state().total_captures_count >= 5


def test_draining_with_custom_writer_object():
    """Verify custom capture writer with .write() and .drain() is handled properly."""
    class CustomWriter:
        def __init__(self):
            self.saved = []
            self.drain_called = False

        def write(self, item):
            self.saved.append(item)

        def drain(self, timeout):
            self.drain_called = True
            return len(self.saved)

    writer = CustomWriter()
    controller = AnalysisLifecycleController(capture_writer=writer)
    controller.start_analysis()

    controller.enqueue_capture_write({"test": 123})
    controller.stop_analysis(drain_timeout=1.0)

    assert writer.drain_called is True
    assert len(writer.saved) == 1
    assert writer.saved[0] == {"test": 123}


def test_drain_timeout_bound():
    """Verify drain respects timeout deadline and does not block indefinitely."""
    def slow_writer(item):
        time.sleep(0.3)

    controller = AnalysisLifecycleController(capture_writer=slow_writer)
    controller.start_analysis()

    for i in range(20):
        controller.enqueue_capture_write(i)

    t0 = time.monotonic()
    controller.stop_analysis(drain_timeout=0.15)
    elapsed = time.monotonic() - t0

    # Ensure it exited near timeout rather than processing all 20 items (which would take ~6.0s)
    assert elapsed < 1.0
    assert controller.lifecycle_state == AnalysisLifecycleState.STOPPED


def test_cancels_unstarted_enhancements_on_stop():
    """Verify unstarted expensive enhancement jobs are cancelled on stop."""
    cancelled_jobs = []

    class MockEnhancementJob:
        def __init__(self, job_id):
            self.job_id = job_id
            self.cancelled = False

        def cancel(self):
            self.cancelled = True
            cancelled_jobs.append(self.job_id)

    controller = AnalysisLifecycleController()
    controller.start_analysis()

    jobs = [MockEnhancementJob(f"job_{i}") for i in range(4)]
    for job in jobs:
        controller.enqueue_enhancement(job)

    assert controller.enhancement_queue.qsize() == 4
    assert controller.get_state().pending_enhancements == 4

    # Stop analysis cancels unstarted enhancement jobs
    controller.stop_analysis()

    assert controller.enhancement_queue.empty() is True
    assert controller.get_state().pending_enhancements == 0
    assert len(cancelled_jobs) == 4
    for job in jobs:
        assert job.cancelled is True


def test_camera_stream_remains_unaffected():
    """Verify camera stream is preserved and completely unaffected by start/stop."""
    class MockStreamManager:
        def __init__(self):
            self.is_active = True
            self.frame_count = 100
            self.closed = False

        def close(self):
            self.closed = True

        def stop(self):
            self.closed = True

    stream = MockStreamManager()
    controller = AnalysisLifecycleController(stream_manager=stream)

    # Start analysis
    controller.start_analysis()
    assert stream.is_active is True
    assert stream.closed is False

    # Stop analysis
    controller.stop_analysis()
    assert stream.is_active is True
    assert stream.closed is False

    # Restart analysis
    controller.restart_analysis()
    assert stream.is_active is True
    assert stream.closed is False

    controller.stop_analysis()
    assert stream.is_active is True
    assert stream.closed is False


def test_accumulated_profiles_preserved():
    """Verify accumulated profiles in profile storage are preserved intact."""
    class MockProfileStorage:
        def __init__(self):
            self.profiles = {"track_1": {"name": "person_1"}, "track_2": {"name": "person_2"}}
            self.cleared = False

        def clear(self):
            self.cleared = True

        def reset(self):
            self.cleared = True

    storage = MockProfileStorage()
    controller = AnalysisLifecycleController(profile_storage=storage)

    controller.start_analysis()
    controller.stop_analysis()
    controller.restart_analysis()
    controller.stop_analysis()

    assert storage.cleared is False
    assert len(storage.profiles) == 2
    assert "track_1" in storage.profiles


def test_pipeline_components_activation():
    """Verify registered components receive lifecycle signals."""
    detector = MagicMock()
    tracker = MagicMock()
    face_evaluator = MagicMock()
    capture_selector = MagicMock()

    controller = AnalysisLifecycleController(
        detector=detector,
        tracker=tracker,
        face_evaluator=face_evaluator,
        capture_selector=capture_selector,
    )

    controller.start_analysis(epoch_id="epoch_xyz")
    detector.activate.assert_called_once_with(epoch="epoch_xyz")
    tracker.activate.assert_called_once_with(epoch="epoch_xyz")
    face_evaluator.activate.assert_called_once_with(epoch="epoch_xyz")
    capture_selector.activate.assert_called_once_with(epoch="epoch_xyz")

    controller.stop_analysis()
    detector.deactivate.assert_called_once()
    tracker.deactivate.assert_called_once()
    face_evaluator.deactivate.assert_called_once()
    capture_selector.deactivate.assert_called_once()


def test_worker_thread_panic_reporting():
    """Verify worker thread panics are caught and reported in AnalysisState."""
    controller = AnalysisLifecycleController()

    def bad_worker():
        time.sleep(0.05)
        raise RuntimeError("Fatal worker failure simulation")

    controller.register_worker("failing_worker", bad_worker)

    controller.start_analysis()
    assert controller.is_running is True

    # Wait for the worker thread to panic
    deadline = time.monotonic() + 1.0
    while controller.lifecycle_state != AnalysisLifecycleState.ERROR and time.monotonic() < deadline:
        time.sleep(0.02)

    assert controller.lifecycle_state == AnalysisLifecycleState.ERROR
    assert controller.is_running is False
    state = controller.get_state()
    assert state.is_running is False
    assert "Fatal worker failure simulation" in str(state.error)
    assert controller.stop_event.is_set() is True


def test_error_recovery_via_start_and_stop():
    """Verify controller can recover from ERROR state by start_analysis or stop_analysis."""
    controller = AnalysisLifecycleController()
    controller.start_analysis()

    # Trigger error
    controller.report_worker_error(RuntimeError("Engine failure"), worker_name="evaluator")
    assert controller.lifecycle_state == AnalysisLifecycleState.ERROR
    assert controller.get_state().error is not None

    # Recovery 1: Calling start_analysis() resets error and returns to RUNNING
    recovered = controller.start_analysis()
    assert recovered is True
    assert controller.lifecycle_state == AnalysisLifecycleState.RUNNING
    assert controller.is_running is True
    assert controller.get_state().error is None

    # Trigger error again
    controller.report_worker_error("Second crash")
    assert controller.lifecycle_state == AnalysisLifecycleState.ERROR

    # Recovery 2: Calling stop_analysis() cleans up and transitions to STOPPED
    stopped = controller.stop_analysis()
    assert stopped is True
    assert controller.lifecycle_state == AnalysisLifecycleState.STOPPED
    assert controller.is_running is False


def test_listener_subscription_and_unsubscription():
    """Verify listeners subscribe, receive events, and unsubscribe cleanly."""
    events = []

    def my_listener(life_state, ana_state):
        events.append((life_state, ana_state.status_message))

    controller = AnalysisLifecycleController()
    unsubscribe = controller.subscribe(my_listener)

    controller.start_analysis()
    controller.stop_analysis()

    assert len(events) >= 4
    observed = [e[0] for e in events]
    assert AnalysisLifecycleState.STARTING in observed
    assert AnalysisLifecycleState.RUNNING in observed
    assert AnalysisLifecycleState.STOPPING in observed
    assert AnalysisLifecycleState.STOPPED in observed

    # Unsubscribe and verify no further notifications
    count_before = len(events)
    unsubscribe()

    controller.start_analysis()
    controller.stop_analysis()
    assert len(events) == count_before


def test_faulty_listener_does_not_break_controller():
    """Verify that an exception in a listener does not disrupt controller state."""
    def buggy_listener(life_state, ana_state):
        raise ValueError("Bug in listener")

    controller = AnalysisLifecycleController()
    controller.subscribe(buggy_listener)

    # Should not raise exception
    assert controller.start_analysis() is True
    assert controller.is_running is True
    assert controller.stop_analysis() is True
    assert controller.is_running is False


def test_worker_with_stop_event():
    """Verify worker registered with pass_stop_event halts cleanly when signaled."""
    worker_finished = threading.Event()

    def cooperative_worker(stop_event):
        while not stop_event.is_set():
            time.sleep(0.02)
        worker_finished.set()

    controller = AnalysisLifecycleController()
    controller.register_worker("coop", cooperative_worker, pass_stop_event=True)

    controller.start_analysis()
    assert controller.is_running is True
    time.sleep(0.05)

    controller.stop_analysis()
    assert worker_finished.wait(timeout=1.0) is True
    assert controller.is_running is False


def test_concurrent_start_stop_thread_safety():
    """Verify thread-safety when multiple threads concurrently start and stop."""
    controller = AnalysisLifecycleController()
    errors = []

    def start_worker():
        for _ in range(15):
            try:
                controller.start_analysis()
                time.sleep(0.005)
            except Exception as e:
                errors.append(e)

    def stop_worker():
        for _ in range(15):
            try:
                controller.stop_analysis(drain_timeout=0.05)
                time.sleep(0.005)
            except Exception as e:
                errors.append(e)

    threads = [
        threading.Thread(target=start_worker),
        threading.Thread(target=start_worker),
        threading.Thread(target=stop_worker),
        threading.Thread(target=stop_worker),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    # Final state must be either RUNNING or STOPPED, never corrupt
    assert controller.lifecycle_state in (AnalysisLifecycleState.RUNNING, AnalysisLifecycleState.STOPPED)

    # Clean shutdown
    controller.stop_analysis()
    assert controller.lifecycle_state == AnalysisLifecycleState.STOPPED


def test_update_metrics_and_to_dict():
    """Verify update_metrics updates state counters and to_dict serialization works."""
    controller = AnalysisLifecycleController()
    controller.start_analysis(epoch_id="epoch_metrics")

    controller.update_metrics(
        fps=24.5,
        active_tracks_count=4,
        total_profiles_count=12,
        total_captures_count=35,
        pending_enhancements=2,
    )

    state = controller.get_state()
    assert state.fps == 24.5
    assert state.active_tracks_count == 4
    assert state.total_profiles_count == 12
    assert state.total_captures_count == 35
    assert state.pending_enhancements == 2

    info = controller.to_dict()
    assert info["lifecycle_state"] == "RUNNING"
    assert info["current_epoch"] == "epoch_metrics"
    assert info["is_detection_active"] is True
    assert info["state"]["fps"] == 24.5
    assert info["state"]["total_profiles_count"] == 12

    controller.stop_analysis()
