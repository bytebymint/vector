"""Operational lifecycle controller for the People Analysis Engine.

Provides thread-safe state management across STOPPED, STARTING, RUNNING,
STOPPING, and ERROR states, supporting idempotent controls, safe draining of
capture writes, cancellation of unstarted enhancement tasks, graceful worker
panic recovery, and listener subscription hooks.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
import logging
import queue
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid

try:
    from .face_contracts import AnalysisState
except ImportError:
    try:
        from services.face_contracts import AnalysisState
    except ImportError:
        # Fallback dataclass if face_contracts is not in sys.path
        @dataclass
        class AnalysisState:  # type: ignore
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

LOG = logging.getLogger(__name__)


class AnalysisLifecycleState(str, Enum):
    """Lifecycle states of the People Analysis Engine."""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


LifecycleState = AnalysisLifecycleState


@dataclass
class WorkerSpec:
    """Registered worker specification for automatic lifecycle management."""
    name: str
    target: Callable[..., Any]
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    daemon: bool = True
    pass_stop_event: bool = False


class AnalysisLifecycleController:
    """Controls the thread-safe operational lifecycle of People Analysis.

    Coordinates transitions between STOPPED, STARTING, RUNNING, STOPPING, and ERROR.
    Manages worker threads, task queues, session epochs, and graceful error handling.
    """

    def __init__(
        self,
        stream_manager: Any = None,
        detector: Any = None,
        tracker: Any = None,
        face_evaluator: Any = None,
        capture_selector: Any = None,
        capture_writer: Any = None,
        enhancer: Any = None,
        profile_storage: Any = None,
        drain_timeout: float = 2.0,
        on_state_change: Optional[Callable[[AnalysisLifecycleState, AnalysisState], None]] = None,
    ):
        self._lock = threading.RLock()
        self._transition_lock = threading.RLock()

        # Pluggable subsystems (loose coupling)
        self.stream_manager = stream_manager
        self.detector = detector
        self.tracker = tracker
        self.face_evaluator = face_evaluator
        self.capture_selector = capture_selector
        self.capture_writer = capture_writer
        self.enhancer = enhancer
        self.profile_storage = profile_storage
        self.default_drain_timeout = float(drain_timeout)

        # State management
        self._lifecycle_state = AnalysisLifecycleState.STOPPED
        self._state = AnalysisState(
            is_running=False,
            status_message="Stopped",
        )
        self._current_epoch: Optional[str] = None

        # Synchronization primitives
        self._stop_event = threading.Event()
        self._stop_event.set()  # Initially stopped
        self._running_event = threading.Event()

        # Pipeline stage active flags
        self._is_detection_active = False
        self._is_tracking_active = False
        self._is_face_eval_active = False
        self._is_capture_selection_active = False

        # Queues for background tasks
        self._capture_queue: queue.Queue = queue.Queue()
        self._enhancement_queue: queue.Queue = queue.Queue()

        # Workers
        self._worker_specs: Dict[str, WorkerSpec] = {}
        self._active_threads: Dict[str, threading.Thread] = {}

        # Listeners
        self._listeners: List[Callable[[AnalysisLifecycleState, AnalysisState], None]] = []
        if on_state_change:
            self._listeners.append(on_state_change)

    # -------------------------------------------------------------------------
    # Lifecycle Control Methods
    # -------------------------------------------------------------------------

    def start_analysis(self, epoch_id: Optional[str] = None) -> bool:
        """Activates detection, tracking, face evaluation, and capture selection.

        Starts worker threads and task queues, initializes a new session epoch.
        Idempotent: safe if called multiple times while already starting or running.

        Returns:
            bool: True if started, False if already running or starting.
        """
        with self._transition_lock:
            with self._lock:
                if self._lifecycle_state in (AnalysisLifecycleState.RUNNING, AnalysisLifecycleState.STARTING):
                    LOG.info("start_analysis called while already %s (idempotent no-op)", self._lifecycle_state.value)
                    return False

                # Transition to STARTING
                self._set_lifecycle_state_locked(AnalysisLifecycleState.STARTING)
                self._state.error = None
                self._state.status_message = "Starting analysis"

            self._notify_listeners(self._lifecycle_state, self._state)

            try:
                # 1. Initialize session epoch
                epoch = epoch_id or f"epoch_{int(time.time())}_{uuid.uuid4().hex[:8]}"

                with self._lock:
                    self._current_epoch = epoch
                    self._stop_event.clear()
                    self._running_event.set()

                    # 2. Activate pipeline stages
                    self._is_detection_active = True
                    self._is_tracking_active = True
                    self._is_face_eval_active = True
                    self._is_capture_selection_active = True

                # 3. Activate registered components if applicable (never touching camera stream)
                self._activate_components(epoch)

                # 4. Start registered worker threads
                self._start_registered_workers()

                with self._lock:
                    now = time.time()
                    self._state.is_running = True
                    self._state.started_at = now
                    self._state.stopped_at = None
                    self._state.status_message = "Analysis running"
                    self._set_lifecycle_state_locked(AnalysisLifecycleState.RUNNING)

                self._notify_listeners(self._lifecycle_state, self._state)
                LOG.info("People Analysis started successfully in epoch %s", epoch)
                return True

            except Exception as exc:
                LOG.exception("Failed to start analysis: %s", exc)
                with self._lock:
                    self._is_detection_active = False
                    self._is_tracking_active = False
                    self._is_face_eval_active = False
                    self._is_capture_selection_active = False
                    self._stop_event.set()
                    self._running_event.clear()
                    self._state.is_running = False
                    self._state.error = str(exc)
                    self._state.status_message = f"Startup failed: {exc}"
                    self._set_lifecycle_state_locked(AnalysisLifecycleState.ERROR)

                self._notify_listeners(self._lifecycle_state, self._state)
                return False

    def stop_analysis(self, drain_timeout: Optional[float] = None) -> bool:
        """Halts analysis workers and safely drains pending capture disk writes.

        Cancels unstarted expensive enhancement jobs, preserves accumulated
        profiles and captures intact, and leaves camera stream unaffected.
        Idempotent: safe if called repeatedly.

        Returns:
            bool: True if stopped, False if already stopped or stopping.
        """
        timeout = float(self.default_drain_timeout if drain_timeout is None else drain_timeout)
        with self._transition_lock:
            with self._lock:
                if self._lifecycle_state == AnalysisLifecycleState.STOPPED:
                    LOG.info("stop_analysis called while already STOPPED (idempotent no-op)")
                    return False
                if self._lifecycle_state == AnalysisLifecycleState.STOPPING:
                    LOG.info("stop_analysis called while already STOPPING")
                    return False

                # Transition to STOPPING
                self._set_lifecycle_state_locked(AnalysisLifecycleState.STOPPING)
                self._state.status_message = "Stopping analysis"

                # Signal workers to halt
                self._stop_event.set()
                self._running_event.clear()

                # Deactivate pipeline flags immediately so no new analysis jobs are accepted
                self._is_detection_active = False
                self._is_tracking_active = False
                self._is_face_eval_active = False
                self._is_capture_selection_active = False

            self._notify_listeners(self._lifecycle_state, self._state)

            # Deactivate registered components (preserving stream and profiles!)
            self._deactivate_components()

            # Cancel unstarted expensive enhancement jobs
            cancelled_enhancements = self.cancel_pending_enhancements()
            LOG.debug("Cancelled %d unstarted enhancement jobs", cancelled_enhancements)

            # Safely drain pending capture disk writes within timeout
            drained_captures = self.drain_capture_writes(timeout=timeout)
            LOG.debug("Drained %d pending capture disk writes", drained_captures)

            # Join active worker threads within timeout
            self._stop_active_workers(timeout=max(0.2, timeout / 2.0))

            with self._lock:
                now = time.time()
                self._state.is_running = False
                self._state.stopped_at = now
                self._state.status_message = "Stopped"
                self._set_lifecycle_state_locked(AnalysisLifecycleState.STOPPED)

            self._notify_listeners(self._lifecycle_state, self._state)
            LOG.info("People Analysis stopped successfully")
            return True

    def pause_analysis(self) -> bool:
        """Pause active analysis processing without stopping workers."""
        with self._lock:
            if not self._state.is_running:
                return False
            self._is_detection_active = False
            self._is_tracking_active = False
            self._is_face_eval_active = False
            self._is_capture_selection_active = False
            self._state.status_message = "Paused"
            return True

    def resume_analysis(self) -> bool:
        """Resume active analysis processing."""
        with self._lock:
            if not self._state.is_running:
                return False
            self._is_detection_active = True
            self._is_tracking_active = True
            self._is_face_eval_active = True
            self._is_capture_selection_active = True
            self._state.status_message = "Running"
            return True

    def start(self, epoch_id: Optional[str] = None) -> bool:
        """Alias for start_analysis."""
        return self.start_analysis(epoch_id=epoch_id)

    def stop(self, drain_timeout: Optional[float] = None) -> bool:
        """Alias for stop_analysis."""
        return self.stop_analysis(drain_timeout=drain_timeout)

    def restart_analysis(
        self,
        drain_timeout: Optional[float] = None,
        epoch_id: Optional[str] = None
    ) -> bool:
        """Atomic stop followed by clean start."""
        with self._transition_lock:
            self.stop_analysis(drain_timeout=drain_timeout)
            return self.start_analysis(epoch_id=epoch_id)

    # -------------------------------------------------------------------------
    # Pipeline & Component Coordination
    # -------------------------------------------------------------------------

    def _activate_components(self, epoch: str) -> None:
        """Activate pipeline components without touching camera stream."""
        for comp in [self.detector, self.tracker, self.face_evaluator, self.capture_selector]:
            if comp is None:
                continue
            try:
                if hasattr(comp, "activate") and callable(comp.activate):
                    comp.activate(epoch=epoch)
                elif hasattr(comp, "reset") and callable(comp.reset):
                    comp.reset(epoch)
                elif hasattr(comp, "set_active") and callable(comp.set_active):
                    comp.set_active(True)
                elif hasattr(comp, "start") and callable(comp.start):
                    comp.start()
            except Exception as exc:
                LOG.warning("Failed to activate component %s: %s", comp, exc)

    def _deactivate_components(self) -> None:
        """Deactivate pipeline components without touching camera stream or profile storage."""
        for comp in [self.detector, self.tracker, self.face_evaluator, self.capture_selector]:
            if comp is None:
                continue
            try:
                if hasattr(comp, "deactivate") and callable(comp.deactivate):
                    comp.deactivate()
                elif hasattr(comp, "set_active") and callable(comp.set_active):
                    comp.set_active(False)
                elif hasattr(comp, "stop") and callable(comp.stop):
                    comp.stop()
            except Exception as exc:
                LOG.warning("Failed to deactivate component %s: %s", comp, exc)

    # -------------------------------------------------------------------------
    # Task Queues: Capture Draining & Enhancement Cancellation
    # -------------------------------------------------------------------------

    def enqueue_capture_write(self, item: Any) -> bool:
        """Enqueue a capture item for disk writing."""
        with self._lock:
            if not self.is_running and self._lifecycle_state != AnalysisLifecycleState.STARTING:
                LOG.warning("Cannot enqueue capture write when analysis is %s", self._lifecycle_state.value)
                return False
            self._capture_queue.put(item)
            return True

    def enqueue_enhancement(self, job: Any) -> bool:
        """Enqueue an enhancement job."""
        with self._lock:
            if not self.is_running and self._lifecycle_state != AnalysisLifecycleState.STARTING:
                LOG.warning("Cannot enqueue enhancement when analysis is %s", self._lifecycle_state.value)
                return False
            self._enhancement_queue.put(job)
            self._state.pending_enhancements += 1
            return True

    def cancel_pending_enhancements(self) -> int:
        """Cancels all unstarted expensive enhancement jobs from queue."""
        cancelled = 0
        while not self._enhancement_queue.empty():
            try:
                item = self._enhancement_queue.get_nowait()
            except queue.Empty:
                break

            try:
                if hasattr(item, "cancel") and callable(item.cancel):
                    item.cancel()
                cancelled += 1
            except Exception as exc:
                LOG.warning("Error cancelling enhancement item: %s", exc)
            finally:
                self._enhancement_queue.task_done()

        with self._lock:
            self._state.pending_enhancements = 0

        return cancelled

    def drain_capture_writes(self, timeout: float = 2.0) -> int:
        """Safely drains remaining pending capture disk writes within timeout."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        drained_count = 0

        # 1. If writer has dedicated drain method, invoke it
        if (
            self.capture_writer is not None
            and hasattr(self.capture_writer, "drain")
            and callable(self.capture_writer.drain)
        ):
            try:
                ext_drained = self.capture_writer.drain(timeout=max(0.1, deadline - time.monotonic()))
                if isinstance(ext_drained, int):
                    drained_count += ext_drained
            except Exception as exc:
                LOG.error("Error in capture_writer.drain: %s", exc)

        # 2. Drain any remaining items in queue
        while not self._capture_queue.empty() and time.monotonic() < deadline:
            try:
                item = self._capture_queue.get_nowait()
            except queue.Empty:
                break

            try:
                self._persist_capture_item(item)
                drained_count += 1
            except Exception as exc:
                LOG.error("Failed to write capture during drain: %s", exc)
            finally:
                self._capture_queue.task_done()

        with self._lock:
            self._state.total_captures_count += drained_count

        return drained_count

    def _persist_capture_item(self, item: Any) -> None:
        """Helper to write or persist an enqueued capture item."""
        if self.capture_writer is not None:
            if callable(self.capture_writer):
                self.capture_writer(item)
            elif hasattr(self.capture_writer, "write") and callable(self.capture_writer.write):
                self.capture_writer.write(item)
            elif hasattr(self.capture_writer, "save") and callable(self.capture_writer.save):
                self.capture_writer.save(item)
        elif callable(item):
            item()

    def set_capture_writer(self, writer: Callable[[Any], None] | Any) -> None:
        """Set a capture writer callable or component."""
        with self._lock:
            self.capture_writer = writer

    # -------------------------------------------------------------------------
    # Worker Thread Management & Panic Recovery
    # -------------------------------------------------------------------------

    def register_worker(
        self,
        name: str,
        target: Callable[..., Any],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        daemon: bool = True,
        pass_stop_event: bool = False,
    ) -> None:
        """Register a worker function to run during RUNNING lifecycle."""
        with self._lock:
            spec = WorkerSpec(
                name=name,
                target=target,
                args=args,
                kwargs=kwargs or {},
                daemon=daemon,
                pass_stop_event=pass_stop_event,
            )
            self._worker_specs[name] = spec
            if self._lifecycle_state == AnalysisLifecycleState.RUNNING:
                self._spawn_worker_thread(spec)

    def _start_registered_workers(self) -> None:
        """Spawn threads for all registered workers."""
        with self._lock:
            for spec in self._worker_specs.values():
                self._spawn_worker_thread(spec)

    def _spawn_worker_thread(self, spec: WorkerSpec) -> threading.Thread:
        """Wrap and spawn a worker thread with panic capture."""
        def _runner():
            LOG.debug("Worker '%s' started", spec.name)
            try:
                kwargs = dict(spec.kwargs)
                if spec.pass_stop_event:
                    kwargs["stop_event"] = self._stop_event
                spec.target(*spec.args, **kwargs)
                LOG.debug("Worker '%s' exited normally", spec.name)
            except Exception as exc:
                LOG.exception("Worker '%s' panicked: %s", spec.name, exc)
                self.report_worker_error(exc, worker_name=spec.name)

        thread = threading.Thread(
            target=_runner,
            name=f"analysis-worker-{spec.name}",
            daemon=spec.daemon,
        )
        with self._lock:
            self._active_threads[spec.name] = thread
        thread.start()
        return thread

    def _stop_active_workers(self, timeout: float = 1.0) -> None:
        """Join active worker threads within timeout."""
        with self._lock:
            threads = list(self._active_threads.values())
            self._active_threads.clear()

        deadline = time.monotonic() + timeout
        for t in threads:
            if t.is_alive() and t is not threading.current_thread():
                rem = max(0.01, deadline - time.monotonic())
                t.join(timeout=rem)

    def report_worker_error(
        self,
        error: Exception | str,
        worker_name: Optional[str] = None
    ) -> None:
        """Records a worker error/panic, transitions to ERROR state, and signals stop."""
        with self._lock:
            err_text = f"Worker '{worker_name}' error: {error}" if worker_name else str(error)
            LOG.error("Reporting worker error: %s", err_text)
            self._state.error = err_text
            self._state.status_message = f"Error: {err_text}"
            self._state.is_running = False
            self._state.stopped_at = time.time()
            self._set_lifecycle_state_locked(AnalysisLifecycleState.ERROR)
            self._stop_event.set()
            self._running_event.clear()

            # Deactivate pipeline flags on error
            self._is_detection_active = False
            self._is_tracking_active = False
            self._is_face_eval_active = False
            self._is_capture_selection_active = False

        self._notify_listeners(AnalysisLifecycleState.ERROR, self._state)

    # -------------------------------------------------------------------------
    # Listener Subscription Hooks
    # -------------------------------------------------------------------------

    def subscribe(
        self,
        listener: Callable[[AnalysisLifecycleState, AnalysisState], None]
    ) -> Callable[[], None]:
        """Subscribe a listener to lifecycle state changes.

        Returns an unsubscribe callable.
        """
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

        return lambda: self.unsubscribe(listener)

    def unsubscribe(
        self,
        listener: Callable[[AnalysisLifecycleState, AnalysisState], None]
    ) -> bool:
        """Unsubscribe a listener from lifecycle state changes."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)
                return True
            return False

    def _notify_listeners(
        self,
        state: AnalysisLifecycleState,
        analysis_state: AnalysisState
    ) -> None:
        """Dispatch notifications to listeners outside the internal lock."""
        with self._lock:
            listeners_copy = list(self._listeners)
            state_copy = copy.copy(analysis_state)

        for listener in listeners_copy:
            try:
                listener(state, state_copy)
            except Exception as exc:
                LOG.exception("Error in lifecycle listener %s: %s", listener, exc)

    # -------------------------------------------------------------------------
    # State Inspection & Helpers
    # -------------------------------------------------------------------------

    def _set_lifecycle_state_locked(self, new_state: AnalysisLifecycleState) -> None:
        self._lifecycle_state = new_state
        if new_state == AnalysisLifecycleState.RUNNING:
            self._state.is_running = True
        elif new_state in (AnalysisLifecycleState.STOPPED, AnalysisLifecycleState.ERROR):
            self._state.is_running = False

    @property
    def lifecycle_state(self) -> AnalysisLifecycleState:
        with self._lock:
            return self._lifecycle_state

    @property
    def state(self) -> AnalysisState:
        with self._lock:
            return copy.copy(self._state)

    def get_state(self) -> AnalysisState:
        with self._lock:
            return copy.copy(self._state)

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._lifecycle_state == AnalysisLifecycleState.RUNNING

    @property
    def is_stopping(self) -> bool:
        with self._lock:
            return self._lifecycle_state == AnalysisLifecycleState.STOPPING

    @property
    def current_epoch(self) -> Optional[str]:
        with self._lock:
            return self._current_epoch

    @property
    def is_detection_active(self) -> bool:
        with self._lock:
            return self._is_detection_active

    @property
    def is_tracking_active(self) -> bool:
        with self._lock:
            return self._is_tracking_active

    @property
    def is_face_eval_active(self) -> bool:
        with self._lock:
            return self._is_face_eval_active

    @property
    def is_capture_selection_active(self) -> bool:
        with self._lock:
            return self._is_capture_selection_active

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    @property
    def running_event(self) -> threading.Event:
        return self._running_event

    @property
    def capture_queue(self) -> queue.Queue:
        return self._capture_queue

    @property
    def enhancement_queue(self) -> queue.Queue:
        return self._enhancement_queue

    def update_metrics(
        self,
        fps: Optional[float] = None,
        active_tracks_count: Optional[int] = None,
        total_profiles_count: Optional[int] = None,
        total_captures_count: Optional[int] = None,
        pending_enhancements: Optional[int] = None,
    ) -> None:
        """Thread-safe update of analysis state counters and FPS."""
        with self._lock:
            if fps is not None:
                self._state.fps = float(fps)
            if active_tracks_count is not None:
                self._state.active_tracks_count = int(active_tracks_count)
            if total_profiles_count is not None:
                self._state.total_profiles_count = int(total_profiles_count)
            if total_captures_count is not None:
                self._state.total_captures_count = int(total_captures_count)
            if pending_enhancements is not None:
                self._state.pending_enhancements = int(pending_enhancements)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete controller status for JSON/HTTP consumers."""
        with self._lock:
            return {
                "lifecycle_state": self._lifecycle_state.value,
                "current_epoch": self._current_epoch,
                "is_detection_active": self._is_detection_active,
                "is_tracking_active": self._is_tracking_active,
                "is_face_eval_active": self._is_face_eval_active,
                "is_capture_selection_active": self._is_capture_selection_active,
                "state": self._state.to_dict(),
                "pending_captures_count": self._capture_queue.qsize(),
                "pending_enhancements_count": self._enhancement_queue.qsize(),
                "active_workers": list(self._active_threads.keys()),
            }
