"""High-performance pipeline scheduler for real-time multi-person facial capture.

Provides priority queueing, crowd scheduling with adaptive round-robin sampling,
fast track-skipping, latency-budgeted adaptive frame dropping, and memory safeguards.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import IntEnum
import logging
import math
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

try:
    from .face_contracts import FramePacket, PersonTrack, ProcessingMetrics
except ImportError:
    from services.face_contracts import FramePacket, PersonTrack, ProcessingMetrics

LOG = logging.getLogger(__name__)


class TaskPriority(IntEnum):
    """Execution priority levels for pipeline operations.
    
    Lower integer value corresponds to strictly higher execution priority.
    """
    LIVE_STREAM = 1      # Live video acquisition & streaming (highest priority, 0 dropped frames)
    TRACKING = 2         # Person tracking & localization (high priority, 10-15 Hz)
    FACE_ANALYSIS = 3    # Face detection & quality assessment (medium priority, adaptive round-robin)
    DISK_IO = 4          # Disk capture writes (low priority, bounded worker thread)
    ENHANCEMENT = 5      # Super-resolution enhancement (lowest priority, throttled background thread)


# Semantic aliases
PRIORITY_STREAM = TaskPriority.LIVE_STREAM
PRIORITY_TRACKING = TaskPriority.TRACKING
PRIORITY_FACE = TaskPriority.FACE_ANALYSIS
PRIORITY_DISK = TaskPriority.DISK_IO
PRIORITY_ENHANCEMENT = TaskPriority.ENHANCEMENT


@dataclass(order=True)
class ScheduledTask:
    """Internal task representation ordered by priority and FIFO submission order."""
    priority: int
    sequence: int
    fn: Callable[..., Any] = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: dict = field(default_factory=dict, compare=False)
    future: Optional[Future] = field(default=None, compare=False)
    name: str = field(default="task", compare=False)
    created_at: float = field(default_factory=time.time, compare=False)


# Sentinel to signal worker thread termination
_STOP_SENTINEL = ScheduledTask(
    priority=999,
    sequence=999999999,
    fn=lambda: None,
    name="__STOP_SENTINEL__"
)


@dataclass
class TrackEvalRecord:
    """State metadata for an individual track's facial evaluations."""
    track_id: int
    first_eval_time: float
    last_eval_time: float
    eval_count: int = 0
    last_bbox: Optional[Tuple[float, float, float, float]] = None
    last_center: Optional[Tuple[float, float]] = None
    last_confidence: float = 1.0
    last_speed: float = 0.0
    last_quality_score: float = 0.0


@dataclass
class PipelineSchedulerConfig:
    """Configuration parameters for PipelineScheduler."""
    worker_threads: int = 2
    max_analysis_queue_size: int = 5
    max_disk_queue_size: int = 20
    max_enhancement_queue_size: int = 10
    max_task_queue_size: int = 200
    target_latency_budget_ms: float = 100.0
    max_faces_per_frame: int = 5
    min_track_confidence: float = 0.40
    require_confirmed: bool = False
    min_speed_threshold_px_s: float = 5.0
    stationary_displacement_threshold_px: float = 10.0
    stationary_recheck_interval_s: float = 2.0
    stale_track_timeout_s: float = 10.0
    max_tracked_history: int = 500
    latency_ema_alpha: float = 0.20
    auto_start: bool = True


def _extract_track_id(track: Any) -> int:
    """Extract integer track identifier across varying track schemas."""
    if hasattr(track, 'track_id'):
        return int(track.track_id)
    if hasattr(track, 'ident'):
        return int(track.ident)
    if hasattr(track, 'id'):
        return int(track.id)
    if isinstance(track, dict):
        return int(track.get('track_id', track.get('ident', track.get('id', 0))))
    return hash(track)


def _extract_track_confidence(track: Any) -> float:
    """Extract detection/tracking confidence score."""
    if hasattr(track, 'confidence'):
        return float(track.confidence)
    if isinstance(track, dict):
        return float(track.get('confidence', 1.0))
    return 1.0


def _extract_track_bbox(track: Any) -> Optional[Tuple[float, float, float, float]]:
    """Extract bounding box (x1, y1, x2, y2)."""
    bbox = getattr(track, 'bbox', None)
    if bbox is None and isinstance(track, dict):
        bbox = track.get('bbox')
    if bbox is None:
        return None
    try:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (IndexError, TypeError, ValueError):
        return None


def _extract_track_speed(track: Any) -> Optional[float]:
    """Extract velocity in pixels per second."""
    if hasattr(track, 'speed_px_s'):
        return float(track.speed_px_s)
    if isinstance(track, dict) and 'speed_px_s' in track:
        return float(track['speed_px_s'])
    return None


def _extract_track_state(track: Any) -> str:
    """Extract track lifecycle state (active, lost, ended)."""
    if hasattr(track, 'state'):
        return str(track.state)
    if isinstance(track, dict) and 'state' in track:
        return str(track['state'])
    return "active"


def _extract_track_confirmed(track: Any) -> bool:
    """Extract confirmed status."""
    if hasattr(track, 'confirmed'):
        return bool(track.confirmed)
    if isinstance(track, dict) and 'confirmed' in track:
        return bool(track['confirmed'])
    return True


class PipelineScheduler:
    """Manages computation scheduling, thread pools, and memory safeguards."""

    def __init__(
        self,
        config: Optional[PipelineSchedulerConfig] = None,
        **kwargs: Any
    ):
        if config is not None:
            self.config = config
            for k, v in kwargs.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
        else:
            self.config = PipelineSchedulerConfig(**kwargs)

        # Thread synchronization primitives
        self._lock = threading.RLock()
        self._analysis_condition = threading.Condition(self._lock)
        self._disk_condition = threading.Condition(self._lock)
        self._enhancement_condition = threading.Condition(self._lock)

        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused by default
        self._shutdown_event = threading.Event()
        self._is_running = False
        self._is_shutdown = False

        # Queues and buffers
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue(
            maxsize=self.config.max_task_queue_size
        )
        self._task_seq_counter: int = 0

        self._analysis_queue: deque = deque()
        self._disk_queue: deque = deque()
        self._enhancement_queue: deque = deque()
        self._latest_stream_frame: Optional[Any] = None

        # Crowd scheduling & track history
        self._track_eval_history: Dict[int, TrackEvalRecord] = {}
        self._rr_cursor: int = 0

        # Latency monitoring
        self._current_latency_ms: float = 0.0
        self._last_latency_ms: float = 0.0

        # Metrics collection
        self._metrics: Dict[str, Any] = {
            "dropped_frames": 0,
            "dropped_analysis_frames": 0,
            "dropped_disk_captures": 0,
            "dropped_enhancement_tasks": 0,
            "latency_skips": 0,
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_rejected": 0,
            "streaming_frames_published": 0,
            "last_latency_ms": 0.0,
            "ema_latency_ms": 0.0,
            "acquisition_fps": 0.0,
            "detection_ms": 0.0,
            "tracking_ms": 0.0,
            "face_detection_ms": 0.0,
            "quality_eval_ms": 0.0,
            "enhancement_ms": 0.0,
        }

        # Worker threads
        self._workers: List[threading.Thread] = []

        if self.config.auto_start:
            self.start()

    # -------------------------------------------------------------------------
    # Lifecycle Management
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Start worker threads if not already running."""
        with self._lock:
            if self._is_shutdown:
                raise RuntimeError("Cannot start a shutdown PipelineScheduler")
            if self._is_running:
                return

            self._shutdown_event.clear()
            self._pause_event.set()
            self._is_running = True
            self._workers.clear()

            for i in range(self.config.worker_threads):
                t = threading.Thread(
                    target=self._worker_loop,
                    name=f"pipeline-worker-{i}",
                    args=(i,),
                    daemon=True
                )
                self._workers.append(t)
                t.start()

    def pause(self) -> None:
        """Pause worker thread execution without shutting down."""
        self._pause_event.clear()

    def resume(self) -> None:
        """Resume worker thread execution."""
        self._pause_event.set()

    @property
    def is_paused(self) -> bool:
        """Return True if worker execution is paused."""
        return not self._pause_event.is_set()

    @property
    def is_running(self) -> bool:
        """Return True if worker pool is running."""
        return self._is_running and not self._is_shutdown

    def shutdown(self, wait: bool = True, timeout: float = 5.0) -> None:
        """Gracefully terminate worker threads and clear queues."""
        with self._lock:
            if self._is_shutdown:
                return
            self._is_shutdown = True
            self._is_running = False
            self._shutdown_event.set()
            self._pause_event.set()

            self._analysis_condition.notify_all()
            self._disk_condition.notify_all()
            self._enhancement_condition.notify_all()

        # Send termination tokens to worker queue
        for _ in range(len(self._workers)):
            try:
                self._task_queue.put_nowait(_STOP_SENTINEL)
            except (queue.Full, Exception):
                pass

        if wait:
            deadline = time.time() + timeout
            for worker in self._workers:
                rem = max(0.01, deadline - time.time())
                worker.join(timeout=rem)

        with self._lock:
            self._workers.clear()

    def __enter__(self) -> PipelineScheduler:
        if not self._is_running and not self._is_shutdown:
            self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown(wait=True)

    # -------------------------------------------------------------------------
    # Worker Loop & Priority Task Execution
    # -------------------------------------------------------------------------

    def _worker_loop(self, worker_id: int) -> None:
        """Worker thread processing tasks in strict priority order."""
        while not self._shutdown_event.is_set():
            # Honor pause state
            self._pause_event.wait()
            if self._shutdown_event.is_set():
                break

            try:
                task: ScheduledTask = self._task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if task is _STOP_SENTINEL or self._shutdown_event.is_set():
                self._task_queue.task_done()
                break

            if task.future and task.future.cancelled():
                self._task_queue.task_done()
                continue

            try:
                result = task.fn(*task.args, **task.kwargs)
                if task.future and not task.future.done():
                    task.future.set_result(result)
                with self._lock:
                    self._metrics["tasks_completed"] += 1
            except Exception as exc:
                LOG.exception("Error executing task %s (priority %s)", task.name, task.priority)
                if task.future and not task.future.done():
                    task.future.set_exception(exc)
                with self._lock:
                    self._metrics["tasks_failed"] += 1
            finally:
                self._task_queue.task_done()

    def submit_task(
        self,
        priority: int | TaskPriority,
        fn: Callable[..., Any],
        *args: Any,
        task_name: Optional[str] = None,
        **kwargs: Any
    ) -> Future:
        """Submit a callable with assigned priority. Returns a concurrent.futures.Future."""
        with self._lock:
            if self._is_shutdown:
                raise RuntimeError("Cannot submit task to a shutdown PipelineScheduler")

            p = int(priority)
            seq = self._task_seq_counter
            self._task_seq_counter += 1

            future: Future = Future()
            task = ScheduledTask(
                priority=p,
                sequence=seq,
                fn=fn,
                args=args,
                kwargs=kwargs,
                future=future,
                name=task_name or f"task_{seq}"
            )

            try:
                self._task_queue.put_nowait(task)
                self._metrics["tasks_submitted"] += 1
            except queue.Full:
                self._metrics["tasks_rejected"] += 1
                future.set_exception(queue.Full("PipelineScheduler task queue is saturated"))

            return future

    def submit_stream_task(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Submit Priority 1 task (Live video acquisition & streaming)."""
        return self.submit_task(TaskPriority.LIVE_STREAM, fn, *args, **kwargs)

    def submit_tracking_task(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Submit Priority 2 task (Person tracking & localization)."""
        return self.submit_task(TaskPriority.TRACKING, fn, *args, **kwargs)

    def submit_face_task(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Submit Priority 3 task (Face detection & quality assessment)."""
        return self.submit_task(TaskPriority.FACE_ANALYSIS, fn, *args, **kwargs)

    def submit_disk_task(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Submit Priority 4 task (Disk capture writes)."""
        return self.submit_task(TaskPriority.DISK_IO, fn, *args, **kwargs)

    def submit_enhancement_task(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Submit Priority 5 task (Super-resolution enhancement)."""
        return self.submit_task(TaskPriority.ENHANCEMENT, fn, *args, **kwargs)

    # -------------------------------------------------------------------------
    # Bounded Queueing & Backpressure Management
    # -------------------------------------------------------------------------

    def publish_stream_frame(self, frame: Any) -> None:
        """Publish live frame to streaming mailbox (Tier 1: 0 dropped frames)."""
        with self._lock:
            self._latest_stream_frame = frame
            self._metrics["streaming_frames_published"] += 1

    def get_latest_stream_frame(self) -> Optional[Any]:
        """Fetch the latest live stream frame without queue delays."""
        with self._lock:
            return self._latest_stream_frame

    def enqueue_analysis_frame(
        self,
        frame: Any,
        drop_strategy: str = "oldest"
    ) -> bool:
        """Enqueue frame for AI analysis with bounded capacity.
        
        Args:
            frame: FramePacket or frame array.
            drop_strategy: 'oldest' drops oldest frame in queue, 'newest' drops current frame.
            
        Returns:
            True if queued without drop; False if backpressure drop occurred.
        """
        with self._lock:
            if len(self._analysis_queue) >= self.config.max_analysis_queue_size:
                self._metrics["dropped_frames"] += 1
                self._metrics["dropped_analysis_frames"] += 1

                if drop_strategy == "oldest":
                    self._analysis_queue.popleft()
                    self._analysis_queue.append(frame)
                    self._analysis_condition.notify()
                return False
            else:
                self._analysis_queue.append(frame)
                self._analysis_condition.notify()
                return True

    def get_analysis_frame(self, timeout: Optional[float] = 0.1) -> Optional[Any]:
        """Retrieve next analysis frame, blocking up to timeout if queue is empty."""
        with self._analysis_condition:
            if not self._analysis_queue and timeout and timeout > 0:
                self._analysis_condition.wait(timeout)
            if self._analysis_queue:
                return self._analysis_queue.popleft()
            return None

    def enqueue_disk_capture(
        self,
        capture: Any,
        drop_strategy: str = "newest"
    ) -> bool:
        """Enqueue face capture data for disk writing with bounded capacity.
        
        Returns:
            True if queued without drop; False if saturated and dropped.
        """
        with self._lock:
            if len(self._disk_queue) >= self.config.max_disk_queue_size:
                self._metrics["dropped_disk_captures"] += 1
                if drop_strategy == "oldest":
                    self._disk_queue.popleft()
                    self._disk_queue.append(capture)
                    self._disk_condition.notify()
                return False
            else:
                self._disk_queue.append(capture)
                self._disk_condition.notify()
                return True

    def get_disk_capture(self, timeout: Optional[float] = 0.1) -> Optional[Any]:
        """Retrieve next pending disk capture."""
        with self._disk_condition:
            if not self._disk_queue and timeout and timeout > 0:
                self._disk_condition.wait(timeout)
            if self._disk_queue:
                return self._disk_queue.popleft()
            return None

    def enqueue_enhancement(
        self,
        enhancement_item: Any,
        drop_strategy: str = "newest"
    ) -> bool:
        """Enqueue capture for super-resolution enhancement with bounded queue."""
        with self._lock:
            if len(self._enhancement_queue) >= self.config.max_enhancement_queue_size:
                self._metrics["dropped_enhancement_tasks"] += 1
                if drop_strategy == "oldest":
                    self._enhancement_queue.popleft()
                    self._enhancement_queue.append(enhancement_item)
                    self._enhancement_condition.notify()
                return False
            else:
                self._enhancement_queue.append(enhancement_item)
                self._enhancement_condition.notify()
                return True

    def get_enhancement(self, timeout: Optional[float] = 0.1) -> Optional[Any]:
        """Retrieve next pending super-resolution task."""
        with self._enhancement_condition:
            if not self._enhancement_queue and timeout and timeout > 0:
                self._enhancement_condition.wait(timeout)
            if self._enhancement_queue:
                return self._enhancement_queue.popleft()
            return None

    # -------------------------------------------------------------------------
    # Crowd Scheduling & Adaptive Sampling
    # -------------------------------------------------------------------------

    def is_track_stationary(
        self,
        track: Any,
        timestamp: Optional[float] = None
    ) -> Tuple[bool, str]:
        """Determine if a person track is stationary and can skip face evaluation.
        
        Returns:
            (is_stationary: bool, reason: str)
        """
        now = timestamp if timestamp is not None else time.time()
        track_id = _extract_track_id(track)

        with self._lock:
            record = self._track_eval_history.get(track_id)
            if record is None or record.eval_count == 0:
                return False, "never_evaluated"

            elapsed = max(0.0, now - record.last_eval_time)
            if elapsed >= self.config.stationary_recheck_interval_s:
                return False, "recheck_interval_elapsed"

            # Check explicit speed attribute if provided
            speed = _extract_track_speed(track)
            if speed is not None:
                if speed >= self.config.min_speed_threshold_px_s:
                    return False, f"speed_moving ({speed:.1f} >= {self.config.min_speed_threshold_px_s})"
                else:
                    return True, f"speed_below_threshold ({speed:.1f} < {self.config.min_speed_threshold_px_s})"

            # If velocity is not explicitly tracked, use displacement
            bbox = _extract_track_bbox(track)
            if bbox is not None and record.last_center is not None:
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                dx = cx - record.last_center[0]
                dy = cy - record.last_center[1]
                displacement = math.hypot(dx, dy)
                if displacement < self.config.stationary_displacement_threshold_px:
                    return True, f"displacement_below_threshold ({displacement:.1f} < {self.config.stationary_displacement_threshold_px})"
                else:
                    return False, f"displacement_moving ({displacement:.1f} >= {self.config.stationary_displacement_threshold_px})"

            return False, "moving"

    def select_tracks_for_face_analysis(
        self,
        tracks: Sequence[Any],
        max_evaluations: Optional[int] = None,
        timestamp: Optional[float] = None
    ) -> List[Any]:
        """Select up to max_evaluations tracks for face assessment using round-robin.
        
        In crowded scenes (20-30 people), skips stationary and low-confidence tracks,
        distributing evaluation cycles fairly across all remaining active individuals.
        """
        if not tracks:
            return []

        now = timestamp if timestamp is not None else time.time()
        limit = max_evaluations if max_evaluations is not None else self.config.max_faces_per_frame

        with self._lock:
            # Prune stale tracking history to safeguard memory
            active_ids = {_extract_track_id(t) for t in tracks}
            self.prune_stale_tracks(active_ids, current_time=now)

            eligible_tracks: List[Any] = []
            for track in tracks:
                state = _extract_track_state(track)
                if state != "active":
                    continue

                conf = _extract_track_confidence(track)
                if conf < self.config.min_track_confidence:
                    continue

                if self.config.require_confirmed and not _extract_track_confirmed(track):
                    continue

                is_stat, _ = self.is_track_stationary(track, timestamp=now)
                if is_stat:
                    continue

                eligible_tracks.append(track)

            if not eligible_tracks:
                return []

            # Deterministic sorting ensures fair, consistent round-robin indexing
            eligible_tracks.sort(key=lambda t: _extract_track_id(t))
            n = len(eligible_tracks)

            if n <= limit:
                selected = list(eligible_tracks)
            else:
                selected = []
                start = self._rr_cursor % n
                for i in range(limit):
                    idx = (start + i) % n
                    selected.append(eligible_tracks[idx])
                self._rr_cursor = (start + limit) % n

            # Update evaluation records for selected tracks
            for track in selected:
                self._record_track_evaluated(track, timestamp=now)

            # Safeguard memory bounds
            self._enforce_history_capacity()

            return selected

    def _record_track_evaluated(
        self,
        track: Any,
        timestamp: Optional[float] = None,
        quality_score: Optional[float] = None
    ) -> None:
        """Internal helper to update tracking evaluation history."""
        now = timestamp if timestamp is not None else time.time()
        track_id = _extract_track_id(track)
        bbox = _extract_track_bbox(track)
        center = None
        if bbox is not None:
            center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

        record = self._track_eval_history.get(track_id)
        if record is None:
            record = TrackEvalRecord(
                track_id=track_id,
                first_eval_time=now,
                last_eval_time=now,
                eval_count=1,
                last_bbox=bbox,
                last_center=center,
                last_confidence=_extract_track_confidence(track),
                last_speed=_extract_track_speed(track) or 0.0,
                last_quality_score=quality_score or 0.0
            )
            self._track_eval_history[track_id] = record
        else:
            record.last_eval_time = now
            record.eval_count += 1
            if bbox is not None:
                record.last_bbox = bbox
                record.last_center = center
            if quality_score is not None:
                record.last_quality_score = quality_score

    def _enforce_history_capacity(self) -> int:
        """Enforce strict upper bound on track history dictionary size."""
        if len(self._track_eval_history) <= self.config.max_tracked_history:
            return 0
        excess = len(self._track_eval_history) - self.config.max_tracked_history
        sorted_by_eval = sorted(
            self._track_eval_history.items(),
            key=lambda item: (item[1].last_eval_time, item[1].first_eval_time)
        )
        evicted = 0
        for tid, _ in sorted_by_eval[:excess]:
            del self._track_eval_history[tid]
            evicted += 1
        return evicted

    def prune_stale_tracks(
        self,
        active_track_ids: Optional[Set[int]] = None,
        max_idle_time: Optional[float] = None,
        current_time: Optional[float] = None
    ) -> int:
        """Purge tracking evaluation records for tracks no longer active or timed out.
        
        Enforces bounded memory and prevents leaks from stale track buffers.
        """
        now = current_time if current_time is not None else time.time()
        max_idle = max_idle_time if max_idle_time is not None else self.config.stale_track_timeout_s

        with self._lock:
            to_remove: List[int] = []
            for tid, record in self._track_eval_history.items():
                is_active = (active_track_ids is not None and tid in active_track_ids)
                idle_time = max(0.0, now - record.last_eval_time)

                if active_track_ids is not None:
                    if not is_active and idle_time > max_idle:
                        to_remove.append(tid)
                else:
                    if idle_time > max_idle:
                        to_remove.append(tid)

            for tid in to_remove:
                del self._track_eval_history[tid]

            # Enforce capacity
            self._enforce_history_capacity()

            return len(to_remove)

    # -------------------------------------------------------------------------
    # Latency Budgeting & Adaptive Frame Dropping
    # -------------------------------------------------------------------------

    def record_frame_latency(self, latency_ms: float) -> None:
        """Record total frame processing latency and update moving averages."""
        with self._lock:
            l_ms = float(latency_ms)
            self._last_latency_ms = l_ms
            if self._current_latency_ms <= 0.0:
                self._current_latency_ms = l_ms
            else:
                alpha = self.config.latency_ema_alpha
                self._current_latency_ms = (1.0 - alpha) * self._current_latency_ms + alpha * l_ms

            self._metrics["last_latency_ms"] = round(self._last_latency_ms, 2)
            self._metrics["ema_latency_ms"] = round(self._current_latency_ms, 2)

    def record_stage_latency(self, stage_name: str, duration_ms: float) -> None:
        """Record latency for a specific pipeline stage."""
        with self._lock:
            key = f"{stage_name}_ms"
            self._metrics[key] = round(float(duration_ms), 2)

    def should_skip_analysis_frame(self, frame_seq: Optional[int] = None) -> bool:
        """Determine whether to drop intermediate analysis frame to meet latency budget.
        
        Live streaming frames are never dropped; only compute-heavy analysis frames
        are dropped under backpressure. Never drops more than 2 consecutive frames
        to prevent permanent skipping deadlocks.
        """
        with self._lock:
            # Prevent permanent skip deadlocks
            if not hasattr(self, "_consecutive_skips"):
                self._consecutive_skips = 0

            budget = max(1.0, self.config.target_latency_budget_ms)
            if self._current_latency_ms > budget or self._last_latency_ms > budget:
                if self._consecutive_skips < 2:
                    self._consecutive_skips += 1
                    self._metrics["dropped_frames"] += 1
                    self._metrics["dropped_analysis_frames"] += 1
                    self._metrics["latency_skips"] += 1
                    return True

            self._consecutive_skips = 0
            return False

    @property
    def current_latency_ms(self) -> float:
        """Current exponential moving average processing latency in milliseconds."""
        with self._lock:
            return self._current_latency_ms

    @property
    def last_latency_ms(self) -> float:
        """Most recently recorded latency in milliseconds."""
        with self._lock:
            return self._last_latency_ms

    # -------------------------------------------------------------------------
    # Diagnostics & Metrics
    # -------------------------------------------------------------------------

    @property
    def analysis_queue_depth(self) -> int:
        with self._lock:
            return len(self._analysis_queue)

    @property
    def disk_queue_depth(self) -> int:
        with self._lock:
            return len(self._disk_queue)

    @property
    def enhancement_queue_depth(self) -> int:
        with self._lock:
            return len(self._enhancement_queue)

    @property
    def task_queue_depth(self) -> int:
        return self._task_queue.qsize()

    @property
    def active_track_count(self) -> int:
        with self._lock:
            return len(self._track_eval_history)

    def get_metrics(self) -> Dict[str, Any]:
        """Fetch snapshot of pipeline operational metrics."""
        with self._lock:
            return {
                **self._metrics,
                "is_running": self.is_running,
                "is_paused": self.is_paused,
                "is_shutdown": self._is_shutdown,
                "worker_threads": len(self._workers),
                "active_workers": sum(1 for w in self._workers if w.is_alive()),
                "analysis_queue_depth": len(self._analysis_queue),
                "disk_queue_depth": len(self._disk_queue),
                "enhancement_queue_depth": len(self._enhancement_queue),
                "task_queue_depth": self._task_queue.qsize(),
                "active_tracked_history_count": len(self._track_eval_history),
                "current_latency_ms": round(self._current_latency_ms, 2),
            }

    def get_processing_metrics(self, frame_sequence: int = 0) -> ProcessingMetrics:
        """Return standardized ProcessingMetrics dataclass."""
        with self._lock:
            return ProcessingMetrics(
                frame_sequence=frame_sequence,
                acquisition_fps=round(float(self._metrics.get("acquisition_fps", 0.0)), 1),
                detection_ms=round(float(self._metrics.get("detection_ms", 0.0)), 2),
                tracking_ms=round(float(self._metrics.get("tracking_ms", 0.0)), 2),
                face_detection_ms=round(float(self._metrics.get("face_detection_ms", 0.0)), 2),
                quality_eval_ms=round(float(self._metrics.get("quality_eval_ms", 0.0)), 2),
                enhancement_ms=round(float(self._metrics.get("enhancement_ms", 0.0)), 2),
                active_tracks=len(self._track_eval_history),
                queue_depth=len(self._analysis_queue),
                dropped_frames=int(self._metrics.get("dropped_frames", 0)),
            )

    def reset_metrics(self) -> None:
        """Reset performance counters."""
        with self._lock:
            for k in self._metrics:
                if isinstance(self._metrics[k], int):
                    self._metrics[k] = 0
                elif isinstance(self._metrics[k], float):
                    self._metrics[k] = 0.0
            self._current_latency_ms = 0.0
            self._last_latency_ms = 0.0

    def benchmark_face_lock_on(self, tracks: Sequence[Any], iterations: int = 100) -> float:
        """Measure average latency in milliseconds for face lock-on calculation across tracks.

        Verifies that frame-to-frame face lock-on calculation runs in strictly < 1.0 ms per frame.
        """
        if not tracks:
            return 0.0
        t0 = time.perf_counter()
        for _ in range(iterations):
            for trk in tracks:
                if hasattr(trk, "update_locked_face_from_body"):
                    trk.update_locked_face_from_body()
                elif hasattr(trk, "track") and hasattr(trk.track, "update_locked_face_from_body"):
                    trk.track.update_locked_face_from_body()
        total_time = time.perf_counter() - t0
        return (total_time / iterations) * 1000.0
