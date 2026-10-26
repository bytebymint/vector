"""Live video pipeline: frame acquisition, ring-buffered storage, and downstream distribution.

Preserves full original resolution (e.g. 1080p, 4K) for high-fidelity facial capture,
maintains bounded thread-safe ring buffers, prevents redundant video decoding,
and distributes frames to multiple downstream consumers without blocking acquisition.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union
import uuid

import cv2
import numpy as np

try:
    from services.face_contracts import FramePacket, ProcessingMetrics
except ImportError:
    from .face_contracts import FramePacket, ProcessingMetrics

LOG = logging.getLogger(__name__)


def extract_crop_coordinates(
    packet: FramePacket,
    bbox: Union[Tuple[float, float, float, float], List[float], np.ndarray],
    margin: float = 0.2,
    preview_shape: Optional[Tuple[int, int]] = None,
    preview_size: Optional[Tuple[int, int]] = None,
    normalized: Optional[bool] = None,
) -> Tuple[int, int, int, int]:
    """Calculate clamped pixel bounding box (x1, y1, x2, y2) on original high-res frame.

    Args:
        packet: Full-resolution FramePacket instance.
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        margin: Fractional expansion margin applied to width and height (e.g. 0.2 = 20%).
        preview_shape: (height, width) of resized preview if bbox originated from preview.
        preview_size: (width, height) of resized preview if bbox originated from preview.
        normalized: Whether bbox coordinates are in [0.0, 1.0]. Auto-detected if None.

    Returns:
        Clamped integer tuple (x1, y1, x2, y2) within [0, packet.width] and [0, packet.height].
    """
    if packet is None or packet.image is None:
        return 0, 0, 0, 0

    img_h, img_w = packet.image.shape[:2]
    # Synchronize with packet attributes if given
    orig_w = packet.width if packet.width > 0 else img_w
    orig_h = packet.height if packet.height > 0 else img_h

    if len(bbox) != 4:
        raise ValueError(f"bbox must contain exactly 4 coordinates, got length {len(bbox)}")

    raw_x1, raw_y1, raw_x2, raw_y2 = (
        float(bbox[0]),
        float(bbox[1]),
        float(bbox[2]),
        float(bbox[3]),
    )

    x_min, x_max = min(raw_x1, raw_x2), max(raw_x1, raw_x2)
    y_min, y_max = min(raw_y1, raw_y2), max(raw_y1, raw_y2)

    # Determine if normalized coordinates were passed
    if normalized is True:
        is_norm = True
    elif normalized is False:
        is_norm = False
    else:
        # Auto-detect normalized: all coordinates in [0, 1] and non-zero span
        is_norm = (
            0.0 <= x_min <= 1.0
            and 0.0 <= x_max <= 1.0
            and 0.0 <= y_min <= 1.0
            and 0.0 <= y_max <= 1.0
            and (x_max > 0.0 or y_max > 0.0)
            and (orig_w > 1 and orig_h > 1)
        )

    if is_norm:
        bx1 = x_min * orig_w
        bx2 = x_max * orig_w
        by1 = y_min * orig_h
        by2 = y_max * orig_h
    elif preview_shape is not None:
        prev_h = float(preview_shape[0])
        prev_w = float(preview_shape[1])
        scale_x = orig_w / prev_w if prev_w > 0 else 1.0
        scale_y = orig_h / prev_h if prev_h > 0 else 1.0
        bx1 = x_min * scale_x
        bx2 = x_max * scale_x
        by1 = y_min * scale_y
        by2 = y_max * scale_y
    elif preview_size is not None:
        prev_w = float(preview_size[0])
        prev_h = float(preview_size[1])
        scale_x = orig_w / prev_w if prev_w > 0 else 1.0
        scale_y = orig_h / prev_h if prev_h > 0 else 1.0
        bx1 = x_min * scale_x
        bx2 = x_max * scale_x
        by1 = y_min * scale_y
        by2 = y_max * scale_y
    else:
        bx1, bx2, by1, by2 = x_min, x_max, y_min, y_max

    bw = max(0.0, bx2 - bx1)
    bh = max(0.0, by2 - by1)
    m = max(0.0, float(margin))
    dx = bw * m
    dy = bh * m

    exp_x1 = bx1 - dx
    exp_y1 = by1 - dy
    exp_x2 = bx2 + dx
    exp_y2 = by2 + dy

    ix1 = max(0, min(orig_w, int(np.floor(exp_x1))))
    iy1 = max(0, min(orig_h, int(np.floor(exp_y1))))
    ix2 = max(0, min(orig_w, int(np.ceil(exp_x2))))
    iy2 = max(0, min(orig_h, int(np.ceil(exp_y2))))

    return ix1, iy1, ix2, iy2


def extract_crop(
    packet: FramePacket,
    bbox: Union[Tuple[float, float, float, float], List[float], np.ndarray],
    margin: float = 0.2,
    preview_shape: Optional[Tuple[int, int]] = None,
    preview_size: Optional[Tuple[int, int]] = None,
    normalized: Optional[bool] = None,
) -> np.ndarray:
    """Extract a high-resolution sub-image crop from a FramePacket with margin and bounds clamping.

    Args:
        packet: Full-resolution FramePacket containing the master uncompressed image.
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        margin: Fractional margin around detection box (default 0.2 = 20% expansion on each side).
        preview_shape: Optional (height, width) if detection coordinates originated from preview image.
        preview_size: Optional (width, height) if detection coordinates originated from preview image.
        normalized: Optional flag forcing normalized or absolute coordinate interpretation.

    Returns:
        np.ndarray cropped from master uncompressed frame.
    """
    if packet is None or packet.image is None:
        return np.empty((0, 0, 3), dtype=np.uint8)

    ix1, iy1, ix2, iy2 = extract_crop_coordinates(
        packet=packet,
        bbox=bbox,
        margin=margin,
        preview_shape=preview_shape,
        preview_size=preview_size,
        normalized=normalized,
    )

    if ix2 <= ix1 or iy2 <= iy1:
        channels = packet.image.shape[2] if packet.image.ndim == 3 else 0
        empty_shape = (0, 0, channels) if channels > 0 else (0, 0)
        return np.empty(empty_shape, dtype=packet.image.dtype)

    return packet.image[iy1:iy2, ix1:ix2].copy()


class RingBuffer:
    """Thread-safe bounded circular frame buffer with FIFO drop behavior."""

    def __init__(self, maxlen: int = 10):
        if maxlen <= 0:
            raise ValueError(f"RingBuffer maxlen must be positive, got {maxlen}")
        self._maxlen = maxlen
        self._buffer: deque[FramePacket] = deque(maxlen=maxlen)
        self._lock = threading.RLock()
        self._total_pushed = 0
        self._dropped_count = 0

    def append(self, packet: FramePacket) -> None:
        """Append packet. Drops oldest frame if capacity reached."""
        with self._lock:
            if len(self._buffer) == self._maxlen:
                self._dropped_count += 1
            self._buffer.append(packet)
            self._total_pushed += 1

    def get_latest(self) -> Optional[FramePacket]:
        """Return the most recent FramePacket, or None if buffer is empty."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def get_by_sequence(self, sequence: int) -> Optional[FramePacket]:
        """Find frame packet by its sequence ID."""
        with self._lock:
            for packet in reversed(self._buffer):
                if packet.sequence == sequence:
                    return packet
            return None

    def get_recent(self, count: int = 1) -> List[FramePacket]:
        """Return the last `count` frames in chronological order."""
        with self._lock:
            if count <= 0:
                return []
            return list(self._buffer)[-count:]

    def clear(self) -> None:
        """Clear all frames in buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def capacity(self) -> int:
        return self._maxlen

    @property
    def total_pushed(self) -> int:
        with self._lock:
            return self._total_pushed

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count


class FrameConsumerQueue:
    """Thread-safe bounded queue for downstream consumers.

    Ensures that acquisition never blocks even if the consumer is slow or delayed.
    When full, either drops the oldest unread frame (default) or drops the incoming frame.
    """

    def __init__(self, name: str, max_size: int = 5, drop_oldest: bool = True):
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.name = name
        self.max_size = max_size
        self.drop_oldest = drop_oldest
        self._queue: deque[FramePacket] = deque()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._dropped_count = 0
        self._total_received = 0
        self._consumed_count = 0
        self._is_closed = False

    def put(self, packet: FramePacket) -> bool:
        """Non-blocking put for acquisition thread. Never blocks!

        Returns:
            True if queued without drop; False if a frame was dropped.
        """
        with self._lock:
            if self._is_closed:
                return False

            self._total_received += 1
            dropped = False

            if len(self._queue) >= self.max_size:
                self._dropped_count += 1
                dropped = True
                if self.drop_oldest:
                    self._queue.popleft()
                    self._queue.append(packet)
                # If drop_oldest is False, incoming packet is discarded
            else:
                self._queue.append(packet)

            self._condition.notify()
            return not dropped

    def get(self, block: bool = True, timeout: Optional[float] = None) -> FramePacket:
        """Retrieve next frame packet. Raises queue.Empty on timeout or non-blocking empty."""
        with self._lock:
            if not block:
                if not self._queue:
                    raise queue.Empty
                self._consumed_count += 1
                return self._queue.popleft()

            if timeout is None:
                while not self._queue and not self._is_closed:
                    self._condition.wait()
            else:
                end_time = time.time() + timeout
                while not self._queue and not self._is_closed:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)

            if not self._queue:
                raise queue.Empty

            self._consumed_count += 1
            return self._queue.popleft()

    def get_nowait(self) -> FramePacket:
        """Non-blocking retrieval."""
        return self.get(block=False)

    def get_optional(self, timeout: Optional[float] = None) -> Optional[FramePacket]:
        """Convenience method returning None instead of raising queue.Empty."""
        try:
            return self.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0

    def qsize(self) -> int:
        with self._lock:
            return len(self._queue)

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
            self._condition.notify_all()

    def close(self) -> None:
        """Close the consumer queue and unblock any waiting consumer."""
        with self._lock:
            self._is_closed = True
            self._condition.notify_all()

    def iter_frames(self, timeout: float = 1.0) -> Generator[FramePacket, None, None]:
        """Convenience generator yielding frame packets until closed."""
        while not self.is_closed:
            frame = self.get_optional(timeout=timeout)
            if frame is not None:
                yield frame

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._is_closed

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    @property
    def total_received(self) -> int:
        with self._lock:
            return self._total_received

    @property
    def consumed_count(self) -> int:
        with self._lock:
            return self._consumed_count


class VideoPipeline:
    """Live Video Pipeline for People Facial Capture System.

    Responsibilities:
    - Master resolution preservation (1080p, 4K, native camera stream).
    - Timestamped FramePacket packaging with timing metadata.
    - Thread-safe bounded ring buffer (default maxlen=10).
    - Unblocked frame distribution to multiple named consumer queues.
    - High-resolution crop extraction with margin clamping and preview scaling.
    - Redundant decoding avoidance via central shared frame distribution.
    """

    def __init__(
        self,
        source_id: str = "camera_default",
        buffer_size: int = 10,
        epoch: Optional[str] = None,
    ):
        self.source_id = str(source_id)
        self.epoch = str(epoch or uuid.uuid4())
        self.ring_buffer = RingBuffer(maxlen=buffer_size)
        self._consumers: Dict[str, FrameConsumerQueue] = {}
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._sequence = 0
        self._total_published = 0
        self._fps_window: deque[float] = deque(maxlen=60)
        self._worker_thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._is_closed = False

    def publish_frame(
        self,
        image: np.ndarray,
        timestamp: Optional[float] = None,
        pts: Optional[float] = None,
        sequence: Optional[int] = None,
    ) -> FramePacket:
        """Publish an uncompressed native-resolution frame.

        Enforces FramePacket contract, updates bounded ring buffer, and distributes
        non-blockingly to all registered consumers.

        Args:
            image: Original unscaled numpy array (H, W, C) or (H, W).
            timestamp: Optional epoch timestamp in seconds (default time.time()).
            pts: Optional presentation timestamp from stream/container.
            sequence: Optional explicit sequence number; increments monotonically if None.

        Returns:
            The published FramePacket instance.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be a numpy.ndarray, got {type(image)}")
        if image.ndim < 2:
            raise ValueError(f"image must have at least 2 dimensions, got shape {image.shape}")

        height, width = image.shape[:2]
        now = time.time() if timestamp is None else float(timestamp)

        with self._lock:
            if sequence is None:
                self._sequence += 1
                seq = self._sequence
            else:
                seq = int(sequence)
                self._sequence = max(self._sequence, seq)

            self._total_published += 1
            self._fps_window.append(now)

            packet = FramePacket(
                sequence=seq,
                timestamp=now,
                image=image,
                source_id=self.source_id,
                epoch=self.epoch,
                width=width,
                height=height,
                pts=pts,
            )

            # Store master uncompressed frame in ring buffer
            self.ring_buffer.append(packet)

            # Distribute non-blockingly to all registered consumer queues
            for consumer in list(self._consumers.values()):
                consumer.put(packet)

            self._condition.notify_all()
            return packet

    def get_latest_frame(self) -> Optional[FramePacket]:
        """Thread-safe accessor for the latest uncompressed native-resolution FramePacket."""
        return self.ring_buffer.get_latest()

    def wait_for_frame(
        self,
        after_sequence: int = -1,
        timeout: Optional[float] = 1.0,
    ) -> Optional[FramePacket]:
        """Wait for a new FramePacket with sequence > after_sequence."""
        with self._lock:
            def _predicate():
                latest = self.ring_buffer.get_latest()
                return (latest is not None and latest.sequence > after_sequence) or self._is_closed

            success = self._condition.wait_for(_predicate, timeout=timeout)
            if not success or self._is_closed:
                latest = self.ring_buffer.get_latest()
                if latest is not None and latest.sequence > after_sequence:
                    return latest
                return None
            return self.ring_buffer.get_latest()

    def register_consumer(
        self,
        name: str,
        max_size: int = 5,
        drop_oldest: bool = True,
    ) -> FrameConsumerQueue:
        """Register a named, bounded consumer queue for downstream processing.

        Args:
            name: Identifier for the consumer (e.g. "yolo_detector", "face_crop_worker").
            max_size: Maximum queue capacity before drop behavior takes effect.
            drop_oldest: True to drop oldest frame on overflow (ensuring low latency).

        Returns:
            FrameConsumerQueue instance.
        """
        with self._lock:
            if name in self._consumers:
                existing = self._consumers[name]
                if existing.max_size == max_size and existing.drop_oldest == drop_oldest and not existing.is_closed:
                    return existing
                existing.close()

            consumer = FrameConsumerQueue(name=name, max_size=max_size, drop_oldest=drop_oldest)
            self._consumers[name] = consumer
            return consumer

    def unregister_consumer(self, name: str) -> bool:
        """Unregister and close a downstream consumer queue."""
        with self._lock:
            consumer = self._consumers.pop(name, None)
            if consumer is not None:
                consumer.close()
                return True
            return False

    def get_consumer(self, name: str) -> Optional[FrameConsumerQueue]:
        """Retrieve registered consumer queue by name."""
        with self._lock:
            return self._consumers.get(name)

    def list_consumers(self) -> List[str]:
        """List all active consumer names."""
        with self._lock:
            return list(self._consumers.keys())

    def extract_crop(
        self,
        packet: FramePacket,
        bbox: Union[Tuple[float, float, float, float], List[float], np.ndarray],
        margin: float = 0.2,
        preview_shape: Optional[Tuple[int, int]] = None,
        preview_size: Optional[Tuple[int, int]] = None,
        normalized: Optional[bool] = None,
    ) -> np.ndarray:
        """High-resolution crop extraction helper.

        Ensures valid bounds clamping and scales bounding box if originated
        from a resized preview.
        """
        return extract_crop(
            packet=packet,
            bbox=bbox,
            margin=margin,
            preview_shape=preview_shape,
            preview_size=preview_size,
            normalized=normalized,
        )

    def get_preview_frame(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        target_max_dim: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Generate a lightweight preview copy without altering master uncompressed frame.

        Args:
            target_size: Optional explicit (width, height).
            target_max_dim: Optional maximum bounding box dimension.

        Returns:
            Resized copy of latest image, or None if no frames acquired.
        """
        latest = self.get_latest_frame()
        if latest is None or latest.image is None:
            return None

        img = latest.image
        h, w = img.shape[:2]

        if target_size is not None:
            tw, th = target_size
            return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

        if target_max_dim is not None and max(h, w) > target_max_dim:
            scale = target_max_dim / float(max(h, w))
            tw = max(1, int(round(w * scale)))
            th = max(1, int(round(h * scale)))
            return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

        return img.copy()

    @property
    def fps(self) -> float:
        """Estimated acquisition frame rate based on recent frame arrivals."""
        with self._lock:
            if len(self._fps_window) < 2:
                return 0.0
            dt = self._fps_window[-1] - self._fps_window[0]
            if dt <= 0.0:
                return 0.0
            return (len(self._fps_window) - 1) / dt

    def get_metrics(self) -> Dict[str, Any]:
        """Comprehensive operational metrics."""
        with self._lock:
            latest = self.ring_buffer.get_latest()
            consumer_stats = {
                name: {
                    "qsize": c.qsize(),
                    "max_size": c.max_size,
                    "dropped": c.dropped_count,
                    "total_received": c.total_received,
                    "consumed": c.consumed_count,
                }
                for name, c in self._consumers.items()
            }
            return {
                "source_id": self.source_id,
                "epoch": self.epoch,
                "sequence": self._sequence,
                "total_published": self._total_published,
                "acquisition_fps": round(self.fps, 1),
                "buffer_size": len(self.ring_buffer),
                "buffer_capacity": self.ring_buffer.capacity,
                "buffer_dropped": self.ring_buffer.dropped_count,
                "active_consumers": len(self._consumers),
                "consumer_stats": consumer_stats,
                "resolution": [latest.width, latest.height] if latest else None,
            }

    def get_processing_metrics(self) -> ProcessingMetrics:
        """Operational metrics formatted as ProcessingMetrics contract."""
        with self._lock:
            max_depth = max([c.qsize() for c in self._consumers.values()], default=0)
            total_drops = self.ring_buffer.dropped_count + sum(
                c.dropped_count for c in self._consumers.values()
            )
            return ProcessingMetrics(
                frame_sequence=self._sequence,
                acquisition_fps=round(self.fps, 1),
                queue_depth=max_depth,
                dropped_frames=total_drops,
            )

    def reset(self, epoch: Optional[str] = None) -> None:
        """Reset sequence counter, ring buffer, and queues for a new session."""
        with self._lock:
            self.epoch = str(epoch or uuid.uuid4())
            self._sequence = 0
            self._total_published = 0
            self._fps_window.clear()
            self.ring_buffer.clear()
            for consumer in self._consumers.values():
                consumer.clear()

    def start_ingestion(
        self,
        source: Any,
        fps: Optional[float] = None,
    ) -> None:
        """Start a background ingestion thread from a video stream, device, or iterable."""
        with self._lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._stopping.clear()
            self._worker_thread = threading.Thread(
                target=self._ingestion_worker,
                args=(source, fps),
                name=f"VideoIngest-{self.source_id}",
                daemon=True,
            )
            self._worker_thread.start()

    def stop_ingestion(self, timeout: float = 2.0) -> None:
        """Signal and stop background ingestion worker thread."""
        self._stopping.set()
        with self._lock:
            thread = self._worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    @property
    def is_ingesting(self) -> bool:
        """Check if background ingestion worker thread is running."""
        with self._lock:
            return self._worker_thread is not None and self._worker_thread.is_alive()

    def _ingestion_worker(self, source: Any, target_fps: Optional[float] = None) -> None:
        interval = (1.0 / target_fps) if target_fps and target_fps > 0 else None

        # Case 1: Iterable or generator yielding images or (image, timestamp) tuples
        if hasattr(source, "__iter__") and not isinstance(source, (str, int)):
            for item in source:
                if self._stopping.is_set():
                    break
                t0 = time.time()
                if isinstance(item, np.ndarray):
                    self.publish_frame(item)
                elif isinstance(item, (tuple, list)) and len(item) >= 2:
                    img = item[0]
                    stamp = item[1]
                    pts = item[2] if len(item) > 2 else None
                    self.publish_frame(img, timestamp=stamp, pts=pts)
                if interval:
                    elapsed = time.time() - t0
                    sleep_time = interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            return

        # Case 2: OpenCV VideoCapture or URL/device index
        cap = None
        opened_internally = False
        if isinstance(source, cv2.VideoCapture):
            cap = source
        elif isinstance(source, (str, int)):
            cap = cv2.VideoCapture(source)
            opened_internally = True

        if cap is not None:
            try:
                while not self._stopping.is_set() and cap.isOpened():
                    t0 = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        break
                    pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    self.publish_frame(frame, pts=pts)
                    if interval:
                        elapsed = time.time() - t0
                        sleep_time = interval - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
            finally:
                if opened_internally:
                    cap.release()

    def close(self) -> None:
        """Close pipeline, stopping ingestion and releasing consumer queues."""
        self.stop_ingestion()
        with self._lock:
            self._is_closed = True
            for consumer in list(self._consumers.values()):
                consumer.close()
            self._consumers.clear()
            self._condition.notify_all()

    def __enter__(self) -> VideoPipeline:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
