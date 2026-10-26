"""Bounded analytics decoder independent of native packet presentation."""
from __future__ import annotations
import logging
import queue
import threading
import time

LOG = logging.getLogger(__name__)


class AnalyticsDecoder:
    """Decode interframes in order, materializing only recent native-resolution samples.

    If CPU decoding falls behind, recover at a source keyframe. Native browser
    packets keep flowing even while analytics skips a GOP to catch up.
    """
    def __init__(self, publish, max_fps=24, max_pending=90):
        self.publish = publish
        self.interval = 1.0 / max(1, max_fps)
        self.pending = queue.Queue(maxsize=max_pending)
        self.stopping = threading.Event()
        self.resync = threading.Event()
        self.skipped = 0
        self.decode_ms = 0.0
        self.materialize_ms = 0.0
        self.decoded_frames = 0
        self.published_frames = 0
        self.copy_ema = 0.0
        self.thread = threading.Thread(target=self._run, name='native-analytics-decoder', daemon=True)
        self.thread.start()

    def submit(self, packet, wall_offset, due):
        try:
            self.pending.put_nowait((packet, wall_offset, due))
        except queue.Full:
            self.skipped += 1
            self.resync.set()

    def _run(self):
        seeking = True
        last_image = None
        while not self.stopping.is_set():
            try:
                packet, wall_offset, due = self.pending.get(timeout=.1)
            except queue.Empty:
                continue
            if self.resync.is_set() or time.monotonic() - due > .8:
                seeking = True
                self.resync.clear()
            if seeking:
                if not packet.is_keyframe or time.monotonic() - due > .8:
                    self.skipped += 1
                    continue
                packet.stream.codec_context.flush_buffers()
                seeking = False
                last_image = None
            try:
                started = time.perf_counter()
                frames = packet.decode()
                self.decode_ms = (time.perf_counter()-started)*1000
                for frame in frames:
                    self.decoded_frames += 1
                    media = float(frame.pts * frame.time_base)
                    sample_interval = max(self.interval, min(.12, self.copy_ema * 1.15))
                    if last_image is not None and media - last_image < sample_interval - .001:
                        continue
                    started = time.perf_counter()
                    self.publish(frame, wall_offset)
                    self.materialize_ms = (time.perf_counter()-started)*1000
                    self.copy_ema = .8*self.copy_ema + .2*self.materialize_ms/1000
                    self.published_frames += 1
                    last_image = media
            except Exception as exc:
                seeking = True
                LOG.warning('Analytics decoder awaiting next keyframe: %s', type(exc).__name__)

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=3)

    def metrics(self):
        return dict(decode_ms=round(self.decode_ms,2), materialize_ms=round(self.materialize_ms,2),
                    decoded_frames=self.decoded_frames, published_frames=self.published_frames,
                    skipped_packets=self.skipped, pending=self.pending.qsize())
