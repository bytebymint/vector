"""Light browser packet relay sharing the native AI source's presentation clock.

The preview never decodes, resizes or re-encodes frames. Only a second rendition
of the same YouTube extraction is opened; the original AI/capture source stays
untouched. No URLs, headers or cookies are exposed through status/logs.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import logging
import threading
import time
import av

from services.native_video import NativeMuxer

LOG=logging.getLogger(__name__)


@dataclass(frozen=True)
class ClockSnapshot:
    pts: float
    dts: float
    monotonic: float
    wall: float
    revision: int = 0

    def due(self, dts: float) -> float:
        return self.monotonic+dts-self.dts

    def timestamp(self, pts: float) -> float:
        return self.wall+pts-self.pts


class SourceClock:
    def __init__(self):
        self.lock=threading.Lock()
        self.value: ClockSnapshot | None=None
        self.ready=threading.Event()

    def start(self, pts: float, dts: float, monotonic: float, wall: float):
        with self.lock:
            self.value=ClockSnapshot(pts,dts,monotonic,wall)
            self.ready.set()

    def snapshot(self) -> ClockSnapshot | None:
        with self.lock:
            return self.value

    def shift(self, delay: float):
        with self.lock:
            if self.value is not None:
                value=self.value
                self.value=replace(value,monotonic=value.monotonic+delay,
                                   wall=value.wall+delay,revision=value.revision+1)


class PreviewStream:
    def __init__(self, source: dict, relay, clock: SourceClock, epoch: str, stopping,
                 max_lateness: float = .5, resume_position: float = 0.0, is_paused_fn=None):
        self.source=source
        self.relay=relay
        self.clock=clock
        self.epoch=epoch
        self.stopping=stopping
        self.max_lateness=max_lateness
        self.resume_position=float(resume_position or 0.0)
        self.is_paused_fn=is_paused_fn
        self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self._run,name='1080p-native-preview',daemon=True)
        self.state='starting'
        self.error=None
        self.skipped=0
        self.packets=0
        self.quality={key:source.get(key) for key in ('width','height','fps','vcodec','tbr','format_id')}

    def start(self):
        self.thread.start()
        return self

    def stopped(self):
        return self.stop_event.is_set() or self.stopping()

    def status(self):
        return dict(self.quality,state=self.state,error=self.error,
                    skipped_packets=self.skipped,packets_relayed=self.packets,
                    reencoded=False,separate_rendition=True)

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=15)

    def _run(self):
        sources=[self.source,*self.source.get('alternatives',[])]
        for source in sources:
            if self.stopped():return
            try:
                self._relay(source)
                if not self.stopped():self.state='ended'
                return
            except Exception as exc:
                self.error=type(exc).__name__
                self.state='retrying'
                LOG.warning('Preview rendition %s unavailable: %s',source.get('format_id','unknown'),self.error)
        self.state='failed'
        LOG.warning('1080p preview unavailable; reverting to source playback: %s',self.error)

    def _relay(self, source):
        muxer=None
        revision=None
        try:
            options={'rw_timeout':'8000000'}
            headers=source.get('http_headers') or {}
            # These are yt-dlp's trusted extraction headers, never dashboard input.
            if headers:
                options['headers']=''.join(f'{k}: {v}\r\n' for k,v in headers.items())
            with av.open(source['url'],options=options,timeout=(12,8)) as container:
                stream=container.streams.video[0]
                codec=stream.codec_context
                self.quality={key:source.get(key) for key in ('width','height','fps','vcodec','tbr','format_id')}
                self.quality.update(width=codec.width,height=codec.height,codec=codec.name,
                                    fps=float(stream.average_rate) if stream.average_rate else source.get('fps'))
                if self.resume_position > 0:
                    try:
                        target_pts = int(self.resume_position / stream.time_base)
                        container.seek(target_pts, backward=True, stream=stream)
                        LOG.info('Preview stream sought to %.2fs', self.resume_position)
                    except Exception as s_err:
                        LOG.warning('Preview stream seek failed: %s', s_err)
                # No packet.decode() call belongs in this presentation path.
                for packet in container.demux(stream):
                    if self.stopped():break
                    if not packet.size or packet.pts is None:continue
                    while not self.clock.ready.wait(.1):
                        if self.stopped():return
                    pts=float(packet.pts*packet.time_base)
                    dts=float((packet.dts if packet.dts is not None else packet.pts)*packet.time_base)
                    while not self.stopped():
                        if self.is_paused_fn and self.is_paused_fn():
                            self.stop_event.wait(0.05)
                            continue
                        clock=self.clock.snapshot()
                        if clock is None:continue
                        delay=clock.due(dts)-time.monotonic()
                        if delay<=0:break
                        self.stop_event.wait(min(.025,delay))
                    if self.stopped():break
                    clock=self.clock.snapshot()
                    if clock is None:
                        continue
                    if revision!=clock.revision:
                        if muxer is not None:
                            muxer.close()
                            muxer=None
                        revision=clock.revision
                    # Starting mid-GOP is unsafe. Catch up to a keyframe.
                    if muxer is None:
                        if not packet.is_keyframe:
                            self.skipped+=1
                            continue
                        muxer=NativeMuxer(self.relay,stream,packet,
                                          f'{self.epoch}-preview-{clock.revision}-{self.skipped}',
                                          clock.timestamp(pts),source.get('vcodec'))
                    try:
                        muxer.mux(packet)
                        self.packets+=1
                        self.state='live'
                        self.error=None
                    except Exception as m_err:
                        LOG.warning('Preview packet mux error: %s', m_err)
        finally:
            if muxer is not None:
                try:muxer.close()
                except Exception:pass
