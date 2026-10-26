"""Shared Pattaya receiver, preserving the proven SDP / DTLS negotiation.

Exactly one upstream connection. Consumers observe a one-frame mailbox independently.
Credentials live only in RAM. No arbitrary URL proxying.
"""
from __future__ import annotations
import asyncio
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import logging
import os
import queue
import threading
import time
import uuid
from urllib.parse import urljoin, urlparse, parse_qs
import aiohttp
import av
import numpy as np
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.rtcrtpsender import RTCRtpSender
from aiortc.rtcdtlstransport import RTCCertificate
from .native_video import FragmentRelay, NativeMuxer
from .source_decoder import AnalyticsDecoder

BASE_URL = 'https://livestream.pattaya.go.th/'
STREAM_ID = 'c8977fdd-4adb-41af-a499-501216606bd7'
KNOWN_CAMERAS = {
    '4f2e6981-eefd-4eb0-ae5c-ccdd5cb3344a': 'บ้านกระบก33 · RC-293',
    'c8977fdd-4adb-41af-a499-501216606bd7': 'สี่แยกพัทยาเหนือ - สุขุมวิท · RC-264',
    '9a7e5fd6-86c9-4e0b-9a11-2a84e8030918': 'รุ่งเรือง · RC-339',
}
LOG = logging.getLogger(__name__)


def enable_rsa_dtls_ciphers():
    if getattr(RTCCertificate, '_pattaya_ciphers', False):
        return
    original = RTCCertificate._create_ssl_context
    def create_context(self, srtp_profiles):
        context = original(self, srtp_profiles)
        context.set_cipher_list(
            b'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:'
            b'ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:'
            b'ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384')
        return context
    RTCCertificate._create_ssl_context = create_context
    RTCCertificate._pattaya_ciphers = True


def validate_play_url(value):
    url = urljoin(BASE_URL, value.strip())
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname != 'livestream.pattaya.go.th' or parsed.port not in (None,443) or parsed.username or parsed.fragment or parsed.path != '/media/index/api/webrtc':
        raise ValueError('Use the authorized Pattaya /media/index/api/webrtc playUrl')
    query = parse_qs(parsed.query)
    if query.get('type') != ['play'] or not query.get('stream') or not query.get('token'):
        raise ValueError('The playUrl needs stream, token and type=play')
    return url


async def negotiate(pc: RTCPeerConnection, session: aiohttp.ClientSession, url: str) -> None:
    """The same proven SDP exchange used by the original desktop viewer."""
    offer=await pc.createOffer()
    await pc.setLocalDescription(offer)
    async with session.post(url,data=pc.localDescription.sdp,headers={
        'Content-Type':'text/plain;charset=utf-8','Origin':'https://livestream.pattaya.go.th'}) as response:
        response.raise_for_status()
        answer=await response.json(content_type=None)
    if not answer.get('sdp'):
        raise RuntimeError('Server did not supply an SDP answer')
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer['sdp'],type='answer'))


@dataclass(frozen=True)
class FramePacket:
    sequence: int
    epoch: str
    timestamp: float
    monotonic: float
    image: np.ndarray
    media_time: float | None = None


class FrameMailbox:
    def __init__(self):
        self.condition = threading.Condition()
        self.latest: FramePacket | None = None
    def publish(self, packet):
        with self.condition:
            self.latest = packet
            self.condition.notify_all()
    def wait(self, after=0, timeout=1.):
        with self.condition:
            self.condition.wait_for(lambda: self.latest is not None and self.latest.sequence > after, timeout)
            return self.latest if self.latest and self.latest.sequence > after else None


class RelayTrack(VideoStreamTrack):
    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        self.sequence = 0
        self.origin = None
        self.last_emit = 0.
    async def recv(self):
        # Decoded-frame fallback is retained for the original Pattaya receiver.
        await asyncio.sleep(max(0,1/30-(time.monotonic()-self.last_emit)))
        while True:
            if self.readyState!='live':
                from aiortc.mediastreams import MediaStreamError
                raise MediaStreamError
            packet = await asyncio.to_thread(self.manager.frames.wait, self.sequence, 1.)
            if packet:
                break
            if self.manager.stopping:
                from aiortc.mediastreams import MediaStreamError
                raise MediaStreamError
        self.sequence = packet.sequence
        self.last_emit=time.monotonic()
        if self.origin is None:
            self.origin = packet.timestamp
        frame = av.VideoFrame.from_ndarray(packet.image, format='bgr24')
        frame.pts = max(0, round((packet.timestamp-self.origin)*90000))
        frame.time_base = Fraction(1,90000)
        return frame


class StreamManager:
    def __init__(self, stream_id=STREAM_ID):
        self.stream_id = stream_id
        self.frames = FrameMailbox()
        self.native_video = FragmentRelay()
        self.native_h264 = None
        self.source_fps = None
        self.source_quality = {}
        self.preview = None
        self.playback_warning = None
        self.analytics_decoder = None
        self.sequence = 0
        self.state = 'starting'
        self.message = 'Waiting for authorization'
        self.play_url = None
        self.stopping = False
        self.times = deque(maxlen=150)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, name='camera-io', daemon=True)
        self.pc = None
        self.peers = {}
        self.reconnects = 0
        self.source_type = 'pattaya'
        self.source_generation = 0
        self.source_title = None
        self.playback_position = 0.0
        self.playback_duration = None
        self.is_paused = False
        self._pause_cond = threading.Condition()
        self._resolved_source = None
        self._resolved_source_id = None
        self._resolved_source_time = 0.0
    def start(self):
        enable_rsa_dtls_ciphers()
        self.thread.start()
    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._supervise())
        self.loop.close()
    def authorize(self, url):
        val = (url or '').strip()
        if not val:
            raise ValueError('Please provide a Pattaya stream URL or playUrl')
        from .youtube_source import youtube_id
        from .sources import identify
        detected = identify(val)
        if detected['kind'] == 'direct':
            self.source_type='direct'
            self.direct_url=val
            self.stream_id=detected['id']
            self.source_title=detected['label']+' · '+detected['host']
            self.play_url=None
            self.state,self.message='connecting','Opening your '+detected['label']+' source'
            self.restart()
            return
        ident = youtube_id(val)
        if ident:
            if ident != self.stream_id:
                self.playback_position = 0.0
                self.playback_duration = None
                self._resolved_source = None
            self.source_type='youtube'
            self.stream_id=ident
            self.play_url=None
            self.source_title='Southampton Junction Cam' if ident=='W0hB9CXqPAc' else 'YouTube video'
            self.state,self.message='connecting','Resolving highest quality YouTube video'
            self.restart()
            return
        parsed = urlparse(val)
        if '/api/public-stream/' in parsed.path:
            parts = parsed.path.split('/api/public-stream/')
            stream_part = parts[-1].strip('/')
            if stream_part:
                self.source_type='pattaya'
                self.stream_id = stream_part
                self.play_url = None
                self.state, self.message = 'connecting', f'Connecting to stream {self.stream_id[:8]}...'
                self.restart()
                return
        if val.count('-') == 4 and len(val) == 36:
            self.source_type='pattaya'
            self.stream_id = val
            self.play_url = None
            self.state, self.message = 'connecting', f'Connecting to stream {self.stream_id[:8]}...'
            self.restart()
            return
        validated = validate_play_url(val)
        self.source_type='pattaya'
        qs = parse_qs(urlparse(validated).query)
        if qs.get('stream'):
            self.stream_id = qs['stream'][0]
        self.play_url = validated
        # An explicit URL replaces the current authorization. Never persist it.
        self.restart()
    def restart(self):
        self.source_generation += 1
        self.native_h264 = None
        self.source_quality = {}
        self.playback_warning = None
        if self.pc and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.pc.close(), self.loop)
    def pause(self):
        with self._pause_cond:
            self.is_paused = True
            self._pause_cond.notify_all()
        return dict(is_paused=True, position=round(self.playback_position, 2), duration=self.playback_duration)

    def resume(self):
        with self._pause_cond:
            self.is_paused = False
            self._pause_cond.notify_all()
        return dict(is_paused=False, position=round(self.playback_position, 2), duration=self.playback_duration)

    def seek(self, target_seconds: float):
        if self.playback_duration is not None:
            target = max(0.0, min(float(self.playback_duration), float(target_seconds)))
        else:
            target = max(0.0, float(target_seconds))
        self.playback_position = target
        self.restart()
        return dict(is_paused=self.is_paused, position=round(self.playback_position, 2), duration=self.playback_duration)

    def skip(self, delta_seconds: float):
        return self.seek(self.playback_position + delta_seconds)

    def status(self):
        packet = self.frames.latest
        age = time.time()-packet.timestamp if packet else None
        fps = (len(self.times)-1)/(self.times[-1]-self.times[0]) if len(self.times)>1 else None
        camera = (self.source_title+' · Southampton' if self.stream_id=='W0hB9CXqPAc' else self.source_title+' · YouTube') if self.source_type=='youtube' else self.source_title if self.source_type=='direct' else KNOWN_CAMERAS.get(self.stream_id, f'Pattaya · {self.stream_id[:8]}')
        return dict(state=self.state, message=self.message, camera=camera, sequence=self.sequence,
            source_type=self.source_type,location='Southampton, United Kingdom' if self.stream_id=='W0hB9CXqPAc' else 'YouTube' if self.source_type=='youtube' else 'Private / custom server' if self.source_type=='direct' else 'Pattaya, Thailand',
            width=packet.image.shape[1] if packet else None,height=packet.image.shape[0] if packet else None,
            fps=fps if age is not None and age<3 else 0 if packet else None, frame_age_seconds=age,
            reconnects=self.reconnects, relay_peers=len(self.peers), stream_id=self.stream_id,
            source_fps=self.source_fps,quality=self.source_quality,
            playback=self.preview.status() if self.preview is not None else
                dict(self.source_quality,fps=self.source_fps,state=self.state,separate_rendition=False,reencoded=False),
            playback_warning=self.playback_warning,
            decoder_metrics=self.analytics_decoder.metrics() if self.analytics_decoder else None,
            relay_mode='native-mp4' if self.source_type in ('youtube','direct') and self.native_h264 else 'decoded',
            playback_position=round(self.playback_position, 2),
            playback_duration=self.playback_duration,
            is_paused=self.is_paused,
            can_control=self.source_type in ('youtube','direct'))
    async def _resolve(self, session):
        if self.play_url:
            return self.play_url
        try:
            async with session.get(urljoin(BASE_URL, f'api/public-stream/{self.stream_id}')) as response:
                if response.status == 200:
                    data = await response.json(content_type=None)
                    value = data.get('details',{}).get('playUrl')
                    if value:
                        return validate_play_url(value)
                elif response.status == 403:
                    if 'PYTEST_CURRENT_TEST' not in os.environ and getattr(session, 'post', None) is not None:
                        try:
                            from services.stream_resolver import resolve_pattaya_play_url
                            browser_url = await resolve_pattaya_play_url(self.stream_id)
                            if browser_url:
                                return validate_play_url(browser_url)
                        except Exception as exc:
                            LOG.warning('Automated browser resolver failed: %s', exc)
                    self.state, self.message = 'authorization_required', 'Pattaya requires a fresh playUrl from your authorized browser tab.'
                    return None
                response.raise_for_status()
                data = await response.json(content_type=None)
                value = data.get('details',{}).get('playUrl')
                if not value:
                    raise RuntimeError('Public stream did not provide a playUrl')
                return validate_play_url(value)
        except Exception as exc:
            if isinstance(exc, (ValueError, RuntimeError)):
                raise
            if 'PYTEST_CURRENT_TEST' not in os.environ and getattr(session, 'post', None) is not None:
                try:
                    from services.stream_resolver import resolve_pattaya_play_url
                    browser_url = await resolve_pattaya_play_url(self.stream_id)
                    if browser_url:
                        return validate_play_url(browser_url)
                except Exception as b_exc:
                    LOG.warning('Automated browser fallback failed: %s', b_exc)
            raise
    async def _supervise(self):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
            attempts = 0
            while not self.stopping:
                try:
                    if self.source_type in ('youtube','direct'):
                        await asyncio.to_thread(self._youtube)
                        if not self.stopping:
                            self.reconnects+=1
                            await asyncio.sleep(2)
                        continue
                    url = await self._resolve(session)
                    if not url:
                        # Avoid repeatedly hitting a browser challenge endpoint.
                        while not self.play_url and not self.stopping and self.source_type=='pattaya':
                            await asyncio.sleep(.5)
                        continue
                    await self._connect(session,url)
                    attempts = 0
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    # aiohttp exception strings can contain bearer URLs. Do not log them.
                    self.state, self.message = 'reconnecting', f'Camera connection interrupted ({type(exc).__name__}). Retrying.'
                    LOG.warning('Upstream interrupted: %s',type(exc).__name__)
                    attempts += 1
                    if attempts >= 3:
                        self.play_url = None
                        attempts = 0
                if not self.stopping:
                    self.reconnects += 1
                    await asyncio.sleep(2)
        await asyncio.gather(*(pc.close() for pc,_ in list(self.peers.values())),return_exceptions=True)
    def _youtube(self):
        """Read HLS ahead, pace original packets, and decode once for analytics."""
        from .youtube_source import resolve
        from .preview_stream import SourceClock, PreviewStream
        generation=self.source_generation
        now=time.monotonic()
        if (self.source_type != 'direct' and self._resolved_source
                and self._resolved_source_id == self.stream_id
                and now - self._resolved_source_time < 7200):
            source = self._resolved_source
        else:
            source = resolve(self.stream_id) if self.source_type != 'direct' else {'url': self.direct_url, 'fps': None}
            if self.source_type != 'direct':
                self._resolved_source = source
                self._resolved_source_id = self.stream_id
                self._resolved_source_time = now
        if self.stopping or generation!=self.source_generation:return
        is_live = bool(source.get('is_live', True))
        if source.get('title') and self.stream_id != 'W0hB9CXqPAc':
            self.source_title = source['title']
        if not is_live and source.get('duration'):
            self.playback_duration = float(source['duration'])
        self.source_quality={key:source.get(key) for key in ('width','height','fps','vcodec','tbr','format_id')}
        epoch=uuid.uuid4().hex[:12]
        clock=SourceClock()
        resume_pts = 0.0
        if not is_live and self.playback_position > 0:
            resume_pts = max(0.0, self.playback_position - 0.5)
        preview=None
        if source.get('preview'):
            preview=PreviewStream(source['preview'],self.native_video,clock,epoch,
                                  lambda:self.stopping or generation!=self.source_generation,
                                  resume_position=resume_pts if not is_live else 0.0,
                                  is_paused_fn=lambda:self.is_paused).start()
        self.preview=preview
        preview_start=time.monotonic()
        self.times.clear()
        self.state,self.message='connecting','Opening video source'
        pending=queue.Queue(maxsize=180)  # At most six seconds at the current 30 fps.
        reader_stop=threading.Event()
        primed=threading.Event()
        def put(value):
            while not reader_stop.is_set() and not self.stopping and generation==self.source_generation:
                try:pending.put(value,timeout=.1);return True
                except queue.Full:pass
            return False
        def read():
            try:
                # Stay behind the HLS publication edge, then fetch ahead while the
                # presentation worker plays. Network fetches cannot pause each GOP.
                av_options = {'rw_timeout': '12000000'}
                if is_live:
                    av_options['live_start_index'] = '-3'
                if source['url'].startswith(('rtsp:', 'rtsps:')):
                    av_options['rtsp_transport'] = 'tcp'
                hw = None
                if os.name == 'nt':
                    try:
                        from av.codec.hwaccel import HWAccel, hwdevices_available
                        if 'd3d11va' in hwdevices_available():
                            # Keep decoded surfaces on the GPU until a sampled
                            # frame is actually needed by analytics.
                            hw = HWAccel('d3d11va', allow_software_fallback=True, is_hw_owned=True)
                    except (ImportError, ValueError, TypeError):
                        pass
                try:
                    opened = av.open(source['url'], options=av_options, timeout=(15, 12), **({'hwaccel': hw} if hw else {}))
                except Exception:
                    if self.source_type != 'direct':
                        self._resolved_source = None
                    if hw is None: raise
                    LOG.info('Hardware video device unavailable; using software decoder')
                    opened = av.open(source['url'], options=av_options, timeout=(15, 12))
                with opened as container:
                    stream=container.streams.video[0]
                    stream.thread_type='AUTO';stream.thread_count=2
                    self.source_quality['decode_acceleration'] = 'd3d11va' if getattr(stream.codec_context,'is_hwaccel',False) else 'software'
                    # Keep the legacy flag name for API compatibility. It now
                    # means packet-copy transport, including UHD VP9 and AV1.
                    self.native_h264=stream.codec_context.name in ('h264','vp9','av1','libdav1d')
                    self.source_quality.update(codec=stream.codec_context.name,
                        width=getattr(stream.codec_context,'width',source.get('width')),height=getattr(stream.codec_context,'height',source.get('height')))
                    self.source_fps=float(stream.average_rate) if stream.average_rate else source.get('fps')
                    if not is_live and resume_pts > 0:
                        try:
                            target_pts = int(resume_pts / stream.time_base)
                            container.seek(target_pts, backward=True, stream=stream)
                            LOG.info('Resumed master container at %.2fs', resume_pts)
                        except Exception as s_err:
                            LOG.warning('Master container seek to %.2fs failed: %s', resume_pts, s_err)
                    for packet in container.demux(stream):
                        if not packet.size or packet.pts is None:continue
                        if not put(packet):return
                        if pending.qsize()>=45:primed.set()
            except Exception as exc:
                put(exc)
            finally:
                primed.set();put(None)
        reader=threading.Thread(target=read,name='youtube-prefetch',daemon=True)
        reader.start()
        origin_pts=origin_dts=origin_mono=origin_wall=None
        muxer=None
        decoder=None
        def publish_frame(frame, wall_offset):
            media=float(frame.pts*frame.time_base)
            self.playback_position=media
            image=frame.to_ndarray(format='bgr24',threads=4)
            self.sequence+=1
            self.times.append(time.monotonic())
            self.frames.publish(FramePacket(self.sequence,epoch,wall_offset+media,time.monotonic(),image,media))
        try:
            while not primed.wait(.1):
                if self.stopping or generation!=self.source_generation:return
            while not self.stopping and generation==self.source_generation:
                try:packet=pending.get(timeout=.5)
                except queue.Empty:continue
                if packet is None:break
                if isinstance(packet,Exception):raise packet
                if self.stopping or generation!=self.source_generation:return
                pts=float(packet.pts*packet.time_base)
                dts=float((packet.dts if packet.dts is not None else packet.pts)*packet.time_base)
                if origin_pts is None:
                    origin_pts,origin_dts,origin_mono,origin_wall=pts,dts,time.monotonic(),time.time()
                    clock.start(origin_pts,origin_dts,origin_mono,origin_wall)
                elapsed=dts-origin_dts
                if elapsed<0:raise RuntimeError('Source video clock reset')
                if self.is_paused:
                    pause_mono = time.monotonic()
                    with self._pause_cond:
                        while self.is_paused and not self.stopping and generation == self.source_generation:
                            self._pause_cond.wait(0.05)
                    pause_delay = time.monotonic() - pause_mono
                    origin_mono += pause_delay
                    origin_wall += pause_delay
                    clock.shift(pause_delay)
                    target = origin_mono + elapsed
                target=origin_mono+elapsed
                while not self.stopping and generation==self.source_generation and target>time.monotonic():
                    if self.is_paused:
                        break
                    time.sleep(min(.025,target-time.monotonic()))
                if self.stopping or generation!=self.source_generation:return
                if is_live:
                    if time.monotonic()-target>1.5:
                        adjustment=time.monotonic()-target
                        origin_mono+=adjustment;origin_wall+=adjustment
                        clock.shift(adjustment)
                        if decoder:decoder.close();decoder=None
                        epoch=uuid.uuid4().hex[:12]
                        if muxer:
                            muxer.close();muxer=None
                else:
                    if time.monotonic()-target>1.0:
                        drift=time.monotonic()-target
                        origin_mono+=drift;origin_wall+=drift
                        clock.shift(drift)
                stamp=origin_wall+pts-origin_pts
                if preview is not None:
                    if preview.state == 'failed' or (preview.packets == 0 and preview.state != 'ended' and time.monotonic() - preview_start > 4.0):
                        self.playback_warning='1080p rendition unavailable; using the original source'
                        preview.close();preview=None;self.preview=None
                if self.native_h264 and preview is None:
                    try:
                        if muxer is None and packet.is_keyframe:
                            muxer=NativeMuxer(self.native_video,packet.stream,packet,epoch,stamp,source.get('vcodec'))
                        if muxer:muxer.mux(packet)
                    except Exception as exc:
                        # Presentation failure must not interrupt acquisition/AI.
                        LOG.warning('Native browser transport disabled: %s',type(exc).__name__)
                        self.native_h264=False
                        if muxer:
                            try:muxer.close()
                            except Exception:pass
                            muxer=None
                if self.native_h264 or decoder is not None:
                    if decoder is None:
                        decoder=AnalyticsDecoder(publish_frame,max_fps=24)
                        self.analytics_decoder=decoder
                    decoder.submit(packet,origin_wall-origin_pts,target)
                else:
                    for frame in packet.decode():
                        publish_frame(frame,origin_wall-origin_pts)
                self.state,self.message='live',('Live video' if is_live else 'Video playback')+(
                    ' · 1080p preview / native-resolution AI captures' if preview else ' · native source quality')
        finally:
            reader_stop.set()
            reader.join(timeout=15)
            if preview:preview.close()
            if decoder:decoder.close()
            if muxer:muxer.close()
            if not is_live and self.playback_duration and self.playback_position >= (self.playback_duration - 2.0):
                self.playback_position = 0.0
    async def _connect(self, session, url):
        pc = self.pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        finished = asyncio.Event()
        readers = []
        epoch = uuid.uuid4().hex[:12]
        self.times.clear()
        self.state, self.message = 'connecting', 'Negotiating the existing Pattaya WebRTC protocol'
        pc.addTransceiver('video',direction='recvonly')
        pc.addTransceiver('audio',direction='recvonly')
        @pc.on('connectionstatechange')
        async def state_changed():
            if pc.connectionState in ('failed','closed'):
                finished.set()
        @pc.on('track')
        def received_track(track):
            if track.kind == 'video':
                async def read():
                    try:
                        while not self.stopping and not finished.is_set():
                            frame = await asyncio.wait_for(track.recv(),timeout=12)
                            image = frame.to_ndarray(format='bgr24')
                            self.sequence += 1
                            stamp, mono = time.time(), time.monotonic()
                            self.times.append(mono)
                            self.frames.publish(FramePacket(self.sequence,epoch,stamp,mono,image))
                            self.state, self.message = 'live', 'Receiving decoded frames'
                    except Exception:
                        finished.set()
                readers.append(asyncio.create_task(read()))
        try:
            await negotiate(pc,session,url)
            # Also detect a successful SDP exchange that never delivers frames.
            started = time.monotonic()
            while not finished.is_set() and not self.stopping:
                await asyncio.sleep(.25)
                latest = self.frames.latest
                if time.monotonic()-started>18 and (not latest or latest.epoch!=epoch):
                    raise TimeoutError('No first camera frame')
        finally:
            for task in readers: task.cancel()
            await asyncio.gather(*readers,return_exceptions=True)
            await pc.close()
            self.pc = None
            self.state = 'reconnecting' if not self.stopping else 'stopped'
    async def relay_offer(self, sdp, kind):
        if len(self.peers)>=4:
            raise ValueError('At most four local relay clients')
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        ident = uuid.uuid4().hex
        track = RelayTrack(self)
        self.peers[ident] = (pc, track)
        @pc.on('connectionstatechange')
        async def changed():
            if pc.connectionState in ('failed','closed'):
                self.peers.pop(ident,None)
                await pc.close()
            elif pc.connectionState=='disconnected':
                async def expire_disconnected():
                    await asyncio.sleep(10)
                    if pc.connectionState=='disconnected':
                        self.peers.pop(ident,None)
                        await pc.close()
                asyncio.create_task(expire_disconnected())
        try:
            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp,type=kind))
            pc.addTrack(track)
            # H.264 is efficient on the existing camera's native 768×432 picture.
            for transceiver in pc.getTransceivers():
                if transceiver.kind=='video':
                    codecs=[c for c in RTCRtpSender.getCapabilities('video').codecs if c.mimeType=='video/H264']
                    transceiver.setCodecPreferences(codecs)
            answer=await pc.createAnswer()
            await pc.setLocalDescription(answer)
            return {'sdp':pc.localDescription.sdp,'type':pc.localDescription.type,'peer_id':ident}
        except Exception:
            self.peers.pop(ident,None)
            await pc.close()
            raise
    def clock(self, ident):
        peer = self.peers.get(ident)
        return peer[1].origin if peer else None
    def close(self):
        self.stopping = True
        self.restart()
        if self.thread.is_alive(): self.thread.join(20)
