"""Copy source video into browser-native MP4 fragments without re-encoding."""
from collections import deque
import json
import struct
import threading
import av
from aiortc.codecs.h264 import H264Encoder


def framed(data):
    return struct.pack('!I', len(data))+data


def starts_with_keyframe(data):
    """Read ISO BMFF sample flags, not a guessed byte pattern inside video data."""
    default=first=None
    def boxes(blob):
        nonlocal default,first
        offset=0
        while offset+8<=len(blob):
            size,kind=struct.unpack_from('!I4s',blob,offset)
            if size<8 or offset+size>len(blob):break
            body=blob[offset+8:offset+size]
            if kind in (b'moof',b'traf'):boxes(body)
            elif kind==b'tfhd':
                flags=int.from_bytes(body[1:4],'big');pos=8
                for bit,length in ((1,8),(2,4),(8,4),(16,4)):
                    if flags&bit:pos+=length
                if flags&32:default=struct.unpack_from('!I',body,pos)[0]
            elif kind==b'trun':
                flags=int.from_bytes(body[1:4],'big');pos=8+(4 if flags&1 else 0)
                if flags&4:first=struct.unpack_from('!I',body,pos)[0]
                elif flags&1024:
                    pos+=(4 if flags&256 else 0)+(4 if flags&512 else 0)
                    first=struct.unpack_from('!I',body,pos)[0]
            offset+=size
    boxes(data)
    value=first if first is not None else default
    return value is not None and not value&0x10000


class FragmentRelay:
    def __init__(self):
        self.condition=threading.Condition()
        self.epoch=None;self.init=None;self.info=None
        self.chunks=deque();self.sequence=0;self.bytes=0

    def reset(self,epoch,origin,mime,**metadata):
        with self.condition:
            self.epoch=epoch;self.init=None
            self.info=dict(epoch=epoch,origin=origin,mime=mime,**metadata)
            self.chunks.clear();self.bytes=0
            self.condition.notify_all()

    def publish(self,data):
        if not data:return
        with self.condition:
            if self.init is None:self.init=data
            else:
                self.sequence+=1;self.chunks.append((self.sequence,data,starts_with_keyframe(data)));self.bytes+=len(data)
                while len(self.chunks)>40 or self.bytes>24*1024*1024:
                    self.bytes-=len(self.chunks.popleft()[1])
            self.condition.notify_all()

    def stream(self,stopping):
        with self.condition:
            self.condition.wait_for(lambda:self.init is not None or stopping(),20)
            if self.init is None:return
            epoch=self.epoch;info=self.info.copy();init=self.init
            key=next((p for p in reversed(self.chunks) if p[2]),None)
            after=key[0]-1 if key else self.sequence
        yield framed(json.dumps(info).encode())
        yield framed(init)
        while not stopping():
            with self.condition:
                self.condition.wait_for(lambda:self.epoch!=epoch or self.sequence>after or stopping(),1)
                if self.epoch!=epoch or stopping():return
                if self.chunks and after<self.chunks[0][0]-1:return
                chunks=[p for p in self.chunks if p[0]>after]
            for ident,data,keyframe in chunks:
                after=ident
                yield framed(data)


class FragmentSink:
    def __init__(self):self.data=bytearray();self.position=0
    def write(self,data):self.data.extend(data);self.position+=len(data);return len(data)
    def tell(self):return self.position
    def writable(self):return True
    def drain(self):
        data=bytes(self.data);self.data.clear();return data


def source_mime(stream, codec_hint=None):
    """RFC 6381 codec identifiers for the source, including UHD VP9/AV1."""
    context = stream.codec_context
    if context.name == 'h264':
        extra = bytes(context.extradata or b'')
        sps = next((n for n in H264Encoder._split_bitstream(extra) if n and n[0]&31==7), None)
        profile = extra[1:4].hex() if extra[:1] == b'\x01' else sps[1:4].hex() if sps else '640028'
        codec = f'avc1.{profile}'
    elif context.name == 'vp9':
        # Profile 0, 8-bit is the normal YouTube SDR stream. Read profile/depth
        # for 10-bit streams rather than advertising every VP9 source as 8-bit.
        depth = 10 if '10' in str(context.pix_fmt) else 12 if '12' in str(context.pix_fmt) else 8
        profile = 2 if depth > 8 else 0
        codec = f'vp09.{profile:02d}.51.{depth:02d}'
    elif context.name in ('av1', 'libdav1d'):
        codec = codec_hint if str(codec_hint).startswith('av01.') else 'av01.0.13M.08'
    else:
        raise ValueError(f'No native MP4 transport for {context.name}')
    return f'video/mp4; codecs="{codec}"'


class NativeMuxer:
    def __init__(self,relay,stream,packet,epoch,wall,codec_hint=None):
        self.relay=relay;self.sink=FragmentSink()
        self.base=packet.dts if packet.dts is not None else packet.pts
        self.time_base=packet.time_base
        origin=wall-float((packet.pts-self.base)*packet.time_base)
        context=stream.codec_context
        rate=getattr(stream,'average_rate',None)
        relay.reset(epoch,origin,source_mime(stream,codec_hint),
                    width=getattr(context,'width',None),height=getattr(context,'height',None),
                    fps=float(rate) if rate else None)
        self.output=av.open(self.sink,'w',format='mp4',options={
            'movflags':'empty_moov+default_base_moof+frag_keyframe','frag_duration':'500000','flush_packets':'1'})
        self.stream=self.output.add_stream_from_template(stream)

    def mux(self,source):
        packet=av.Packet(bytes(source));packet.time_base=self.time_base
        packet.pts=source.pts-self.base
        packet.dts=(source.dts if source.dts is not None else source.pts)-self.base
        packet.duration=source.duration;packet.is_keyframe=source.is_keyframe
        packet.stream=self.stream
        self.output.mux(packet)
        self.relay.publish(self.sink.drain())

    def close(self):
        self.output.close()
        self.relay.publish(self.sink.drain())
