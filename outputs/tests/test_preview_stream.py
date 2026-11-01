from fractions import Fraction
import io
import time
from types import SimpleNamespace

import av
import numpy as np
import pytest

from services.native_video import FragmentRelay
from services.preview_stream import PreviewStream, SourceClock


def test_renditions_share_pts_clock_with_different_decode_origins():
    clock=SourceClock()
    assert clock.snapshot() is None
    clock.start(5.,4.966666,20.,100.)
    original=clock.snapshot()
    assert original.timestamp(6.)==101.
    # B-frame decoding origins may differ from presentation timestamps.
    assert original.due(5.966666)==pytest.approx(21.)
    clock.shift(2.)
    updated=clock.snapshot()
    assert updated.timestamp(6.)==103.
    assert updated.due(5.966666)==pytest.approx(23.)
    assert updated.revision==1 and original.revision==0


def make_video(path,width=96,height=54):
    with av.open(str(path),'w') as output:
        stream=output.add_stream('libx264',rate=60)
        stream.width=width;stream.height=height;stream.pix_fmt='yuv420p'
        stream.options={'preset':'ultrafast','tune':'zerolatency','g':'12','bf':'0'}
        for n in range(36):
            image=np.full((height,width,3),50+n,np.uint8)
            frame=av.VideoFrame.from_ndarray(image,format='bgr24')
            frame.pts=n;frame.time_base=Fraction(1,60)
            for packet in stream.encode(frame):output.mux(packet)
        for packet in stream.encode():output.mux(packet)


def test_native_preview_preserves_60fps_without_decoding_and_has_exact_clock(tmp_path):
    path=tmp_path/'preview.mp4'
    make_video(path)
    relay=FragmentRelay()
    clock=SourceClock()
    clock.start(0.,0.,time.monotonic(),100.)
    preview=PreviewStream({'url':str(path),'fps':60},relay,clock,'test',lambda:False,max_lateness=10.)
    preview.start();preview.thread.join(timeout=5)
    try:
        assert not preview.thread.is_alive()
        assert preview.state=='ended' and preview.error is None
        assert preview.packets==36
        assert relay.info['origin']==pytest.approx(100.)
        assert relay.info['width']==96 and relay.info['height']==54
        assert relay.info['fps']==60
        encoded=relay.init+b''.join(row[1] for row in relay.chunks)
        with av.open(io.BytesIO(encoded)) as container:
            frames=list(container.decode(video=0))
        assert len(frames)==36
        assert float(frames[-1].pts*frames[-1].time_base)==pytest.approx(35/60,abs=.0001)
    finally:preview.close()


def test_missing_preview_is_reported_and_does_not_touch_ai_mailbox(tmp_path):
    relay=FragmentRelay();clock=SourceClock()
    preview=PreviewStream({'url':str(tmp_path/'missing.mp4')},relay,clock,'test',lambda:False)
    preview.start();preview.thread.join(timeout=3)
    assert preview.state=='failed' and preview.error
    assert relay.init is None


def test_preview_shutdown_before_master_clock_is_ready(tmp_path):
    path=tmp_path/'preview.mp4';make_video(path)
    preview=PreviewStream({'url':str(path)},FragmentRelay(),SourceClock(),'test',lambda:False)
    preview.start();preview.close()
    assert not preview.thread.is_alive()


def test_master_frames_stay_at_native_size_with_separate_preview(tmp_path,monkeypatch):
    from services import youtube_source
    from services.stream_manager import StreamManager
    from av.codec import hwaccel
    # Offline fixtures must not initialize a machine-specific Windows GPU driver.
    monkeypatch.setattr(hwaccel,'hwdevices_available',lambda:[])
    master=tmp_path/'master.mp4';preview_file=tmp_path/'preview.mp4'
    make_video(master,192,108);make_video(preview_file,96,54)
    monkeypatch.setattr(youtube_source,'resolve',lambda _:dict(url=str(master),is_live=False,
        preview=dict(url=str(preview_file),fps=60,width=96,height=54)))
    manager=StreamManager();manager.source_type='youtube'
    packets=[];manager.frames.publish=packets.append
    try:
        manager._youtube()
        assert packets and all(p.image.shape[:2]==(108,192) for p in packets)
        assert manager.native_video.info['width']==96
        assert manager.native_video.info['height']==54
        assert manager.preview.status()['separate_rendition']
        assert manager.native_video.info['origin']==pytest.approx(packets[0].timestamp-packets[0].media_time,abs=.01)
    finally:manager.loop.close()


def test_equivalent_rendition_is_retried_after_rejected_preview(tmp_path):
    path=tmp_path/'equivalent.mp4';make_video(path)
    clock=SourceClock();clock.start(0,0,time.monotonic(),100.)
    relay=FragmentRelay()
    source=dict(url=str(tmp_path/'rejected.mp4'),format_id='first',
                alternatives=[dict(url=str(path),format_id='equivalent',fps=60)])
    preview=PreviewStream(source,relay,clock,'test',lambda:False,max_lateness=10.).start()
    preview.thread.join(timeout=5)
    try:
        assert preview.state=='ended' and preview.error is None
        assert preview.status()['format_id']=='equivalent'
        assert preview.packets==36 and relay.info['fps']==60
    finally:preview.close()
