import asyncio
from fractions import Fraction
import json
from types import SimpleNamespace
from pathlib import Path
import av
import cv2
import numpy as np
import pytest
from services.stream_manager import StreamManager
from services.youtube_source import youtube_id
from services.road_geometry import calibration_matrix,ground_speed


def test_public_watch_urls_only():
    assert youtube_id('https://www.youtube.com/watch?v=W0hB9CXqPAc')=='W0hB9CXqPAc'
    assert youtube_id('https://youtu.be/W0hB9CXqPAc')=='W0hB9CXqPAc'
    assert youtube_id('https://youtube.com.evil.test/watch?v=W0hB9CXqPAc') is None
    with pytest.raises(ValueError):youtube_id('https://youtube.com/watch?v=bad')


def test_camera_switch_and_failed_request_preserves_source():
    manager=StreamManager()
    try:
        manager.authorize('https://www.youtube.com/watch?v=W0hB9CXqPAc')
        assert manager.source_type=='youtube'
        with pytest.raises(ValueError):manager.authorize('file:///unsupported-video')
        assert manager.source_type=='youtube'
        manager.authorize('4f2e6981-eefd-4eb0-ae5c-ccdd5cb3344a')
        assert manager.source_type=='pattaya'
    finally:manager.loop.close()


def test_bursty_decode_preserves_source_timestamps(monkeypatch):
    import services.youtube_source as source
    import services.stream_manager as sm
    monkeypatch.setattr(source,'resolve',lambda ident:{'url':'fixture'})
    class Container:
        streams=SimpleNamespace(video=[SimpleNamespace(codec_context=SimpleNamespace(name='fixture'),average_rate=30)])
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def demux(self,stream):
            for i in range(5):
                frame=av.VideoFrame.from_ndarray(np.zeros((24,32,3),np.uint8),format='bgr24')
                frame.pts=90000+i*3000;frame.time_base=Fraction(1,90000)
                yield SimpleNamespace(pts=frame.pts,dts=frame.pts,time_base=frame.time_base,size=1,decode=lambda frame=frame:[frame])
    monkeypatch.setattr(sm.av,'open',lambda *args,**kwargs:Container())
    manager=StreamManager();packets=[]
    manager.frames.publish=lambda packet:packets.append(packet)
    try:
        manager._youtube()
        assert len(packets)==5
        assert all(abs(packets[i+1].media_time-packets[i].media_time-1/30)<1e-5 for i in range(4))
        assert all(packets[i+1].timestamp>=packets[i].timestamp for i in range(4))
    finally:manager.loop.close()


def test_map_plane_physical_speed_and_occlusion():
    geometry=json.loads((Path(__file__).parents[1]/'southampton-geometry.json').read_text(encoding='utf-8'))
    matrix=calibration_matrix(geometry['calibration'])
    inverse=np.linalg.inv(matrix)
    # Known synthetic ground trajectory at 10 m/s, projected through the real calibration.
    world=np.float32([[[3+i*.8,1] for i in range(10)]])
    image=cv2.perspectiveTransform(world,inverse)[0]
    history=[(100+i*.08,*p) for i,p in enumerate(image)]
    assert abs(ground_speed(history,matrix)*3.6-36)<.01
    assert ground_speed(history[:3],matrix) is None
    gapped=[history[0],history[1],history[-2],history[-1]]
    assert ground_speed(gapped,matrix) is None
