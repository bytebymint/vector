"""Exercise worker scheduling separately from wall/performance clock origins."""
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from ultralytics.engine.results import Boxes
import services.traffic_service as ts
from services.storage import Storage
from services.stream_manager import FrameMailbox,FramePacket


def test_latest_frame_worker_is_independent_of_performance_clock_epoch(tmp_path,monkeypatch):
    class Detector:
        def __init__(self,*args):self.key=None;self.model=True;self.subtype=None;self.warning=None;self.resolution=640;self.device='cpu';self.backend='pytorch'
        def load(self,s):self.key=(s['model'],s['backend'])
        def predict(self,image,*args):return Boxes(np.array([[10,10,50,50,.9,2]],np.float32),image.shape[:2])
        def adapt(self,*args):pass
    monkeypatch.setattr(ts,'Detector',Detector)
    real_perf=time.perf_counter
    monkeypatch.setattr(time,'perf_counter',lambda:real_perf()+1000) # Windows clocks may have distinct origins.
    stream=SimpleNamespace(frames=FrameMailbox(),status=lambda:{'fps':25})
    storage=Storage(tmp_path/'worker.db')
    service=ts.TrafficService(stream,storage,tmp_path)
    service.settings['max_fps']=12
    service.start()
    started=time.monotonic()
    try:
        for seq in range(1,26):
            stream.frames.publish(FramePacket(seq,'fixture',time.time(),time.monotonic(),np.zeros((100,100,3),np.uint8)))
            time.sleep(.04)
        assert len(service.samples)>=7
        assert service.skipped>0
        assert service.snapshot['sequence']>=18
        assert service.error is None
    finally:
        service.close();storage.close()


def test_model_warmup_discards_the_frame_acquired_before_loading(tmp_path,monkeypatch):
    mailbox=FrameMailbox()
    processed=[]
    service=None

    class Detector:
        def __init__(self,*args):
            self.key=None;self.model=True;self.subtype=None;self.warning=None
            self.resolution=640;self.device='cpu';self.backend='pytorch'
        def load(self,s):
            self.key=(s['model'],s['backend'])
            # Acquisition continues while model initialization is busy.
            mailbox.publish(FramePacket(2,'fixture',time.time()-1000,time.monotonic(),np.full((100,100,3),2,np.uint8),12.0))
        def predict(self,image,*args):
            processed.append(int(image[0,0,0]))
            service.stopping.set()
            return Boxes(np.array([[10,10,50,50,.9,2]],np.float32),image.shape[:2])
        def adapt(self,*args):pass

    monkeypatch.setattr(ts,'Detector',Detector)
    stream=SimpleNamespace(frames=mailbox,status=lambda:{'fps':25})
    storage=Storage(tmp_path/'worker.db')
    service=ts.TrafficService(stream,storage,tmp_path)
    mailbox.publish(FramePacket(1,'fixture',time.time(),time.monotonic(),np.ones((100,100,3),np.uint8)))
    try:
        service.start();service.thread.join(3)
        assert not service.thread.is_alive()
        assert processed==[2]
        assert service.snapshot['sequence']==2
        assert service.snapshot['media_time']==12.0
        assert service.metrics['analytics_latency_ms']<500
        assert service.error is None
    finally:
        service.close();storage.close()
