import time
import asyncio
import numpy as np
import pytest
from ultralytics.engine.results import Boxes
from services.detector import DEFAULTS,validate_settings
from services.tracker import VehicleTracker
from services.storage import Storage
from services.stream_manager import FrameMailbox,FramePacket,validate_play_url,StreamManager

@pytest.mark.parametrize('kind',['bytetrack','botsort'])
def test_actual_tracker_brief_occlusion_recovery(kind,tmp_path):
    tracker=VehicleTracker(validate_settings({**DEFAULTS,'tracker':kind}),tmp_path)
    image=np.zeros((432,768,3),np.uint8)
    def boxes(x=None):return Boxes(np.empty((0,6),np.float32) if x is None else np.array([[x,100,x+70,160,.9,3]],np.float32),image.shape[:2])
    ids=[]
    for seq,x in [(1,100),(2,101),(3,102),(4,None),(5,None),(6,105),(7,106)]:
        rows=tracker.update(boxes(x),image,seq)
        if len(rows):ids.append(int(rows[0,4]))
    assert len(ids)>=4 and len(set(ids))==1

def test_skipped_frames_preserve_tentative_confirmation(tmp_path):
    tracker=VehicleTracker(validate_settings(DEFAULTS),tmp_path)
    image=np.zeros((432,768,3),np.uint8)
    ids=[]
    for seq in [1,5,9,13,17]:
        boxes=Boxes(np.array([[100+seq*.1,100,170+seq*.1,160,.9,2]],np.float32),image.shape[:2])
        rows=tracker.update(boxes,image,seq)
        if len(rows):ids.append(int(rows[0,4]))
    assert len(ids)>=3 and len(set(ids))==1

def test_sqlite_idempotency_and_reopen(tmp_path):
    path=tmp_path/'traffic.db';db=Storage(path)
    event=dict(key='same-crossing',timestamp=time.time(),type='crossing',uid='a',direction='negative_to_positive',confirmed=True,**{'class':'car'})
    db.enqueue('event',event);db.enqueue('event',event);db.save('geometry',{'items':[]});db.flush();db.close()
    db=Storage(path)
    assert len(db.events())==1 and db.load('geometry')=={'items':[]}
    assert 'same-crossing' in db.export_csv()
    db.close()

def test_mailbox_disconnection_recovery_and_boundedness():
    mailbox=FrameMailbox();image=np.zeros((2,2,3),np.uint8)
    mailbox.publish(FramePacket(1,'first',1,1,image))
    assert mailbox.wait(0,.01).sequence==1
    assert mailbox.wait(1,.01) is None  # disconnected, consumer times out rather than freezing
    for seq in range(2,100):mailbox.publish(FramePacket(seq,'recovered',seq,seq,image))
    packet=mailbox.wait(1,.01)
    assert packet.sequence==99 and packet.epoch=='recovered'

def test_rejects_untrusted_play_url():
    for url in ['https://example.com/media/index/api/webrtc?stream=a&token=x&type=play','file:///etc/passwd','https://livestream.pattaya.go.th/media/index/api/webrtc?stream=a&type=play']:
        with pytest.raises(ValueError):validate_play_url(url)
    assert validate_play_url('/media/index/api/webrtc?stream=a&token=test&type=play').startswith('https://livestream.pattaya.go.th/')

def test_authorization_challenge_waits_instead_of_hammering():
    class Reply:
        status=403
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
    class Session:
        def get(self,*args):return Reply()
    manager=StreamManager()
    assert asyncio.run(manager._resolve(Session())) is None
    assert manager.state=='authorization_required'
    manager.loop.close()

def test_bytetrack_arguments(tmp_path):
    tracker = VehicleTracker(validate_settings(DEFAULTS), tmp_path)
    assert tracker.tracker.args.track_high_thresh == 0.20
    assert tracker.tracker.args.track_low_thresh == 0.06
    assert tracker.tracker.args.new_track_thresh == 0.15
    assert tracker.tracker.args.match_thresh == 0.75

def test_dynamic_stream_authorization():
    manager = StreamManager('4f2e6981-eefd-4eb0-ae5c-ccdd5cb3344a')
    # Public stream API URL
    manager.authorize('https://livestream.pattaya.go.th/api/public-stream/c8977fdd-4adb-41af-a499-501216606bd7?reportedIp=83.110.153.183')
    assert manager.stream_id == 'c8977fdd-4adb-41af-a499-501216606bd7'
    assert 'RC-264' in manager.status()['camera']
    # Direct UUID
    manager.authorize('4f2e6981-eefd-4eb0-ae5c-ccdd5cb3344a')
    assert manager.stream_id == '4f2e6981-eefd-4eb0-ae5c-ccdd5cb3344a'
    assert 'RC-293' in manager.status()['camera']
    # PlayUrl
    manager.authorize('https://livestream.pattaya.go.th/media/index/api/webrtc?type=play&stream=c8977fdd-4adb-41af-a499-501216606bd7&token=tok')
    assert manager.stream_id == 'c8977fdd-4adb-41af-a499-501216606bd7'
    assert manager.play_url is not None
    manager.loop.close()
