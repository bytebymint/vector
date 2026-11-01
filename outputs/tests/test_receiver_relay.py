"""Offline protocol/transport verification. These fixtures are never dashboard input."""
import asyncio
import time
from fractions import Fraction
import av
import numpy as np
from aiortc import RTCPeerConnection,RTCConfiguration,RTCSessionDescription
import services.stream_manager as sm


def test_existing_negotiation_and_disconnect_recovery(monkeypatch):
    async def scenario():
        manager=sm.StreamManager();real_loop=manager.loop;manager.loop=asyncio.get_running_loop()
        calls=[]
        class Track:
            kind='video'
            async def recv(self):
                await asyncio.sleep(.01)
                return av.VideoFrame.from_ndarray(np.zeros((32,48,3),np.uint8),format='bgr24')
        class Peer:
            connectionState='connected'
            localDescription=type('SDP',(),{'sdp':'fixture-offer'})()
            def __init__(self,*args,**kwargs):self.callbacks={};self.transceivers=[]
            def addTransceiver(self,kind,direction):self.transceivers.append((kind,direction));calls.append((kind,direction))
            def on(self,name):
                def register(fn):self.callbacks[name]=fn;return fn
                return register
            async def createOffer(self):return self.localDescription
            async def setLocalDescription(self,offer):pass
            async def setRemoteDescription(self,answer):self.callbacks['track'](Track())
            async def close(self):self.connectionState='closed';await self.callbacks['connectionstatechange']()
        class Reply:
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            def raise_for_status(self):pass
            async def json(self,**kwargs):return {'sdp':'fixture-answer'}
        class Session:
            def post(self,url,**kwargs):calls.append(kwargs);return Reply()
        monkeypatch.setattr(sm,'RTCPeerConnection',Peer)
        for _ in range(2):
            previous=manager.sequence
            task=asyncio.create_task(manager._connect(Session(),'https://livestream.pattaya.go.th/media/index/api/webrtc'))
            for _ in range(40):
                await asyncio.sleep(.01)
                if manager.sequence>previous:break
            assert manager.sequence>previous
            epoch=manager.frames.latest.epoch
            await manager.pc.close();await asyncio.wait_for(task,2)
            assert manager.state=='reconnecting'
            if previous:assert epoch!=first_epoch
            first_epoch=epoch
        assert ('audio','recvonly') in calls and ('video','recvonly') in calls
        post=next(c for c in calls if isinstance(c,dict))
        assert post['headers']['Origin']=='https://livestream.pattaya.go.th'
        assert post['headers']['Content-Type']=='text/plain;charset=utf-8'
        assert post['data']=='fixture-offer'
        real_loop.close()
    asyncio.run(scenario())


def test_actual_local_webrtc_relay_and_timestamp_clock():
    async def scenario():
        manager=sm.StreamManager();manager.stopping=False
        source_loop=manager.loop
        sm.enable_rsa_dtls_ciphers()
        client=RTCPeerConnection(RTCConfiguration(iceServers=[]))
        client.addTransceiver('video',direction='recvonly')
        received=asyncio.Event();answer=None
        @client.on('track')
        def ontrack(track):
            async def read():
                frame=await asyncio.wait_for(track.recv(),8)
                assert frame.width==96 and frame.height==64
                received.set()
            asyncio.create_task(read())
        async def source():
            for i in range(100):
                manager.frames.publish(sm.FramePacket(i+1,'fixture',time.time(),time.monotonic(),np.zeros((64,96,3),np.uint8)))
                await asyncio.sleep(.04)
        publisher=asyncio.create_task(source())
        try:
            await client.setLocalDescription(await client.createOffer())
            answer=await manager.relay_offer(client.localDescription.sdp,'offer')
            await client.setRemoteDescription(RTCSessionDescription(sdp=answer['sdp'],type='answer'))
            await asyncio.wait_for(received.wait(),8)
            assert manager.clock(answer['peer_id']) is not None
        finally:
            publisher.cancel();await asyncio.gather(publisher,return_exceptions=True)
            await client.close()
            await asyncio.gather(*(pc.close() for pc,_ in list(manager.peers.values())),return_exceptions=True)
            source_loop.close()
    asyncio.run(scenario())
