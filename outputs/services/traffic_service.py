"""One inference worker, separate from acquisition, HTTP and event persistence."""
from __future__ import annotations
from collections import Counter, deque
import copy
import logging
import threading
import time
import uuid
import cv2
import numpy as np
from .detector import Detector, DEFAULTS, validate_settings, SubtypeClassifier
from .tracker import VehicleTracker
from .traffic_engine import TrafficEngine
from .performance import Performance
from .presentation import compact_frame
from .lane_learning import LaneLearner
from .people_engine import PeopleCounter
from ultralytics.engine.results import Boxes

LOG=logging.getLogger(__name__)


class TrafficService:
    def __init__(self,stream,storage,models_dir):
        self.stream,self.storage=stream,storage
        self.models_dir=models_dir
        self.lock=threading.RLock()
        self.settings=validate_settings(storage.load('settings',DEFAULTS))
        self.geometry=storage.load('geometry',{'items':[],'calibration':None})
        self.engine=TrafficEngine(storage)
        self.people=PeopleCounter()
        self.lane_learner=LaneLearner()
        self.engine.set_geometry(self.geometry)
        self.performance=Performance()
        self.stopping=threading.Event()
        self.thread=threading.Thread(target=self._run,name='vehicle-inference',daemon=True)
        self.status='starting';self.error=None;self.warning=None;self.detector=None
        self.people_tracker_name='FastTracker (crowd-aware)'
        self.skipped=0;self.samples=deque(maxlen=60);self.latencies=deque(maxlen=60)
        self.snapshot={'tracks':[],'raw':[],'sequence':0,'timestamp':None,'width':None,'height':None}
        self.metrics=dict(inference_ms=None,tracking_ms=None,analytics_ms=None,inference_fps=None,skipped_frames=0,resolution=None)
        self.clients=0
        self.person_frame_listener = None
        self.analysis_focus = 'traffic'
        self.cached_summary={};self.summary_stamp=0
    def start(self):self.thread.start()
    def configure(self,value):
        settings=validate_settings(value)
        with self.lock:self.settings=settings
        self.storage.save('settings',settings)
        return settings
    def set_geometry(self,value):
        with self.lock:
            self.geometry=value
            self.engine.set_geometry(value)
        self.storage.save('geometry',value)
    def _run(self):
        detector=Detector(self.models_dir);self.detector=detector
        subtype_manifest=self.models_dir/'subtypes.json'
        if subtype_manifest.exists():
            try:detector.subtype=SubtypeClassifier(subtype_manifest,self.models_dir)
            except Exception as exc:LOG.warning('Optional subtype classifier disabled: %s',type(exc).__name__)
        last=0;tracker=None;person_tracker=None;key=None;epoch=None;next_inference=0
        while not self.stopping.is_set():
            settings=copy.deepcopy(self.settings)
            if not settings['enabled']:
                self.status='paused';self.stopping.wait(.2);continue
            packet=self.stream.frames.wait(last,.5)
            if packet is None:
                self.status='waiting_for_frames'
                continue
            delay=next_inference-time.monotonic()
            if delay>0 and self.stopping.wait(delay):break
            packet=self.stream.frames.latest
            if packet is None:continue
            try:
                self.status='loading_model' if detector.key!=(settings['model'],settings['backend']) else 'running'
                detector.load(settings)
                # Loading/warming a model can take seconds. Do not infer the old
                # frame held before that work, or publish stale boxes after it.
                packet=self.stream.frames.latest
                if packet is None or packet.sequence<=last:continue
                if self.stopping.is_set():break
                newkey=(packet.epoch,settings['model'],settings['backend'],settings['tracker'],settings['reid'],settings['lifetime'])
                if key!=newkey:
                    tracker=VehicleTracker(settings,self.models_dir,self.stream.status()['fps'] or 25)
                    person_tracker=VehicleTracker(settings,self.models_dir,self.stream.status()['fps'] or 25,person_mode=True)
                    self.people_tracker_name=person_tracker.name
                    epoch=f'{packet.epoch}-{uuid.uuid4().hex[:6]}'
                    with self.lock:self.engine.reset_session(epoch);self.people.reset(epoch)
                    key=newkey
                    self.samples.clear();self.latencies.clear()
                schedule_start=time.monotonic()
                start=time.perf_counter()
                with self.lock:geometry=copy.deepcopy(self.geometry)
                boxes=detector.predict(packet.image,settings,geometry)
                inferred=time.perf_counter()
                fps=self.stream.status()['fps'] or 25
                rows=tracker.update(boxes,packet.image,packet.sequence,fps)
                person_boxes=getattr(detector,'people_boxes',None)
                if person_boxes is None:
                    person_boxes=Boxes(np.empty((0,6),np.float32),packet.image.shape[:2])
                if self.person_frame_listener is not None:
                    self.person_frame_listener(packet,person_boxes,(inferred-start)*1000)
                person_rows=person_tracker.update(person_boxes,packet.image,packet.sequence,fps)
                tracked=time.perf_counter()
                with self.lock:
                    tracks=self.engine.update(rows,packet.timestamp,packet.image.shape[1],packet.image.shape[0],settings,motion_stamp=packet.media_time)
                    people=self.people.update(person_rows,packet.timestamp,packet.image.shape[1],packet.image.shape[0],
                        lifetime=settings['lifetime'],confirm_hits=settings['person_confirm_hits'])
                    self.lane_learner.observe(packet.epoch,tracks)
                    if detector.subtype:
                        for obj in tracks:
                            target=self.engine.tracks[obj['uid']]
                            if target.confirmed and target.hits%15==0:
                                x1,y1,x2,y2=np.clip(target.bbox,0,[packet.image.shape[1],packet.image.shape[0]]*2).astype(int)
                                target.subtype,target.subtype_confidence=detector.subtype.classify(packet.image[y1:y2,x1:x2])
                    self.snapshot=dict(tracks=tracks,people=people,person_detections=len(person_boxes),raw=[dict(bbox=(row/([packet.image.shape[1],packet.image.shape[0]]*2)).tolist(),confidence=float(score)) for row,score in zip(boxes.xyxy,boxes.conf)],
                        sequence=packet.sequence,timestamp=packet.timestamp,media_time=packet.media_time,
                        width=packet.image.shape[1],height=packet.image.shape[0])
                completed=time.perf_counter()
                if last:self.skipped+=max(0,packet.sequence-last-1)
                last=packet.sequence
                self.samples.append(time.monotonic())
                self.latencies.append((completed-start)*1000)
                self.metrics=dict(inference_ms=(inferred-start)*1000,tracking_ms=(tracked-inferred)*1000,
                    analytics_ms=(completed-tracked)*1000,inference_fps=(len(self.samples)-1)/(self.samples[-1]-self.samples[0]) if len(self.samples)>1 else None,
                    analytics_latency_ms=max(0,(time.monotonic()-packet.monotonic)*1000),skipped_frames=self.skipped,
                    resolution=detector.resolution,device=detector.device,backend=detector.backend,reid=tracker.reid,
                    torch_threads=getattr(detector,'cpu_threads',None),
                    preprocess_ms=getattr(detector,'timings',{}).get('preprocess'),
                    model_ms=getattr(detector,'timings',{}).get('inference'),
                    postprocess_ms=getattr(detector,'timings',{}).get('postprocess'),
                    latency_p95_ms=float(np.percentile(self.latencies,95)))
                detector.adapt((inferred-start)*1000,settings)
                self.warning=detector.warning or tracker.warning or person_tracker.warning
                self.status='running';self.error=None
                # The selected monitoring view gets the CPU budget. Keep a
                # one-Hz vehicle context pass while direct face analysis runs.
                next_inference=schedule_start+(1.0 if self.analysis_focus=='people' else 1/settings['max_fps'])
            except Exception as exc:
                self.status='error';self.error=f'{type(exc).__name__}: {str(exc)[:200]}'
                LOG.exception('AI processing failed')
                tracker=None;key=None
                self.stopping.wait(3)
    def state(self):
        with self.lock:
            snap=compact_frame(self.snapshot)
            geometry=copy.deepcopy(self.geometry)
            settings=copy.deepcopy(self.settings)
            retained_confirmed=sum(t.confirmed for t in self.engine.tracks.values())
            people_snapshot=self.people.state()
        age=time.time()-snap['timestamp'] if snap['timestamp'] else None
        tracks=[t for t in snap['tracks'] if t['confirmed']] if age is not None and age<2 and self.status=='running' else []
        people=people_snapshot if age is not None and age<2 and self.status=='running' else dict(visible=0,unique=people_snapshot['unique'],tracks=[],session=people_snapshot['session'])
        if time.monotonic()-self.summary_stamp>2:
            self.cached_summary=self.storage.summary()
            self.summary_stamp=time.monotonic()
        lanes=Counter(t['lane_name'] or 'Unassigned' for t in tracks)
        # Ground occupancy proxy is box footprint / configured road area, never physical occupancy.
        occupancy=None
        if tracks:
            mask=np.zeros((216,384),np.uint8)
            for t in tracks:
                x1,y1,x2,y2=np.asarray(t['bbox'])*[384,216,384,216]
                cv2.rectangle(mask,(round(x1),round(y1)),(round(x2),round(y2)),1,-1)
            road=np.zeros_like(mask)
            for item in geometry['items']:
                if item['kind']=='road':cv2.fillPoly(road,[(np.asarray(item['points'])*[384,216]).astype(np.int32)],1)
            if road.any():occupancy=float(np.count_nonzero(mask & road)/np.count_nonzero(road)*100)
        stopped=sum(t['stopped_seconds']>=settings['stopped_seconds'] for t in tracks)
        return dict(frame=snap,geometry=geometry,settings=settings,stream=self.stream.status(),
            engine=dict(state=self.status,error=self.error,warning=self.warning,weights_ready=bool(self.detector and self.detector.model),people_tracker=self.people_tracker_name),
            performance={**self.metrics,**self.performance.sample(), 'connected_clients':self.clients},
            analytics={**self.cached_summary,'visible':len(tracks),'active_tracks':len(tracks),
                'lost_tracks':max(0,retained_confirmed-len(tracks)),
                'visible_classes':dict(Counter(t['category'] for t in tracks)),'lanes':dict(lanes),'stopped':stopped,
                'queue_estimate':stopped if geometry['items'] else None,'box_occupancy_estimate_percent':occupancy},
            people=people,events=self.storage.events(60),persistence=dict(error=self.storage.error,queue_size=self.storage.pending.qsize()))
    def close(self):
        self.stopping.set();self.thread.join(15)
        with self.lock:
            for track in self.engine.tracks.values():
                if track.confirmed:self.engine.persist(track)
