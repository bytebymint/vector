"""One local perception process with a bounded shared-memory frame slot.

Torch, Flask and video acquisition stay in the parent. No second source is
opened. Each request completes against its exact source frame before the slot
can be overwritten; intermediate camera frames are skipped by the caller.
"""
from __future__ import annotations
import json
from multiprocessing import shared_memory
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import numpy as np


class FaceWorkerClient:
    def __init__(self, models_dir: Path):
        self.models_dir=Path(models_dir)
        self.process=None
        self.memory=None
        self.responses=queue.Queue(maxsize=1)
        self.reader=None
        self._last_shm_seq=None
        self._last_shm_epoch=None

    def _start(self):
        self.responses=queue.Queue(maxsize=1)
        self._last_shm_seq=None
        self._last_shm_epoch=None
        process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(self.models_dir)],
                                 stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                                 text=True,bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.process=process
        responses=self.responses
        def read():
            for line in process.stdout:
                try:responses.put(json.loads(line),timeout=1)
                except (ValueError,queue.Full):continue
            try:responses.put({'error':'Face worker exited'},timeout=1)
            except queue.Full:pass
        self.reader=threading.Thread(target=read,name='face-worker-results',daemon=True)
        self.reader.start()

    def analyze(self, packet, bodies, epoch, config, fast_only: bool = False, extra_hints=None):
        from services.face_contracts import PersonTrack,FaceObservation,FacePose,FaceQualityResult
        if self.process is None or self.process.poll() is not None:
            self.close()
            self._start()
        if self.memory is None or self.memory.size < packet.image.nbytes:
            if self.memory is not None:
                self.memory.close();self.memory.unlink()
            self.memory=shared_memory.SharedMemory(create=True,size=packet.image.nbytes)
            self._last_shm_seq=None
            self._last_shm_epoch=None
        if packet.sequence is None or packet.sequence != self._last_shm_seq or epoch != self._last_shm_epoch:
            np.copyto(np.ndarray(packet.image.shape,np.uint8,buffer=self.memory.buf),packet.image)
            self._last_shm_seq=packet.sequence
            self._last_shm_epoch=epoch
        serialized_hints=[
            list(h.bbox) if hasattr(h,'bbox') else (list(h) if isinstance(h,tuple) else h)
            for h in (extra_hints or [])
        ]
        request={'memory':self.memory.name,'shape':list(packet.image.shape),'sequence':packet.sequence,
                 'timestamp':packet.timestamp,'epoch':epoch,'config':config,
                 'bodies':[t.to_dict() if hasattr(t,'to_dict') else dict(t) for t in bodies],
                 'fast_only':bool(fast_only),'extra_hints':serialized_hints}
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        try:result=self.responses.get(timeout=15.)
        except queue.Empty:
            self.close()
            raise RuntimeError('Face worker exceeded 15-second timeout')
        if result.get('error'):
            raise RuntimeError(result['error'])
        tracks=[PersonTrack(**{k:v for k,v in row.items() if k!='duration'}) for row in result['tracks']]
        observations=[]
        for row in result['faces']:
            if row.get('pose'):row['pose']=FacePose(**row['pose'])
            if row.get('quality'):row['quality']=FaceQualityResult(**row['quality'])
            observations.append(FaceObservation(**row))
        return tracks,observations,result['coverage']

    def close(self):
        process,self.process=self.process,None
        self._last_shm_seq=None
        self._last_shm_epoch=None
        if process is not None:
            try:
                if process.poll() is None:
                    process.stdin.write('{"shutdown":true}\n');process.stdin.flush()
                    try:process.wait(timeout=2)
                    except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=2)
            except (OSError,ValueError):pass
            if self.reader is not None:
                self.reader.join(timeout=1)
            for pipe in (process.stdin,process.stdout):
                try:pipe.close()
                except OSError:pass
        if self.memory is not None:
            self.memory.close();self.memory.unlink();self.memory=None


def run_worker(models_dir):
    import time
    import cv2
    from types import SimpleNamespace
    from services.face_contracts import FramePacket
    from services.face_detector import FaceDetector,FaceDetectorConfig
    from services.face_coverage import FaceCoverage
    from services.face_pose import FacePoseEstimator
    from services.face_quality import FaceQualityAssessor,QualityThresholds
    from services.face_verifier import FaceVerifier
    cv2.setNumThreads(8)
    model=Path(models_dir)/'face_detection_yunet_2026may.onnx'
    runtime='onnxruntime'
    if not model.exists():model=Path(models_dir)/'face_detection_yunet_2023mar.onnx';runtime='opencv'
    detector=FaceDetector(FaceDetectorConfig(model_path=str(model),runtime=runtime,
                                           score_threshold=.5,roi_score_threshold=.5,
                                           max_inference_dim=1280,extract_face_crops=False))
    from services.face_detector import box_iou,is_valid_face
    engine=FaceCoverage(detector)
    verifier=FaceVerifier(Path(models_dir))
    pose=FacePoseEstimator(method='geometric')
    quality=FaceQualityAssessor(QualityThresholds(min_score=55,min_width=48,min_height=48,min_sharpness=5,max_motion_blur=.80))
    verified_tracks: dict[int,float]={}
    best_track_quality: dict[int,float]={}
    memory=None
    image=packet=None
    try:
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('shutdown'):break
            try:
                if memory is None or memory.name!=request['memory']:
                    image=packet=None
                    if memory:memory.close()
                    memory=shared_memory.SharedMemory(name=request['memory'])
                image=np.ndarray(tuple(request['shape']),np.uint8,buffer=memory.buf)
                h,w=image.shape[:2]
                packet=FramePacket(request['sequence'],request['timestamp'],image,'camera',request['epoch'],w,h)
                if engine._epoch!=packet.epoch:
                    engine.reset(packet.epoch)
                    verified_tracks.clear()
                    best_track_quality.clear()
                config=request['config']
                detector.config.score_threshold=detector.config.roi_score_threshold=float(config['face_confidence_threshold'])
                quality.thresholds.min_score=float(config['quality_threshold'])
                quality.thresholds.min_width=quality.thresholds.min_height=int(config['min_face_pixels'])
                bodies=[SimpleNamespace(**row) for row in request['bodies']]
                fast_only=bool(request.get('fast_only',False))
                extra_hints=request.get('extra_hints') or ()
                try:
                    observations=engine.detect(packet,bodies,fast_only=fast_only,extra_hints=extra_hints)
                except TypeError:
                    try:
                        observations=engine.detect(packet,bodies,fast_only=fast_only)
                    except TypeError:
                        observations=engine.detect(packet,bodies)
                proposal_count=len(observations)
                verification_started=time.perf_counter()
                if engine.tracker.active_tracks:
                    # Verified-track continuity: already-confirmed tracks verified within 1.2s
                    # skip redundant Caffe SSD unless they qualify as a new/multi-angle capture upgrade.
                    recent_confirmed=[
                        (t.track_id,engine._flow_boxes.get(t.track_id,t.bbox))
                        for t in engine.tracker.active_tracks
                        if t.confirmed and packet.timestamp-verified_tracks.get(t.track_id,-1e9)<=1.2
                    ]
                    coasted=[]
                    need_verify=[]
                    for obs in observations:
                        x1,y1,x2,y2=obs.bbox
                        tight=image[max(0,int(y1)):min(h,int(np.ceil(y2))),max(0,int(x1)):min(w,int(np.ceil(x2)))]
                        if (obs.confidence>=.65 and len(obs.landmarks)==5
                                and is_valid_face(tight,obs.landmarks,obs.bbox)
                                and any(box_iou(obs.bbox,tbox)>=.40 for _,tbox in recent_confirmed)):
                            obs.face_crop=None
                            coasted.append(obs)
                        else:
                            need_verify.append(obs)
                    verified_new=verifier.validate(image,need_verify) if need_verify else []
                    observations=coasted+verified_new
                else:
                    observations=verifier.validate(image,observations)
                tracks,observations=engine.update(observations,packet.timestamp,packet.epoch)
                for obs in observations:
                    if obs.verification_method=='yunet+res10ssd-v1' and obs.verification_score>=.90:
                        verified_tracks[obs.track_id]=packet.timestamp
                started=time.perf_counter()
                upgrade_candidates=[]
                for obs in observations:
                    if obs.landmarks:obs.pose=pose.estimate_observation(obs)
                    x1,y1,x2,y2=obs.bbox
                    left,top=max(0,int(x1)),max(0,int(y1))
                    tight=image[top:min(h,int(np.ceil(y2))),left:min(w,int(np.ceil(x2)))]
                    obs.quality=quality.assess(tight,pose=obs.pose,
                                              landmarks=[(x-left,y-top) for x,y in obs.landmarks],
                                              landmark_confidence=obs.confidence,bbox=obs.bbox)
                    pose_cat=obs.pose.category if obs.pose else 'unknown'
                    q_key=(obs.track_id,pose_cat)
                    if (obs.verification_method!='yunet+res10ssd-v1'
                            and obs.quality.is_acceptable
                            and obs.quality.score>=best_track_quality.get(q_key,-1.)+2.0):
                        upgrade_candidates.append(obs)
                quality_elapsed=(time.perf_counter()-started)*1000
                if upgrade_candidates:
                    v2_start=time.perf_counter()
                    verified_upgrades=verifier.validate(image,upgrade_candidates)
                    for u_obs in verified_upgrades:
                        verified_tracks[u_obs.track_id]=packet.timestamp
                        u_key=(u_obs.track_id,u_obs.pose.category if u_obs.pose else 'unknown')
                        best_track_quality[u_key]=max(best_track_quality.get(u_key,0.),u_obs.quality.score)
                else:
                    v2_start=None
                for obs in observations:
                    if (obs.verification_method=='yunet+res10ssd-v1' and obs.quality and obs.quality.is_acceptable):
                        o_key=(obs.track_id,obs.pose.category if obs.pose else 'unknown')
                        best_track_quality[o_key]=max(best_track_quality.get(o_key,0.),obs.quality.score)
                verification_ms=((started-verification_started)*1000 + ((time.perf_counter()-v2_start)*1000 if v2_start else 0.))
                coverage=dict(engine.diagnostics,quality_ms=round(quality_elapsed,1),
                              candidate_faces=proposal_count,verified_faces=len(observations),
                              rejected_candidates=proposal_count-len(observations),verification_ms=round(verification_ms,1),
                              detected_faces=len(observations),
                              worker_pid=os.getpid())
                print(json.dumps({'tracks':[t.to_dict() for t in tracks],
                                  'faces':[o.to_dict() for o in observations],'coverage':coverage}),flush=True)
                tight=image=packet=None
            except Exception as exc:
                print(json.dumps({'error':f'{type(exc).__name__}: {exc}'}),flush=True)
    finally:
        image=packet=None
        if memory:memory.close()


if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    if len(sys.argv)==3 and sys.argv[1]=='--worker':run_worker(sys.argv[2])
