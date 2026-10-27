"""Independent semantic face verification; image sharpness is not face evidence."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from services.face_detector import box_iou, is_valid_face

VERIFICATION_METHOD = 'yunet+res10ssd-v1'


@dataclass(frozen=True)
class FaceVerification:
    accepted: bool
    confidence: float = 0.
    bbox: tuple[float,float,float,float] | None = None
    reason: str = 'not_verified'


class FaceVerifier:
    def __init__(self, models_dir: Path, threshold: float = .90):
        self.models_dir=Path(models_dir)
        self.threshold=float(threshold)
        self.net=None
        self._crop_bufs: list[np.ndarray] | None = None

    def _load(self):
        if self.net is not None:return
        config=self.models_dir/'face_ssd.prototxt'
        weights=self.models_dir/'face_ssd_fp16.caffemodel'
        if not config.is_file() or not weights.is_file():
            raise RuntimeError('Independent face verifier weights missing; capture is disabled')
        self.net=cv2.dnn.readNetFromCaffe(str(config),str(weights))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def verify_batch(self, image: np.ndarray, observations) -> list[FaceVerification]:
        results=[FaceVerification(False,reason='invalid_geometry') for _ in observations]
        if not observations:return results
        self._load()
        h,w=image.shape[:2]
        regions=[]
        for index,obs in enumerate(observations):
            x1,y1,x2,y2=map(float,obs.bbox)
            fw,fh=x2-x1,y2-y1
            if obs.confidence < .65 or min(fw,fh)<12 or len(obs.landmarks)!=5:
                continue
            tight=image[max(0,int(y1)):min(h,int(np.ceil(y2))),max(0,int(x1)):min(w,int(np.ceil(x2)))]
            if not is_valid_face(tight,obs.landmarks,obs.bbox):continue
            side=max(fw,fh)*1.8
            cx,cy=(x1+x2)/2,(y1+y2)/2
            left,top=max(0,int(cx-side/2)),max(0,int(cy-side/2))
            right,bottom=min(w,int(np.ceil(cx+side/2))),min(h,int(np.ceil(cy+side/2)))
            if right<=left or bottom<=top:continue
            regions.append((index,left,top,right,bottom,side))
        if not regions:
            return results
        crop_bufs = getattr(self, '_crop_bufs', None)
        if crop_bufs is None:
            crop_bufs = [np.empty((300, 300, 3), dtype=np.uint8) for _ in range(16)]
            self._crop_bufs = crop_bufs
        for offset in range(0,len(regions),16):
            batch=regions[offset:offset+16]
            crops=[]
            for k,(_,l,t,r,b,side) in enumerate(batch):
                raw_crop=image[t:b,l:r]
                if raw_crop.ndim==2:
                    raw_crop=cv2.cvtColor(raw_crop,cv2.COLOR_GRAY2BGR)
                if raw_crop.shape[0]==300 and raw_crop.shape[1]==300:
                    crops.append(raw_crop)
                else:
                    crops.append(cv2.resize(raw_crop,(300,300),dst=crop_bufs[k],interpolation=cv2.INTER_LINEAR))
            blob=cv2.dnn.blobFromImages(crops,1.,(300,300),(104.,177.,123.),swapRB=False,crop=False)
            self.net.setInput(blob)
            detections=self.net.forward().reshape(-1,7)
            for index,_,_,_,_,_ in batch:
                results[index]=FaceVerification(False,reason='independent_model_rejected')
            valid_rows=detections[(detections[:,1]==1)&(detections[:,2]>=self.threshold)]
            for row in valid_rows:
                local=int(row[0])
                if local<0 or local>=len(batch):continue
                index,left,top,right,bottom,_=batch[local]
                score=float(row[2])
                if score<=results[index].confidence:continue
                a,b,c,d=np.clip(row[3:7],0.,1.)
                box=(left+float(a)*(right-left),top+float(b)*(bottom-top),
                     left+float(c)*(right-left),top+float(d)*(bottom-top))
                original=observations[index].bbox
                ow,oh=original[2]-original[0],original[3]-original[1]
                nw,nh=box[2]-box[0],box[3]-box[1]
                ratio=(nw*nh)/max(1.,ow*oh)
                shift=np.hypot((box[0]+box[2]-original[0]-original[2])/2,
                               (box[1]+box[3]-original[1]-original[3])/2)
                if box_iou(box,original)<.35 or not .4<=ratio<=2.5 or shift>max(ow,oh)*.35:
                    continue
                refined=tuple((float(x)+float(y))/2 for x,y in zip(original,box))
                results[index]=FaceVerification(True,score,refined,'two_model_agreement')
        return results

    def validate(self, image, observations):
        verified=[]
        for obs,result in zip(observations,self.verify_batch(image,observations)):
            if not result.accepted:continue
            obs.bbox=result.bbox
            obs.verification_score=result.confidence
            obs.verification_method=VERIFICATION_METHOD
            obs.face_crop=None  # extract the refined box from this exact native frame
            verified.append(obs)
        return verified
