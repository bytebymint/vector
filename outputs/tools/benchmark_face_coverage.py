"""Replay selected manually annotated faces; no camera or network required.

Usage: python tools/benchmark_face_coverage.py annotations.json --output results.json
Annotations: [{"image":"frame.jpg", "boxes":[[x1,y1,x2,y2], ...]}].
Reports selected-target coverage over six static search passes, NOT dataset AP,
an exhaustive face census, or temporal tracking accuracy. Other detections may
be real unannotated faces and must be reviewed separately.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from services.face_contracts import FramePacket
from services.face_coverage import FaceCoverage, merge_faces
from services.face_detector import FaceDetector, FaceDetectorConfig, box_iou


def benchmark(annotations: Path, rounds: int = 6):
    model_dir=Path(__file__).resolve().parents[1]/'models'
    model=model_dir/'face_detection_yunet_2026may.onnx'
    runtime='onnxruntime'
    if not model.exists():
        model=model_dir/'face_detection_yunet_2023mar.onnx'
        runtime='opencv'
    cv2.setNumThreads(2)
    detector=FaceDetector(FaceDetectorConfig(model_path=str(model),runtime=runtime,
                                           score_threshold=.5,roi_score_threshold=.5,max_inference_dim=960))
    results=[]
    for entry in json.loads(annotations.read_text(encoding='utf-8')):
        image=cv2.imread(str(annotations.parent/entry['image']))
        if image is None:
            raise ValueError(f"Unreadable fixture {entry['image']}")
        h,w=image.shape[:2]
        engine=FaceCoverage(detector);engine.reset('benchmark')
        observations=[];times=[];first=[]
        for n in range(rounds):
            packet=FramePacket(n,n*.2,image,'benchmark','benchmark',w,h)
            start=time.perf_counter()
            detected=engine.detect(packet)
            engine.update(detected,packet.timestamp,'benchmark')
            times.append((time.perf_counter()-start)*1000)
            if n==0:first=detected
            observations.extend(detected)
        # Target-wise union across passes: the fixture remains static during
        # rotating search. These results must not be reported as per-frame recall.
        hits=[any(box_iou(tuple(box),o.bbox)>=.25 for o in observations) for box in entry['boxes']]
        initial=[any(box_iou(tuple(box),o.bbox)>=.25 for o in first) for box in entry['boxes']]
        results.append({'image':entry['image'],'selected_targets':len(hits),'matched_over_passes':sum(hits),
                        'matched_first_pass':sum(initial),'target_matches':hits,
                        'mean_warm_ms':round(float(np.mean(times[1:])),1),
                        'p95_warm_ms':round(float(np.percentile(times[1:],95)),1),
                        'passes':rounds,'detections_for_review':[o.to_dict() for o in merge_faces(observations)]})
    return {'scope':'Selected targets over repeated static search passes; not overall accuracy or temporal tracking recall',
            'runtime':detector.config.runtime,'results':results}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotations',type=Path)
    parser.add_argument('--output',type=Path,default=Path('face_coverage_benchmark.json'))
    args=parser.parse_args()
    result=benchmark(args.annotations)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({**result,'results':[{k:v for k,v in r.items() if k!='detections_for_review'} for r in result['results']]},indent=2))
