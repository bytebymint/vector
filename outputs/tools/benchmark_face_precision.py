"""Replay the real face-capture path on reviewed local fixtures, offline.

Usage: python tools/benchmark_face_precision.py --output data/face_precision_report.json
This is a small regression sample, not a general accuracy or video FPS claim.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import tempfile

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from services.face_contracts import FramePacket
from services.face_detector import box_iou
from services.face_worker import FaceWorkerClient
from services.people_orchestrator import PeopleSystemOrchestrator


def benchmark(output: Path):
    fixtures=ROOT/'tests/fixtures/face_precision'
    manifest=json.loads((fixtures/'manifest.json').read_text())
    image=cv2.imread(str(fixtures/manifest['positive_frame']))
    h,w=image.shape[:2]
    with tempfile.TemporaryDirectory(prefix='face-precision-',dir=output.parent) as directory:
        orch=PeopleSystemOrchestrator(data_dir=Path(directory))
        orch.face_worker=FaceWorkerClient(ROOT/'models')
        orch.start(epoch_id='offline-precision')
        samples=[]
        all_faces=[]
        saved=[]
        try:
            for n in range(6):
                packet=FramePacket(n,100.+n*.4,image,'fixture','fixture',w,h)
                start=time.perf_counter()
                result=orch.process_frame(packet,shared_detections=[])
                elapsed=(time.perf_counter()-start)*1000
                all_faces.extend(result['faces'])
                samples.append({'sequence':n,'total_ms':round(elapsed,1),
                                'faces':len(result['faces']),'confirmed':result['active_count'],
                                'captures':result['captures_saved'],'coverage':result['coverage']})
            for profile in orch.repository.list_profiles(page_size=100):
                for capture in orch.repository.get_profile(profile.session_id).capture_history:
                    crop=cv2.imread(capture.original_path)
                    dest=output.parent/f'face-precision-{len(saved):02}.png'
                    cv2.imwrite(str(dest),crop)
                    saved.append({'path':str(dest),'native_size':[capture.crop_width,capture.crop_height],
                                  'quality':capture.quality_score})
            hits=[any(box_iou(box,o['bbox'])>.4 for o in all_faces) for box in manifest['selected_faces']]
            report={'scope':'Six static source-frame passes; reviewed selected targets, not full-scene recall or playback FPS',
                    'configuration':orch.get_config(),
                    'source_size':[w,h],'selected_targets':len(hits),'matched_selected_targets':sum(hits),
                    'warm_mean_ms':round(float(np.mean([s['total_ms'] for s in samples[1:]])),1),
                    'captures':saved,'passes':samples}
            output.write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in report.items() if k!='passes'},indent=2))
        finally:
            orch.stop()
            orch.face_enhancer.stop()
            orch.pipeline_scheduler.shutdown()
            orch.repository.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/face_precision_report.json')
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    benchmark(args.output.resolve())
