"""Reproducible actual inference benchmark. Never inserts events into the live database."""
import argparse
import json
import platform
from pathlib import Path
import statistics
import time
import cv2
import numpy as np
import torch
from services.detector import Detector,DEFAULTS,validate_settings
from services.tracker import VehicleTracker

def main():
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image',type=Path,help='Repeat a decoded real image (throughput only)')
    source.add_argument('--video',type=Path,help='Process successive prerecorded video frames')
    parser.add_argument('--model',choices=['yolo26n','yolo26s'],default='yolo26n')
    parser.add_argument('--resolution',type=int,default=640)
    parser.add_argument('--iterations',type=int,default=30)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--stride',type=int,default=1,help='Source-frame stride when reading a video')
    parser.add_argument('--threads',type=int,help='Override the detector CPU thread budget')
    parser.add_argument('--backend',choices=['pytorch','onnx','tensorrt'],default='pytorch')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'benchmark-result.json')
    args=parser.parse_args()
    if args.iterations<1 or args.warmup<0 or args.stride<1 or (args.threads is not None and args.threads<1):
        parser.error('iterations, stride and threads must be positive; warmup must be nonnegative')
    images=[];source_fps=25
    if args.image:
        image=cv2.imread(str(args.image))
        if image is None:raise SystemExit('Image could not be decoded')
        images=[image]
    else:
        capture=cv2.VideoCapture(str(args.video))
        try:
            source_fps=capture.get(cv2.CAP_PROP_FPS) or 25
            for index in range((args.iterations+args.warmup)*args.stride):
                ok,image=capture.read()
                if not ok:raise SystemExit('Video has too few decodable frames; reduce iterations/stride')
                if index%args.stride==0:images.append(image)
        finally:capture.release()
    root=Path(__file__).parent
    detector=Detector(root/'models',cpu_threads=args.threads)
    settings=validate_settings({**DEFAULTS,'model':args.model,'backend':args.backend,'resolution':args.resolution,'adaptive':False})
    detector.load(settings);tracker=VehicleTracker(settings,root/'models',source_fps)
    times=[];track_times=[];detections=[];stages=[];ids=set()
    for index in range(args.iterations+args.warmup):
        image=images[index] if args.video else images[0]
        start=time.perf_counter();boxes=detector.predict(image,settings,{'items':[]});mid=time.perf_counter()
        rows=tracker.update(boxes,image,index*args.stride+1,source_fps);end=time.perf_counter()
        if index>=args.warmup:
            times.append((mid-start)*1000);track_times.append((end-mid)*1000);detections.append(len(boxes))
            stages.append(detector.timings)
            ids.update(int(row[4]) for row in rows)
    result=dict(model=args.model,resolution=args.resolution,source_shape=list(image.shape),device=detector.device,
        backend=detector.backend,torch_version=torch.__version__,cpu=platform.processor(),torch_threads=torch.get_num_threads(),
        iterations=args.iterations,inference_mean_ms=statistics.mean(times),inference_p95_ms=float(np.percentile(times,95)),
        tracker_mean_ms=statistics.mean(track_times),serial_processing_fps=1000/(statistics.mean(times)+statistics.mean(track_times)),
        detections_per_frame=statistics.mean(detections),tracked_ids=len(ids),
        stage_mean_ms={name:statistics.mean(s[name] for s in stages if name in s) for name in ('preprocess','inference','postprocess') if any(name in s for s in stages)},
        source_kind='successive_video_frames' if args.video else 'repeated_real_frame',source=str(args.video or args.image),stride=args.stride,
        note='Excludes decoding, network, browser relay and event persistence. Throughput benchmark, not a tracking accuracy evaluation.')
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
