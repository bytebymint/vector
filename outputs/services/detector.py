"""Official local YOLO26 vehicle inference; subtype plugins must supply trained weights."""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from .road_geometry import anchor, inside

CLASSES = {1:'bicycle',2:'car',3:'motorcycle',5:'bus',7:'truck'}
CANDIDATE_CLASSES = [0, 1, 2, 3, 5, 7]
DEFAULTS = dict(model='yolo26n', tracker='bytetrack', backend='pytorch', enabled=True, resolution=960,
    adaptive=True, target_ms=180, max_fps=12, lifetime=2.5, confirm_hits=3,
    person_confidence=.08, person_confirm_hits=2,
    thresholds={'car':.30,'motorcycle':.15,'bicycle':.15,'bus':.35,'truck':.35},
    reid=False, stopped_seconds=12, obstruction_seconds=45, stationary_px_s=5, lane_hysteresis=3)


def validate_settings(value):
    out = {**DEFAULTS, **value}
    if out['model'] not in ('yolo26n','yolo26s') or out['tracker'] not in ('bytetrack','botsort') or out['backend'] not in ('pytorch','onnx','tensorrt'):
        raise ValueError('Unsupported model, tracker or backend')
    for key, low, high in [('resolution',320,1280),('target_ms',40,2000),('max_fps',1,30),('lifetime',.5,10),('confirm_hits',2,10),('person_confirm_hits',2,6),('stopped_seconds',3,300),('obstruction_seconds',10,600),('stationary_px_s',1,100),('lane_hysteresis',2,10)]:
        v = float(out[key])
        if not np.isfinite(v) or not low<=v<=high: raise ValueError(f'{key} must be {low}–{high}')
        out[key] = int(v) if key in ('resolution','confirm_hits','person_confirm_hits','lane_hysteresis') else v
    out['resolution'] = round(out['resolution']/32)*32
    out['person_confidence'] = float(out['person_confidence'])
    if not np.isfinite(out['person_confidence']) or not .05 <= out['person_confidence'] <= .95:
        raise ValueError('person_confidence must be 0.05–0.95')
    out['thresholds'] = {**DEFAULTS['thresholds'],**out.get('thresholds',{})}
    for name, value in out['thresholds'].items():
        if name not in CLASSES.values() or not .05 <= float(value) <= .95: raise ValueError('Invalid class threshold')
        out['thresholds'][name] = float(value)
    for key in ('enabled','adaptive','reid'): out[key] = bool(out[key])
    return out


class Detector:
    def __init__(self, models_dir, cpu_threads=None):
        self.directory = Path(models_dir)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # Reserve CPU capacity for camera decode, HTTP and browser composition.
        # Ultralytics resets this in select_device(), so apply it again after setup.
        physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        requested = cpu_threads if cpu_threads is not None else os.environ.get('TRAFFIC_TORCH_THREADS')
        self.cpu_threads = max(1, min(int(requested) if requested is not None else min(2, max(1, physical - 2)), os.cpu_count() or 1))
        torch.set_num_threads(self.cpu_threads)
        cv2.setNumThreads(1)
        self.model = None
        self.key = None
        self.backend = 'pytorch'
        self.warning = None
        self.resolution = 640
        self.configured_resolution = None
        self.samples = 0
        self.inference_ema_ms = None
        self.timings = {}
        self.subtype = None
        self.people_boxes = Boxes(np.empty((0, 6), np.float32), (432, 768))

    def load(self, settings):
        key = (settings['model'],settings['backend'])
        if key == self.key:
            # Resolution can change independently of model/backend. A previously
            # adaptive lower value must not prevent the requested increase applying.
            if self.configured_resolution != settings['resolution']:
                self.configured_resolution = settings['resolution']
                self.resolution = settings['resolution']
            return
        path = self.directory / (settings['model']+'.pt')
        backend = settings['backend']
        self.warning = None
        exported = self.directory/(settings['model']+('.onnx' if backend=='onnx' else '.engine'))
        available = (backend=='onnx' and importlib.util.find_spec('onnxruntime')) or (backend=='tensorrt' and self.device.startswith('cuda') and importlib.util.find_spec('tensorrt'))
        if backend != 'pytorch':
            if exported.exists() and available:
                path = exported
            else:
                self.warning = f'{backend} runtime/export unavailable; using PyTorch'
                backend = 'pytorch'
        self.model = YOLO(str(path),task='detect')
        self.backend = backend
        self.resolution = settings['resolution']
        self.configured_resolution = settings['resolution']
        self.model.predict(np.zeros((432,768,3),np.uint8),imgsz=self.resolution,device=self.device,
            quantize=16 if self.device.startswith('cuda') else 32,verbose=False,save=False,classes=CANDIDATE_CLASSES)
        if self.device == 'cpu':
            torch.set_num_threads(self.cpu_threads)
        self.samples = 0
        self.inference_ema_ms = None
        self.key = key

    def predict(self, image, settings, geometry):
        h, w = image.shape[:2]
        is_uhd = (w >= 1920 or h >= 1080)
        self.is_uhd = is_uhd
        if not settings['adaptive']:
            self.resolution = max(960, settings['resolution']) if is_uhd else settings['resolution']
        else:
            if is_uhd and self.resolution < 960:
                self.resolution = max(960, min(settings['resolution'], 1280)) if settings['resolution'] >= 960 else 960
            else:
                self.resolution = min(self.resolution, settings['resolution'])
        imgsz = max(960, self.resolution) if is_uhd else self.resolution
        result = self.model.predict(image,imgsz=imgsz,conf=.06,iou=.70,max_det=1000,
            classes=CANDIDATE_CLASSES,device=self.device,quantize=16 if self.device.startswith('cuda') else 32,verbose=False,save=False)[0]
        self.timings = dict(getattr(result, 'speed', {}) or {})
        boxes = result.boxes.cpu().numpy()
        if len(boxes) == 0:
            self.people_boxes = Boxes(np.empty((0, 6), np.float32), (h, w))
            return Boxes(np.empty((0, 6), np.float32), (h, w))

        two_wheelers = []
        other_vehicles = []
        persons = []
        for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
            c = int(cls)
            item = {'xyxy': np.array(box, dtype=np.float32), 'conf': float(conf), 'cls': c}
            if c in (1, 3):
                two_wheelers.append(item)
            elif c in (2, 5, 7):
                other_vehicles.append(item)
            elif c == 0:
                if item['conf'] >= float(settings.get('person_confidence', .08)):
                    persons.append(item)

        # A second tracker consumes people from this same inference pass. Do not
        # run a second model or let people enter the vehicle tracker.
        self.people_boxes = Boxes(
            np.asarray([[*p['xyxy'], p['conf'], 0] for p in persons], dtype=np.float32).reshape(-1, 6),
            (h, w))

        def overlap_stats(b1, b2):
            wh = np.maximum(0, np.minimum(b1[2:], b2[2:]) - np.maximum(b1[:2], b2[:2]))
            inter = float(wh[0] * wh[1])
            if inter <= 0:
                return 0.0, 0.0, 0.0
            a1 = float(max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1]))
            a2 = float(max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1]))
            union = a1 + a2 - inter
            iou = inter / max(union, 1e-6)
            return iou, inter / max(a1, 1e-6), inter / max(a2, 1e-6)

        # Merge person overlapping motorcycle or bicycle (IoU or intersection > 0.15)
        for p in persons:
            best_tw = None
            best_score = 0.0
            for tw in two_wheelers:
                iou, r_p, r_tw = overlap_stats(p['xyxy'], tw['xyxy'])
                score = max(iou, r_p, r_tw)
                if (iou > 0.15 or r_p > 0.15 or r_tw > 0.15) and score > best_score:
                    best_score = score
                    best_tw = tw
            if best_tw is not None:
                p['merged'] = True
                best_tw['xyxy'] = np.array([
                    min(best_tw['xyxy'][0], p['xyxy'][0]),
                    min(best_tw['xyxy'][1], p['xyxy'][1]),
                    max(best_tw['xyxy'][2], p['xyxy'][2]),
                    max(best_tw['xyxy'][3], p['xyxy'][3])
                ], dtype=np.float32)
                # Keep the actual vehicle detection confidence.

        # A person alone is not evidence of a motorcycle. Require a vehicle class.
        roads = [i['points'] for i in geometry.get('items', []) if i.get('kind') == 'road']
        candidates = []
        for cand in (two_wheelers + other_vehicles):
            box = cand['xyxy']
            if roads:
                bc = anchor(box, w, h)
                center = ((box[0] + box[2]) / (2 * w), (box[1] + box[3]) / (2 * h))
                bq = ((box[0] + box[2]) / (2 * w), (box[1] + 3 * box[3]) / (4 * h))
                if not any(inside(bc, p) or inside(center, p) or inside(bq, p) for p in roads):
                    continue
            candidates.append(cand)

        # Duplicate suppression: same-class IoU > 0.90
        keep = []
        for cand in candidates:
            duplicate = False
            box = cand['xyxy']
            for prev in keep:
                if cand['cls'] != prev['cls']:
                    continue
                other = prev['xyxy']
                wh = np.maximum(0, np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2]))
                intersection = float(np.prod(wh))
                union = float(np.prod(box[2:] - box[:2]) + np.prod(other[2:] - other[:2]) - intersection)
                if intersection / max(union, 1e-6) > 0.90:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(cand)

        if keep:
            data = np.array([[*c['xyxy'], c['conf'], c['cls']] for c in keep], dtype=np.float32)
            return Boxes(data, (h, w))
        return Boxes(np.empty((0, 6), np.float32), (h, w))

    def adapt(self, elapsed_ms, settings):
        self.samples += 1
        self.inference_ema_ms = elapsed_ms if self.inference_ema_ms is None else .15 * elapsed_ms + .85 * self.inference_ema_ms
        if settings['adaptive'] and self.samples%20==0:
            is_uhd = getattr(self, 'is_uhd', False)
            floor = 960 if is_uhd else min(512, settings['resolution'])
            if self.inference_ema_ms > settings['target_ms']*1.2:
                self.resolution = max(floor, self.resolution - 64)
            elif self.inference_ema_ms < settings['target_ms']*.65:
                ceil = max(1280, settings['resolution']) if is_uhd else settings['resolution']
                self.resolution = min(ceil, self.resolution + 64)


class SubtypeClassifier:
    """Opt-in local classifier, never infer sedan/taxi/etc from COCO classes.

    Manifest JSON: {weights: 'subtypes.pt', labels: [...], threshold: 0.85}.
    The operator is responsible for domain validation of these trained weights.
    """
    def __init__(self, manifest, directory):
        import json
        spec=json.loads(Path(manifest).read_text(encoding='utf-8'))
        path=(Path(directory)/spec['weights']).resolve()
        if not path.is_relative_to(Path(directory).resolve()) or not path.exists(): raise ValueError('Classifier weights must exist inside models/')
        self.model=YOLO(str(path),task='classify')
        self.labels=set(spec['labels'])
        self.threshold=max(.5,float(spec.get('threshold',.85)))
    def classify(self,crop):
        if not crop.size: return 'Unknown subtype',None
        result=self.model(crop,verbose=False)[0]
        label=result.names[result.probs.top1]
        confidence=float(result.probs.top1conf)
        return (label,confidence) if label in self.labels and confidence>=self.threshold else ('Unknown subtype',confidence)
