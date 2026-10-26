"""Version-checked persistent Ultralytics trackers with source-frame cadence."""
from types import SimpleNamespace
import inspect
import importlib.util
from pathlib import Path
import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.bot_sort import BOTSORT
try:
    # Ultralytics 8.4.183 includes its lightweight, crowd-aware tracker.
    from ultralytics.trackers import FASTTracker
except ImportError:  # Older local installs retain the ByteTrack fallback.
    FASTTracker = None


class VehicleTracker:
    def __init__(self, settings, models_dir, source_fps=25, person_mode=False):
        self.settings=settings
        self.last_sequence=None
        self.warning=None
        person_confidence=float(settings.get('person_confidence',.08))
        person_high=min(.95,max(.12,person_confidence*1.5))
        person_low=max(.02,min(.08,person_confidence*.5))
        reid_path=Path(models_dir)/'yolo26n-reid.onnx'
        reid=bool(not person_mode and settings['reid'] and settings['tracker']=='botsort' and reid_path.exists() and importlib.util.find_spec('onnxruntime'))
        if settings['reid'] and not person_mode and not reid: self.warning='Re-ID needs BoT-SORT, ONNX Runtime and models/yolo26n-reid.onnx; disabled'
        args=SimpleNamespace(track_high_thresh=person_high if person_mode else .20,
            track_low_thresh=person_low if person_mode else .06,new_track_thresh=person_confidence if person_mode else .15,
            track_buffer=round(source_fps*settings['lifetime']),match_thresh=.75,fuse_score=True,
            gmc_method='none',proximity_thresh=.5,appearance_thresh=.8,with_reid=reid,
            model=str(reid_path),device='cpu',
            # FastTracker's occlusion recovery and spawn suppression are useful
            # for dense pedestrian scenes, and add only small vectorized overhead.
            reset_velocity_offset_occ=5,reset_pos_offset_occ=3,enlarge_bbox_occ=1.1,
            dampen_motion_occ=.5,active_occ_to_lost_thresh=12,init_iou_suppress=.55,
            occ_cover_thresh=.55,occ_reappear_window=max(20,round(source_fps*1.5)))
        cls=(FASTTracker if person_mode and FASTTracker is not None else
             BYTETracker if person_mode or settings['tracker']=='bytetrack' else BOTSORT)
        if person_mode and FASTTracker is None:
            self.warning='Installed Ultralytics has no FastTracker; person tracking uses ByteTrack'
        self.tracker=cls(args) if 'frame_rate' not in inspect.signature(cls).parameters else cls(args,frame_rate=source_fps)
        self.reid=reid
        self.name='FastTracker' if cls is FASTTracker else ('ByteTrack' if cls is BYTETracker else 'BoT-SORT')

    def update(self, boxes, image, sequence, source_fps=25):
        # Kalman state advances in decoded-source frame units, including skipped AI samples.
        # No interpolated observation is emitted or counted during these prediction steps.
        buffer=round(source_fps*self.settings['lifetime'])
        if hasattr(self.tracker,'max_frames_lost'): self.tracker.max_frames_lost=buffer
        if hasattr(self.tracker,'max_time_lost'): self.tracker.max_time_lost=buffer
        gap=sequence-self.last_sequence-1 if self.last_sequence is not None else 0
        skipped=min(max(gap,0),max(buffer+1,120))
        for _ in range(skipped):
            self.tracker.multi_predict(self.tracker.tracked_stracks+self.tracker.lost_stracks)
        self.tracker.frame_id+=skipped
        self.last_sequence=sequence
        return self.tracker.update(boxes,img=image)
