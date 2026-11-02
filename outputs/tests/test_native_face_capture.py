"""Regression checks for native face detail and quality gating."""
from types import SimpleNamespace
from fractions import Fraction
import threading
import time
import cv2
import numpy as np
import pytest

from services.capture_selector import CaptureSelector
from services.face_contracts import FaceObservation, FaceQualityResult
from services.face_detector import FaceDetector, FaceDetectorConfig
from services.profile_repository import ProfileRepository
from services.native_video import source_mime
from services.source_decoder import AnalyticsDecoder


def test_original_png_preserves_every_source_pixel(tmp_path):
    repo = ProfileRepository(tmp_path/'people.db', tmp_path/'captures')
    original = np.random.default_rng(42).integers(0, 256, (121,137,3), dtype=np.uint8)
    path = repo.write_image_atomic('camera:epoch:1', 'original.png', original)
    assert np.array_equal(cv2.imread(str(path)), original)


@pytest.mark.parametrize('fixed', [0,512])
def test_bounded_face_detection_maps_back_to_master_pixels(fixed):
    class YuNet:
        def setInputSize(self, size): self.size = size
        def setScoreThreshold(self, score): pass
        def detect(self, image):
            return None, np.array([[10,20,80,100,20,40,60,40,40,60,25,90,55,90,.95]], np.float32)
    detector = FaceDetector(FaceDetectorConfig(backend='haar', max_inference_dim=512,fixed_input_size=fixed))
    detector.active_backend, detector._yunet = 'yunet', YuNet()
    frame = np.random.default_rng(7).integers(0,256,(1024,2048,3),dtype=np.uint8)
    face = detector._detect_raw(frame)[0]
    assert detector._yunet.size == (512,512 if fixed else 256)
    assert np.allclose(face['bbox'], (40,80,360,480))
    assert np.allclose(face['landmarks'][0], (80,160))
    obs = FaceObservation('obs',1,1.0,face['bbox'],.95)
    candidate = CaptureSelector().create_candidate(obs, frame_image=frame)
    assert candidate.crop_original.shape == (580,440,3)  # native crop + margin, clipped at top/left


def test_unusable_face_cannot_reach_disk_or_finalization():
    quality = FaceQualityResult(90,100,.1,90,90,90,90,is_acceptable=False)
    obs = FaceObservation('bad',1,1.,(0,0,40,40),.95,quality=quality,face_crop=np.zeros((40,40,3),np.uint8))
    selector = CaptureSelector()
    assert selector.evaluate_observation(obs) == []
    assert selector.finalize_track_decisions(1) == []


def test_native_uhd_codec_is_preserved():
    stream = SimpleNamespace(codec_context=SimpleNamespace(name='vp9',pix_fmt='yuv420p'))
    assert 'vp09.00.51.08' in source_mime(stream)


def test_analytics_samples_native_frames_without_blocking_presentation():
    delivered = []
    finished = threading.Event()
    context = SimpleNamespace(flush_buffers=lambda: None)
    class Packet:
        def __init__(self, n):
            self.n = n
            self.is_keyframe = n == 0
            self.stream = SimpleNamespace(codec_context=context)
        def decode(self):
            if self.n == 11: finished.set()
            return [SimpleNamespace(pts=self.n,time_base=Fraction(1,60))]
    decoder = AnalyticsDecoder(lambda frame, offset: delivered.append(frame.pts), max_fps=15)
    try:
        for n in range(12): decoder.submit(Packet(n),100.,time.monotonic())
        assert finished.wait(2)
        assert delivered == [0,4,8]
    finally:
        decoder.close()


def test_shared_detector_bypasses_duplicate_yolo_and_scores_tight_face(tmp_path, monkeypatch):
    from services.people_orchestrator import PeopleSystemOrchestrator
    from services.face_contracts import FramePacket, PersonTrack, FacePose
    orch = PeopleSystemOrchestrator(data_dir=tmp_path, auto_start=False)
    monkeypatch.setattr(orch, 'is_running', lambda: True)
    monkeypatch.setattr(orch.person_detector, 'detect', lambda _: (_ for _ in ()).throw(AssertionError('duplicate YOLO pass')))
    track = PersonTrack(1,'test',1.,1.,(50.,50.,350.,470.),.95,confirmed=True)
    monkeypatch.setattr(orch.person_tracker, 'update', lambda **kw: [track])
    monkeypatch.setattr(orch.pipeline_scheduler, 'select_tracks_for_face_analysis', lambda **kw: [track])
    obs = FaceObservation('face',1,1.,(100.,100.,200.,200.),.95,pose=FacePose(0.,0.,0.),face_crop=np.zeros((150,150,3),np.uint8))
    obs.frame_sequence=1
    monkeypatch.setattr(orch.face_detector, 'detect', lambda **kw: [obs])
    monkeypatch.setattr(orch.face_coverage, '_detail', lambda *args,**kw: [])
    # Isolate native crop/duplicate-inference behavior from semantic verification.
    monkeypatch.setattr(orch.face_verifier,'validate',lambda image,values: values)
    measured = []
    real_assess = orch.face_quality_evaluator.assess_observation
    def assess(value):
        measured.append(value.face_crop.shape)
        return real_assess(value)
    monkeypatch.setattr(orch.face_quality_evaluator, 'assess_observation', assess)
    packet = FramePacket(1,1.,np.full((480,640,3),128,np.uint8),'camera','test',640,480)
    try:
        orch.process_frame(packet, shared_detections=[], shared_detection_ms=12.)
        assert measured == [(100,100,3)]
        assert obs.face_crop.shape == (150,150,3)
        assert orch.metrics.detection_ms == 12.
    finally:
        orch.face_enhancer.stop()
        orch.pipeline_scheduler.shutdown()
