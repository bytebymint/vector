"""Face semantics, exact-frame capture and fail-closed regression checks.

The photographed fixtures are deliberately separate from synthetic pipeline tests.
Selected positive faces are a small reviewed sample, not a full-scene census.
"""
from pathlib import Path
from types import SimpleNamespace
import json

import cv2
import numpy as np
import pytest

from services.capture_selector import CaptureSelector, CaptureSelectorConfig
from services.face_contracts import FaceObservation, FaceQualityResult, FramePacket, PersonTrack
from services.face_detector import FaceDetector, FaceDetectorConfig
from services.face_verifier import FaceVerifier, VERIFICATION_METHOD

ROOT=Path(__file__).resolve().parents[1]
FIXTURES=Path(__file__).parent/'fixtures/face_precision'


def observation(box, sequence=7, stamp=1.):
    x,y,r,b=box
    w,h=r-x,b-y
    points=[(x+.28*w,y+.35*h),(x+.72*w,y+.35*h),(x+.5*w,y+.55*h),
            (x+.34*w,y+.77*h),(x+.66*w,y+.77*h)]
    return FaceObservation('real_candidate',1,stamp,tuple(box),.99,landmarks=points,frame_sequence=sequence)


def test_independent_model_rejects_all_24_historical_background_captures():
    verifier=FaceVerifier(ROOT/'models')
    manifest=json.loads((FIXTURES/'manifest.json').read_text())
    cv2.setNumThreads(2)
    for name in manifest['negative_crops']:
        image=cv2.imread(str(FIXTURES/name))
        h,w=image.shape[:2]
        # Deliberately plausible landmarks and high first-model confidence.
        # Geometry or sharpness alone must not authorize these background crops.
        obs=observation((.2*w,.2*h,.8*w,.8*h))
        result=verifier.verify_batch(image,[obs])[0]
        assert not result.accepted, (name,result)


def test_two_real_models_preserve_selected_genuine_faces():
    cv2.setNumThreads(2)
    image=cv2.imread(str(FIXTURES/'dubai-native.jpg'))
    detector=FaceDetector(FaceDetectorConfig(
        model_path=str(ROOT/'models/face_detection_yunet_2026may.onnx'),
        runtime='onnxruntime',score_threshold=.65,max_inference_dim=1280,
        extract_face_crops=False))
    observations=detector.detect(image,timestamp=1.,frame_sequence=7,mode='full_frame')
    accepted=FaceVerifier(ROOT/'models').validate(image,observations)
    manifest=json.loads((FIXTURES/'manifest.json').read_text())
    from services.face_detector import box_iou
    for box in manifest['selected_faces']:
        assert any(box_iou(box,obs.bbox)>.4 for obs in accepted),box
    assert all(obs.verification_score>=.90 and obs.verification_method==VERIFICATION_METHOD for obs in accepted)
    assert all(obs.face_crop is None for obs in accepted)


def test_missing_verifier_weights_fail_closed(tmp_path):
    obs=observation((10,10,80,95))
    with pytest.raises(RuntimeError,match='capture is disabled'):
        FaceVerifier(tmp_path).validate(np.ones((120,120,3),np.uint8),[obs])


@pytest.mark.parametrize('failure',['unverified','wrong_model','wrong_frame','no_quality','bad_quality'])
def test_capture_gate_cannot_be_bypassed(failure):
    obs=observation((10,10,80,95))
    obs.verification_score=.99
    obs.verification_method=VERIFICATION_METHOD
    obs.quality=FaceQualityResult(95,100,.1,95,95,95,95,is_acceptable=True)
    if failure=='unverified':obs.verification_score=.89
    elif failure=='wrong_model':obs.verification_method='guessed_head'
    elif failure=='wrong_frame':obs.frame_sequence=6
    elif failure=='no_quality':obs.quality=None
    elif failure=='bad_quality':obs.quality.is_acceptable=False
    selector=CaptureSelector(CaptureSelectorConfig(require_verified_face=True,allow_best_available=True))
    assert selector.evaluate_observation(obs,frame_sequence=7,frame_image=np.ones((120,120,3),np.uint8))==[]
    assert selector.finalize_track_decisions(1)==[]


@pytest.mark.parametrize('old_async_flag',[False,True])
def test_stale_worker_coordinates_cannot_crop_a_new_image(tmp_path,monkeypatch,old_async_flag):
    from services.people_orchestrator import PeopleSystemOrchestrator
    orch=PeopleSystemOrchestrator(data_dir=tmp_path)
    monkeypatch.setattr(orch,'is_running',lambda:True)
    stale=observation((10,10,80,95),sequence=6,stamp=.9)
    stale.verification_score=.99
    stale.verification_method=VERIFICATION_METHOD
    stale_track=PersonTrack(1,'test',.9,.9,stale.bbox,.99,confirmed=True)
    orch.face_worker=SimpleNamespace(analyze=lambda *args:([stale_track],[stale],{}),close=lambda:None)
    packet=FramePacket(7,1.,np.full((120,120,3),100,np.uint8),'cam','epoch',120,120)
    try:
        result=orch.process_frame(packet,shared_detections=[],async_face=old_async_flag)
        assert result['faces']==[] and result['captures_saved']==0
        assert orch.repository.count_profiles()==0
    finally:
        orch.face_enhancer.stop()
        orch.pipeline_scheduler.shutdown()


def test_verified_capture_metadata_and_native_pixels_survive_storage(tmp_path):
    from services.profile_repository import ProfileRepository
    image=np.random.default_rng(43).integers(0,256,(140,180,3),dtype=np.uint8)
    obs=observation((40,40,120,110))
    obs.verification_score=.99
    obs.verification_method=VERIFICATION_METHOD
    obs.quality=FaceQualityResult(95,100,.1,95,95,95,95,is_acceptable=True)
    selector=CaptureSelector(CaptureSelectorConfig(require_verified_face=True))
    decisions=selector.evaluate_observation(obs,7,image)
    candidate=next(d.candidate for d in decisions if d.should_save_disk)
    repo=ProfileRepository(tmp_path/'people.db',tmp_path/'captures')
    record=repo.save_candidate('cam:verified:1',candidate)
    assert np.array_equal(cv2.imread(record.original_path),candidate.crop_original)
    assert repo.get_profile('cam:verified:1').metadata['face_validation_version']==2
