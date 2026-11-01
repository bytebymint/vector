"""Regression tests for face recall independent of body detections."""
from types import SimpleNamespace
import numpy as np

from services.face_contracts import FaceObservation, FramePacket, PersonTrack
from services.face_coverage import FaceCoverage, CoverageConfig, merge_faces
from services.face_detector import FaceDetectorConfig


def face(box=(100.,100.,140.,150.), score=.9, name='f', timestamp=1., sequence=None):
    return FaceObservation(name,-1,timestamp,box,score,frame_sequence=sequence)


def coverage():
    detector = SimpleNamespace(config=FaceDetectorConfig(backend='haar'), detect=lambda **kw: [face()])
    engine = FaceCoverage(detector, CoverageConfig(tiles_per_frame=0))
    engine.reset('test')
    return engine


def test_face_discovery_runs_with_zero_bodies():
    engine = coverage()
    packet = FramePacket(1,1.,np.zeros((480,640,3),np.uint8),'camera','test',640,480)
    observations = engine.detect(packet, [])
    assert len(observations) == 1
    tracks, observations = engine.update(observations,1.,'test')
    assert not tracks[0].confirmed
    ident = tracks[0].track_id
    tracks, observations = engine.update([face((102,100,142,150))],1.2,'test')
    assert tracks[0].confirmed and observations[0].track_id == ident


def test_brief_miss_recovers_spatial_track_and_expiration_does_not():
    engine = coverage()
    engine.update([face()],1.,'test')
    tracks,_ = engine.update([face()],1.2,'test')
    ident = tracks[0].track_id
    engine.update([],1.4,'test')
    tracks,_ = engine.update([face((104,100,144,150))],1.6,'test')
    assert tracks[0].track_id == ident
    engine.update([],5.,'test')
    tracks,_ = engine.update([face()],5.2,'test')
    assert tracks[0].track_id != ident


def test_body_overlap_cannot_discard_distinct_face_observations():
    engine = coverage()
    a,b = face((100,100,140,150),name='a'),face((148,100,188,150),name='b')
    engine.detector.detect = lambda **kw: [a,b]
    engine._detail = lambda *args: []
    packet = FramePacket(1,1.,np.zeros((480,640,3),np.uint8),'camera','test',640,480)
    bodies = [PersonTrack(i,'body',1.,1.,(80,80,250,460),.9) for i in (1,2)]
    assert len(engine.detect(packet,bodies)) == 2


def test_detail_budget_rotates_through_every_body():
    engine = coverage()
    engine.detector.detect = lambda **kw: []
    searched = []
    engine._detail = lambda packet, region: searched.append(region) or []
    bodies = [PersonTrack(i,'body',1.,1.,(i*50,80,i*50+40,460),.9) for i in range(10)]
    for n in range(3):
        packet = FramePacket(n,1.+n*.2,np.zeros((480,640,3),np.uint8),'camera','test',640,480)
        engine.detect(packet,bodies)
    assert len(set(searched)) == 10


def test_cross_scale_duplicates_and_merged_faces():
    a = face((100,100,140,150),.85,'a')
    b = face((148,100,188,150),.80,'b')
    duplicate = face((98,98,142,153),.7,'dup')
    merged = face((90,90,195,170),.95,'merged')
    assert {o.observation_id for o in merge_faces([a,b,duplicate,merged])} == {'a','b'}


def test_confirmed_small_face_saves_original_without_body_detection(tmp_path, monkeypatch):
    from services.people_orchestrator import PeopleSystemOrchestrator
    # Explicit permissive quality configuration isolates the frame/capture
    # contract; production defaults reject this limited-detail synthetic crop.
    orch = PeopleSystemOrchestrator(data_dir=tmp_path,config={'quality_threshold':40.})
    monkeypatch.setattr(orch,'is_running',lambda: True)
    def detected(packet, *args):
        return [face(timestamp=packet.timestamp,sequence=packet.sequence)]
    monkeypatch.setattr(orch.face_coverage,'detect',detected)
    # A pipeline contract test, not an AI accuracy test. Real model regressions
    # live in test_face_verifier.py with photographed faces and negative crops.
    def verified(image, observations):
        for obs in observations:
            obs.verification_score=.99
            obs.verification_method='yunet+res10ssd-v1'
        return observations
    monkeypatch.setattr(orch.face_verifier,'validate',verified)
    import cv2
    from tests.test_face_quality import make_synthetic_face
    image = np.random.default_rng(4).integers(0,256,(480,640,3),dtype=np.uint8)
    image[100:150, 100:140] = cv2.resize(make_synthetic_face(128), (40, 50))
    try:
        results = []
        for n in range(2):
            packet = FramePacket(n,1.+n*.2,image,'camera','test',640,480)
            results.append(orch.process_frame(packet, shared_detections=[]))
        assert results[0]['body_count'] == 0 and results[0]['active_count'] == 0
        assert results[0]['captures_saved'] == 0  # verify before persistence
        assert results[1]['active_count'] == 1
        assert results[1]['captures_saved'] >= 1
        assert orch.repository.count_profiles() == 1
        assert results[1]['faces'][0]['quality']['is_acceptable']
        assert results[1]['tracking_kind'] == 'face'
    finally:
        orch.face_enhancer.stop()
        orch.pipeline_scheduler.shutdown()


def test_motion_hints_recover_face_after_camera_translation():
    import cv2
    engine=coverage()
    original=np.random.default_rng(9).integers(0,256,(480,640,3),dtype=np.uint8)
    first=FramePacket(1,1.,original,'camera','test',640,480)
    obs=engine.detect(first)
    tracks,_=engine.update(obs,1.,'test')
    ident=tracks[0].track_id
    moved=cv2.warpAffine(original,np.float32([[1,0,30],[0,1,5]]),(640,480))
    engine.detector.detect=lambda **kw:[face((130,105,170,155),timestamp=1.2,sequence=2)]
    engine._detail=lambda *args,**kw:[]
    second=FramePacket(2,1.2,moved,'camera','test',640,480)
    obs=engine.detect(second)
    tracks,_=engine.update(obs,1.2,'test')
    assert tracks[0].track_id==ident and tracks[0].confirmed
    assert ident in engine._flow_boxes


def test_limited_detail_policy_is_explicit_and_cannot_claim_high_resolution():
    from services.capture_selector import CaptureSelector,CaptureSelectorConfig
    from services.face_contracts import FaceQualityResult
    observation=face()
    observation.face_crop=np.zeros((50,40,3),np.uint8)
    observation.quality=FaceQualityResult(3.,1.,.9,5.,60.,50.,5.,is_acceptable=False)
    assert CaptureSelector().evaluate_observation(observation)==[]
    selector=CaptureSelector(CaptureSelectorConfig(allow_best_available=True,high_res_min_pixels=160))
    decisions=selector.evaluate_observation(observation)
    assert decisions == []  # even the legacy option cannot bypass quality
    assert selector.finalize_track_decisions(observation.track_id) == []


def test_yunet_runtime_stride_decoding_and_landmark_coordinates():
    from services.yunet_runtime import YuNetRuntime
    outputs={}
    for stride in (8,16,32):
        anchors=(320//stride)**2
        for prefix,size in [('cls',1),('obj',1),('bbox',4),('kps',10)]:
            outputs[f'{prefix}_{stride}']=np.zeros((1,anchors,size),np.float32)
    index=137 # grid column 17, row 3 at stride 8
    outputs['cls_8'][0,index,0]=.81
    outputs['obj_8'][0,index,0]=1.
    outputs['bbox_8'][0,index]=[.5,.5,np.log(2),np.log(3)]
    runtime=object.__new__(YuNetRuntime)
    runtime.names=list(outputs)
    runtime.session=SimpleNamespace(run=lambda names,inputs:[outputs[n] for n in names])
    runtime.input_name='input';runtime.threshold=.5;runtime.nms=.3;runtime.top_k=100
    _,rows=runtime.detect(np.zeros((320,320,3),np.uint8))
    assert rows.shape==(1,15)
    assert np.allclose(rows[0,:4],[132,16,16,24],atol=1e-4)
    assert np.allclose(rows[0,4:6],[136,24])
    assert abs(rows[0,14]-.9)<1e-5


def test_missing_detection_does_not_project_body_hint_into_face():
    engine=coverage()
    image=np.zeros((480,640,3),np.uint8)
    body=PersonTrack(1,'body',1.,1.,(80,80,200,440),.9)
    packet=FramePacket(1,1.,image,'camera','test',640,480)
    observations=engine.detect(packet,[body])
    tracks,_=engine.update(observations,1.,'test')
    ident=tracks[0].track_id
    engine.detector.detect=lambda **kw: []
    engine._detail=lambda *args,**kw: []
    for n in range(2,7):
        stamp=1.+n*.033
        packet=FramePacket(n,stamp,image,'camera','test',640,480)
        observations=engine.detect(packet,[body])
        assert observations == []
        tracks,matched=engine.update(observations,stamp,'test')
        assert tracks == [] and matched == []
    # Hints do not erase the actual short-term lost-track recovery state.
    tracks,_=engine.update([face((102,100,142,150),timestamp=1.4)],1.4,'test')
    assert tracks[0].track_id==ident


def test_stale_detector_output_is_not_an_observation_of_new_frame():
    engine=coverage()
    packet=FramePacket(50,2.,np.zeros((480,640,3),np.uint8),'camera','test',640,480)
    assert engine.detect(packet,[]) == []  # mocked detector returns timestamp 1


def test_tile_threshold_enforced_at_least_0_75():
    """Verify tile scanning enforces threshold >= 0.75."""
    engine = coverage()
    captured_thresholds = []
    original_detect_raw = engine.tile_detector._detect_raw
    engine.tile_detector._detect_raw = lambda image, score_threshold=None: captured_thresholds.append(score_threshold) or []

    packet = FramePacket(1, 1.0, np.zeros((480, 640, 3), np.uint8), 'camera', 'test', 640, 480)
    engine._detail(packet, (100, 100, 300, 300), tile=True)

    assert len(captured_thresholds) == 1
    assert captured_thresholds[0] >= 0.75


def test_tile_scanning_skips_sky_and_asphalt():
    """Verify tile coordinates are bounded away from top sky (< 0.08*h) and bottom asphalt (> 0.78*h)."""
    engine = coverage()
    w, h = 3840, 2160  # 4K frame dimensions
    tiles = [
        (x * w, y * h, min(w, (x + 0.40) * w), min(h, (y + 0.45) * h))
        for y in (0.08, 0.32)
        for x in (0.0, 0.30, 0.60)
    ]
    for x1, y1, x2, y2 in tiles:
        # Must not start in top sky (< 0.08 * h)
        assert y1 >= 0.05 * h, f"Tile top y1={y1} overlaps empty top sky"
        # Must not reach bottom asphalt (> 0.80 * h)
        assert y2 <= 0.80 * h, f"Tile bottom y2={y2} reaches bottom asphalt"


def test_face_coverage_filters_invalid_faces_through_is_valid_face():
    """Verify candidate observations in _detail are filtered through is_valid_face before creating FaceObservations."""
    engine = coverage()
    packet = FramePacket(1, 1.0, np.zeros((480, 640, 3), np.uint8), 'camera', 'test', 640, 480)

    # Mock raw detection with neutral gray (no skin tone)
    mock_raw = [{
        "bbox": (50.0, 50.0, 90.0, 95.0),
        "confidence": 0.88,
        "landmarks": [(60, 60), (80, 60), (70, 70), (62, 82), (78, 82)],
    }]
    # view in _detail will be np.zeros (no skin tone), so is_valid_face will return False
    engine.detail_detector._detect_raw = lambda image, score_threshold=None: mock_raw
    obs = engine._detail(packet, (100, 100, 300, 300), tile=False)
    assert len(obs) == 0, "Candidate face with zero contrast must be filtered out"
