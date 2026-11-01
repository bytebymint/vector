from pathlib import Path
import cv2
import pytest
from services.detector import Detector,DEFAULTS,validate_settings,CLASSES

@pytest.mark.parametrize('name',['yolo26n','yolo26s'])
def test_official_model_on_recorded_native_camera_frame(name):
    root=Path(__file__).resolve().parents[1]
    if not (root/'models'/f'{name}.pt').exists():pytest.skip('Install official weights before model smoke tests')
    image=cv2.imread(str(root/'tests'/'fixtures'/'rc293-native.png'))
    detector=Detector(root/'models');settings=validate_settings({**DEFAULTS,'model':name,'adaptive':False})
    detector.load(settings)
    detections=detector.predict(image,settings,{'items':[]})
    assert len(detections)>0
    assert all(int(cls) in CLASSES for cls in detections.cls)
    assert detections.xyxy.shape[1]==4
    assert detector.model.names[3]=='motorcycle'
    assert detector.model.names[2]=='car'

def test_optional_backend_falls_back_safely(tmp_path):
    root=Path(__file__).resolve().parents[1]
    detector=Detector(root/'models')
    settings=validate_settings({**DEFAULTS,'backend':'onnx'})
    detector.load(settings)
    assert detector.backend=='pytorch' and 'unavailable' in detector.warning

def test_candidate_classes_and_settings():
    from services.detector import CANDIDATE_CLASSES
    assert CANDIDATE_CLASSES == [0, 1, 2, 3, 5, 7]
    assert DEFAULTS['thresholds']['motorcycle'] == 0.15
    assert DEFAULTS['thresholds']['bicycle'] == 0.15
    s = validate_settings({'thresholds': {'motorcycle': 0.05}})
    assert s['thresholds']['motorcycle'] == 0.05
    with pytest.raises(ValueError):
        validate_settings({'thresholds': {'motorcycle': 0.04}})

def test_people_detail_defaults_keep_crowd_recall_reasonable():
    settings = validate_settings({})
    assert settings['resolution'] == 960
    assert settings['person_confidence'] == .08
    assert settings['person_confirm_hits'] == 2

def test_raising_resolution_applies_without_reloading_model():
    import numpy as np
    from types import SimpleNamespace
    from ultralytics.engine.results import Boxes
    root = Path(__file__).resolve().parents[1]
    detector = Detector(root/'models')
    requested = []

    class Model:
        def predict(self, image, **kwargs):
            requested.append(kwargs['imgsz'])
            return [SimpleNamespace(boxes=Boxes(np.empty((0, 6), np.float32), image.shape[:2]))]

    detector.model = Model()
    detector.key = ('yolo26n', 'pytorch')
    detector.configured_resolution = 640
    detector.resolution = 640
    settings = validate_settings({**DEFAULTS, 'resolution': 960, 'adaptive': True})
    detector.load(settings)
    detector.predict(np.zeros((100, 100, 3), np.uint8), settings, {'items': []})
    assert requested == [960]

def test_person_merges_with_actual_motorcycle_but_is_not_classified_alone():
    import numpy as np
    from ultralytics.engine.results import Boxes
    root = Path(__file__).resolve().parents[1]
    detector = Detector(root/'models')
    settings = validate_settings({**DEFAULTS, 'adaptive': False})
    
    class MockModel:
        names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        def predict(self, image, **kwargs):
            raw_data = np.array([
                [100, 100, 140, 150, 0.85, 0],
                [110, 120, 160, 180, 0.60, 3],
                [300, 200, 330, 250, 0.70, 0],
                [50, 50, 70, 120, 0.90, 0]
            ], dtype=np.float32)
            mock_res = type('Res', (), {'boxes': Boxes(raw_data, (432, 768))})()
            return [mock_res]
    detector.model = MockModel()
    
    geometry = {'items': [{'id': 'road1', 'kind': 'road', 'points': [[0.1, 0.2], [0.9, 0.2], [0.9, 0.9], [0.1, 0.9]]}]}
    image = np.zeros((432, 768, 3), dtype=np.uint8)
    detections = detector.predict(image, settings, geometry)
    
    assert len(detections) == 1
    assert all(int(c) == 3 for c in detections.cls)
    
    merged = [b for b in detections.xyxy if b[0] == 100]
    assert len(merged) == 1
    np.testing.assert_allclose(merged[0], [100, 100, 160, 180])
    
    lone = [b for b in detections.xyxy if b[0] == 300]
    assert len(lone) == 0
    assert abs(float(detections.conf[0])-.60)<.001

def test_road_geometry_curb_riding():
    import numpy as np
    from ultralytics.engine.results import Boxes
    root = Path(__file__).resolve().parents[1]
    detector = Detector(root/'models')
    settings = validate_settings({**DEFAULTS, 'adaptive': False})
    
    class MockModel:
        def predict(self, image, **kwargs):
            raw_data = np.array([
                [200, 172.8, 240, 220.32, 0.8, 3]
            ], dtype=np.float32)
            mock_res = type('Res', (), {'boxes': Boxes(raw_data, (432, 768))})()
            return [mock_res]
    detector.model = MockModel()
    geometry = {'items': [{'id': 'road1', 'kind': 'road', 'points': [[0.1, 0.1], [0.9, 0.1], [0.9, 0.5], [0.1, 0.5]]}]}
    image = np.zeros((432, 768, 3), dtype=np.uint8)
    detections = detector.predict(image, settings, geometry)
    assert len(detections) == 1

def test_duplicate_suppression_threshold():
    import numpy as np
    from ultralytics.engine.results import Boxes
    root = Path(__file__).resolve().parents[1]
    detector = Detector(root/'models')
    settings = validate_settings({**DEFAULTS, 'adaptive': False})
    
    class MockModel:
        def predict(self, image, **kwargs):
            raw_data = np.array([
                [100, 100, 200, 200, 0.90, 3],
                [100, 100, 200, 185, 0.85, 3]
            ], dtype=np.float32)
            mock_res = type('Res', (), {'boxes': Boxes(raw_data, (432, 768))})()
            return [mock_res]
    detector.model = MockModel()
    image = np.zeros((432, 768, 3), dtype=np.uint8)
    detections = detector.predict(image, settings, {'items': []})
    assert len(detections) == 2

def test_adapt_resolution_min_bound():
    root = Path(__file__).resolve().parents[1]
    detector = Detector(root/'models')
    settings = validate_settings({**DEFAULTS, 'adaptive': True, 'target_ms': 100, 'resolution': 640})
    detector.resolution = 640
    for _ in range(100):
        detector.adapt(elapsed_ms=500, settings=settings)
    assert detector.resolution >= 512


def test_thread_budget_is_reapplied_after_ultralytics_initialization(monkeypatch, tmp_path):
    import torch
    import services.detector as module
    calls = []
    monkeypatch.setattr(torch, 'cuda', type('Cuda', (), {'is_available': staticmethod(lambda: False)})())
    monkeypatch.setattr(torch, 'set_num_threads', calls.append)

    class Model:
        def __init__(self, *args, **kwargs): pass
        def predict(self, *args, **kwargs):
            calls.append('Ultralytics setup resets the thread pool')

    monkeypatch.setattr(module, 'YOLO', Model)
    detector = Detector(tmp_path, cpu_threads=2)
    detector.load(validate_settings({}))
    assert calls == [2, 'Ultralytics setup resets the thread pool', 2]


def test_reducing_adaptive_resolution_applies_to_the_next_frame():
    import numpy as np
    from types import SimpleNamespace
    from ultralytics.engine.results import Boxes
    root = Path(__file__).resolve().parents[1]
    detector = Detector(root/'models')
    requested = []

    class Model:
        def predict(self, image, **kwargs):
            requested.append(kwargs['imgsz'])
            return [SimpleNamespace(boxes=Boxes(np.empty((0, 6), np.float32), image.shape[:2]))]

    detector.model = Model()
    detector.resolution = 960
    detector.predict(np.zeros((100, 100, 3), np.uint8), validate_settings({'resolution': 640}), {'items': []})
    assert requested == [640]


def test_adaptive_resolution_uses_recent_timing_instead_of_one_fast_frame():
    root = Path(__file__).resolve().parents[1]
    detector = Detector(root/'models')
    settings = validate_settings({'resolution': 640, 'target_ms': 100})
    for _ in range(19):
        detector.adapt(300, settings)
    detector.adapt(10, settings)
    assert detector.resolution == 576
