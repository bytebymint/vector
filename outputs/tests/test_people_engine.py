import numpy as np
import pytest
from services.people_engine import PeopleCounter
from services.detector import DEFAULTS, validate_settings


def test_counts_confirmed_people_once_even_when_repeated_over_many_frames():
    counter = PeopleCounter()
    counter.reset('camera-session')
    rows = np.asarray([[i, 10, i+20, 90, i+1, .8, 0, 0] for i in range(250)], dtype=float)
    for n in range(3):
        people = counter.update(rows, n*.1, 1920, 1080, confirm_hits=3)
    assert len(people) == 250
    assert sum(p['confirmed'] for p in people) == 250
    assert counter.state()['visible'] == 250
    assert counter.state()['unique'] == 250
    for n in range(3, 20):
        counter.update(rows, n*.1, 1920, 1080, confirm_hits=3)
    assert counter.state()['unique'] == 250


def test_two_person_count_and_stale_tracks_expire_without_recounting_id():
    counter = PeopleCounter()
    counter.reset('source-1')
    rows = [[0, 0, 20, 60, 10, .9, 0], [30, 0, 50, 60, 11, .7, 0]]
    for n in range(3):
        counter.update(rows, n*.2, 100, 100, confirm_hits=3)
    assert counter.state()['unique'] == 2
    counter.update([], 5, 100, 100, lifetime=2.5)
    assert counter.state()['visible'] == 0
    counter.update(rows, 5.2, 100, 100, confirm_hits=3)
    assert counter.state()['unique'] == 2


def test_strong_person_detection_is_visible_immediately_but_weak_candidate_waits():
    counter = PeopleCounter()
    counter.reset('scene')
    counter.update([[0, 0, 20, 60, 1, .8, 0], [30, 0, 50, 60, 2, .25, 0]], 0, 100, 100)
    assert counter.state()['visible'] == 1
    assert counter.state()['unique'] == 1
    counter.update([[30, 0, 50, 60, 2, .25, 0]], .1, 100, 100)
    assert counter.state()['visible'] == 1
    assert counter.state()['unique'] == 2


def test_person_threshold_validation():
    assert validate_settings({**DEFAULTS, 'person_confidence': .08})['person_confidence'] == .08
    with pytest.raises(ValueError):
        validate_settings({**DEFAULTS, 'person_confidence': .01})


def test_person_boxes_are_separate_from_vehicle_detections():
    from pathlib import Path
    import cv2
    from ultralytics.engine.results import Boxes
    from services.detector import Detector

    class Model:
        names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        def predict(self, image, **kwargs):
            data = np.array([[10,10,30,60,.09,0],[40,10,70,70,.8,0],[80,20,120,70,.9,2]],np.float32)
            return [type('Result',(),{'boxes':Boxes(data,image.shape[:2])})()]

    detector = Detector(Path('.'))
    detector.model = Model()
    image = np.zeros((100, 150, 3), np.uint8)
    vehicles = detector.predict(image, validate_settings({**DEFAULTS, 'person_confidence': .10}), {'items': []})
    assert list(vehicles.cls) == [2]
    assert len(detector.people_boxes) == 1
    assert list(detector.people_boxes.cls) == [0]

