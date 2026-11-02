from pathlib import Path

import numpy as np
from ultralytics.engine.results import Boxes

from services.detector import DEFAULTS, validate_settings
from services.tracker import VehicleTracker


def test_people_use_fasttracker_and_recover_after_brief_occlusion(tmp_path):
    settings = validate_settings({**DEFAULTS, 'tracker': 'bytetrack', 'lifetime': 1.5})
    tracker = VehicleTracker(settings, tmp_path, source_fps=10, person_mode=True)
    image = np.zeros((120, 160, 3), dtype=np.uint8)

    def update(sequence, box):
        data = (np.array([[*box, .9, 0]], dtype=np.float32) if box
                else np.empty((0, 6), dtype=np.float32))
        return tracker.update(Boxes(data, image.shape[:2]), image, sequence, source_fps=10)

    assert tracker.name == 'FastTracker'
    assert update(1, [40, 10, 65, 90])[0, 4] == 1
    assert update(2, [42, 10, 67, 90])[0, 4] == 1
    assert update(3, [44, 10, 69, 90])[0, 4] == 1
    assert len(update(4, None)) == 0
    recovered = update(5, [48, 10, 73, 90])
    assert recovered[0, 4] == 1

