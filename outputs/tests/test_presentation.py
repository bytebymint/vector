import json

from services.presentation import compact_frame, MAX_TRAJECTORY_SAMPLES


def test_live_payload_keeps_motion_but_bounds_inspection_history():
    track = dict(uid='session:42', id=42, bbox=[.1, .2, .3, .4],
                 speed_kmh=35.5, speed_calibrated=True, velocity=[10, 5],
                 history=[(i / 30, .2, .3) for i in range(90)],
                 events=[{'type': 'crossing'}] * 50,
                 detection_history=[{'confidence': .9}] * 40)
    source = dict(sequence=7, timestamp=12.0, tracks=[track], raw=[])
    live = compact_frame(source)
    vehicle = live['tracks'][0]
    assert vehicle['uid'] == track['uid']
    assert vehicle['speed_kmh'] == 35.5
    assert vehicle['bbox'] == track['bbox']
    assert vehicle['velocity'] == [10, 5]
    assert vehicle['history'] == track['history'][-MAX_TRAJECTORY_SAMPLES:]
    assert vehicle['history'][-1] == track['history'][-1]
    assert 'events' not in vehicle and 'detection_history' not in vehicle
    assert len(json.dumps(live)) < len(json.dumps(source)) / 2
    # Compacting does not truncate inspection history or modify its container.
    live['tracks'].clear()
    assert len(source['tracks']) == 1 and len(track['history']) == 90
    assert len(track['events']) == 50
