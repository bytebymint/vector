"""Perspective lane association against the saved RC-339 geometry."""
import json
from pathlib import Path
import numpy as np
from services.road_geometry import inside, validate_geometry
from services.storage import Storage
from services.traffic_engine import TrafficEngine
from services.detector import DEFAULTS, validate_settings


def test_four_approaching_vehicles_remain_in_distinct_lanes(tmp_path):
    geometry = validate_geometry(json.loads((Path(__file__).parents[1]/'rc339-geometry.json').read_text(encoding='utf-8')))
    lanes = [item for item in geometry['items'] if item['kind']=='lane']
    assert len(lanes)==4
    # Manually sampled centres between the visible painted divider positions.
    samples = [(.30,[.495,.5875,.6975,.80]),(.40,[.355,.4765,.614,.7625]),(.55,[.1685,.3005,.4855,.7065])]
    storage = Storage(tmp_path/'lanes.sqlite3')
    try:
        engine = TrafficEngine(storage)
        engine.reset_session('rc339-test')
        engine.set_geometry(geometry)
        settings = validate_settings({**DEFAULTS,'confirm_hits':2,'lane_hysteresis':2})
        for frame,(y,xs) in enumerate(samples):
            for i,x in enumerate(xs):
                assert [lane['id'] for lane in lanes if inside((x,y),lane['points'])]==[f'rc339-lane-{i+1}']
            rows = [[x*1000-10,y*1000-20,x*1000+10,y*1000,i+1,.9,2,0] for i,x in enumerate(xs)]
            engine.update(np.asarray(rows),100+frame*.2,1000,1000,settings)
        assert {t.ident:t.lane for t in engine.tracks.values()}=={i:f'rc339-lane-{i}' for i in range(1,5)}
        assert not inside((.97,.55),next(i for i in geometry['items'] if i['kind']=='road')['points'])
    finally:
        storage.close()
