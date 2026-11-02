import numpy as np
import pytest
from services.lane_learning import LaneLearner, suggest_lanes
from services.lane_speed import LaneTimer
from services.road_geometry import validate_geometry
from services.traffic_engine import TrafficEngine
from services.detector import DEFAULTS

LANE=dict(id='l1',kind='lane',name='Lane 1',points=[[.3,.2],[.5,.2],[.5,.8],[.3,.8]],
          distance_metres=60,color='#56ded2',direction=[0,1])


@pytest.mark.parametrize('reverse',[False,True])
def test_measured_lane_distance_over_interpolated_gate_time(reverse):
    timer=LaneTimer();readings=[]
    for i,y in enumerate(np.linspace(.9,.1,41) if reverse else np.linspace(.1,.9,41)):
        result=timer.update((.4,float(y)),i*.2,LANE,1000,1000)
        if result:readings.append(result)
        if i<34:assert not readings
    assert len(readings)==1
    assert readings[0]['elapsed_seconds']==pytest.approx(6)
    assert readings[0]['kmh']==pytest.approx(36)
    assert readings[0]['direction']==('end_to_start' if reverse else 'start_to_end')
    assert timer.update((.4,.1),9,LANE,1000,1000) is None


def test_seen_midway_is_not_measured_and_jitter_cannot_complete():
    timer=LaneTimer()
    for i,y in enumerate(np.linspace(.5,.9,25)):
        assert timer.update((.4,float(y)),i*.2,LANE,1000,1000) is None
    timer=LaneTimer()
    for i,y in enumerate([.1,.25,.5,.79,.799,.801,.798,.802]):
        assert timer.update((.4,y),i*.2,LANE,1000,1000) is None


def test_tracking_gap_or_lane_departure_cancels_timing():
    timer=LaneTimer()
    for point,t in [((.4,.1),0),((.4,.3),.4),((.4,.5),2),((.4,.9),2.4)]:
        assert timer.update(point,t,LANE,1000,1000) is None
    timer=LaneTimer()
    for point,t in [((.4,.1),0),((.4,.3),.4),((.8,.5),.8),((.4,.7),1.2),((.4,.9),1.6)]:
        assert timer.update(point,t,LANE,1000,1000) is None


def test_distance_persists_and_invalid_lane_shape_rejected():
    assert validate_geometry({'items':[LANE]})['items'][0]['distance_metres']==60
    for distance in [-5,float('nan'),True,6000]:
        with pytest.raises(ValueError):validate_geometry({'items':[{**LANE,'distance_metres':distance}]})
    with pytest.raises(ValueError):
        validate_geometry({'items':[{**LANE,'points':[[.3,.2],[.5,.8],[.5,.2],[.3,.8]]}]})


def test_learns_multiple_supported_lanes_without_inventing_empty_lanes():
    learner=LaneLearner()
    tracks=[]
    for lane,x in enumerate([.3,.6]):
        for j in range(3):
            tracks.append(dict(uid=f'{lane}:{j}',confirmed=True,
                history=[(i*.1,x+j*.002,float(y)) for i,y in enumerate(np.linspace(.2,.8,40))]))
    learner.observe('camera-a',tracks)
    result=suggest_lanes(learner.snapshot(),1920,1080)
    assert len(result['items'])==2
    assert all(i['suggestion']['supporting_tracks']==3 for i in result['items'])
    assert all(i['distance_metres'] is None for i in result['items'])
    assert all(len(i['points'])==4 for i in result['items'])
    learner.observe('camera-b',[])
    assert not suggest_lanes(learner.snapshot(),1920,1080)['items']
    learner.observe('camera-b',tracks[:1]);learner.observe('camera-b',tracks[:1])
    assert not suggest_lanes(learner.snapshot(),1920,1080)['items']


def test_lane_speed_uses_source_clock_and_persists_once():
    class Store:
        def __init__(self):self.events=[]
        def enqueue(self,kind,value):
            if kind=='event':self.events.append(value)
    store=Store();engine=TrafficEngine(store)
    engine.set_geometry(validate_geometry({'items':[LANE]}));engine.reset_session('fixture')
    for i,y in enumerate(np.linspace(.1,.9,41)):
        rows=[[350,y*1000-40,450,y*1000,1,.95,2]]
        tracks=engine.update(rows,1000+i*.2,1000,1000,DEFAULTS,motion_stamp=i*.2)
    track=tracks[0]
    assert track['speed_kmh']==pytest.approx(36)
    assert track['speed_method']=='lane_average'
    assert track['lane_speed']['elapsed_seconds']==pytest.approx(6)
    assert len([e for e in store.events if e['type']=='lane_speed'])==1
    engine.set_geometry(validate_geometry({'items':[LANE]}))
    assert engine.tracks['fixture:1'].lane_speed is None


def test_simultaneous_vehicles_keep_separate_travel_times():
    class Store:
        def __init__(self):self.events=[]
        def enqueue(self,kind,value):
            if kind=='event':self.events.append(value)
    store=Store();engine=TrafficEngine(store)
    engine.set_geometry(validate_geometry({'items':[LANE]}));engine.reset_session('multi')
    for i in range(41):
        rows=[]
        for ident,increment,x in [(1,.02,360),(2,.04,430)]:
            y=.1+increment*i
            if y<=.95:rows.append([x-20,y*1000-40,x+20,y*1000,ident,.95,2])
        engine.update(rows,1000+i*.2,1000,1000,DEFAULTS,motion_stamp=i*.2)
    results=[e for e in store.events if e['type']=='lane_speed']
    assert len(results)==2
    assert sorted(e['speed_kmh'] for e in results)==pytest.approx([36,72])
    assert len({e['uid'] for e in results})==2
