import json
from pathlib import Path
import numpy as np
import pytest
from services.detector import DEFAULTS,validate_settings
from services.storage import Storage
from services.traffic_engine import TrafficEngine


@pytest.fixture
def system(tmp_path):
    db=Storage(tmp_path/'events.sqlite3');engine=TrafficEngine(db);engine.reset_session('offline')
    yield engine,db,validate_settings({**DEFAULTS,'confirm_hits':2,'lane_hysteresis':2})
    db.close()

def row(ident,x,y,label=2):return [x*1000-20,y*1000-40,x*1000+20,y*1000,ident,.9,label,0]
def update(engine,settings,stamp,rows):return engine.update(np.asarray(rows,np.float32).reshape(-1,8),stamp,1000,1000,settings)
def polygon(ident,kind,left,right):return dict(id=ident,name=ident,kind=kind,points=[[left,0],[right,0],[right,1],[left,1]],direction=[0,1],color='#36d9d0')

def test_actual_lane_transition_and_hysteresis(system):
    e,db,s=system;e.set_geometry({'items':[polygon('A','lane',0,.5),polygon('B','lane',.5,1)]})
    for i,x in enumerate([.3,.3,.3,.51,.49,.51,.52,.52]):update(e,s,100+i*.1,[row(1,x,.6)])
    db.flush();events=db.events()
    transitions=[v for v in events if v['type']=='lane_transition']
    assert len(transitions)==1 and transitions[0]['from_lane']=='A' and transitions[0]['to_lane']=='B'

def test_multiple_vehicles_cross_once(system):
    e,db,s=system;e.set_geometry({'items':[dict(id='L',kind='line',name='L',points=[[0,.5],[1,.5]],role='in')]})
    for i,y in enumerate([.3,.35,.4,.6,.65,.7,.7,.7]):update(e,s,100+i*.1,[row(1,.3,y),row(2,.7,y,3)])
    db.flush();events=[v for v in db.events() if v['type']=='crossing']
    assert len(events)==2 and all(v['confirmed'] and v['role']=='in' for v in events)
    assert db.summary()['unique']==2

def test_zone_entry_exit(system):
    e,db,s=system;e.set_geometry({'items':[polygon('Z','zone',.4,.6)]})
    for i,x in enumerate([.2,.2,.3,.5,.5,.7,.8]):update(e,s,100+i*.1,[row(1,x,.6)])
    db.flush();types=[v['type'] for v in db.events()]
    assert types.count('zone_entry')==1 and types.count('zone_exit')==1

def test_track_expiry_preserves_history(system):
    e,db,s=system
    for i in range(3):update(e,s,100+i*.1,[row(1,.4,.6)])
    assert len(e.tracks)==1
    update(e,s,104,[]);db.flush()
    assert not e.tracks and db.summary()['unique']==1

def test_temporal_class_voting(system):
    e,db,s=system
    for i,label in enumerate([2,2,2,7,2]):update(e,s,100+i*.1,[row(1,.4,.6,label)])
    assert e.current[0]['category']=='car' and e.current[0]['subtype']=='Unknown subtype'

def test_frame_timestamps_drive_motion(system):
    e,db,s=system
    for i in range(8):update(e,s,100+i*.2,[row(1,.2+i*.01,.6)])
    assert 40<e.current[0]['speed_px_s']<51
    assert e.current[0]['speed_kmh'] is None


def test_media_timing_preserves_speed_despite_network_arrival_delay(system):
    e,db,s=system
    for i in range(10):
        e.update(np.asarray([row(1,.2+i*.01,.6)],np.float32),100+i*.2,1000,1000,s,motion_stamp=100+i*.1)
    assert 98<e.current[0]['speed_px_s']<102
    assert abs(e.current[0]['history'][-1][0]-100.9)<.001

def test_stream_epoch_does_not_reuse_uid(system):
    e,db,s=system
    for i in range(3):update(e,s,100+i*.1,[row(1,.4,.6)])
    first=e.current[0]['uid'];e.reset_session('reconnected')
    for i in range(3):update(e,s,110+i*.1,[row(1,.4,.6)])
    assert e.current[0]['uid']!=first
    db.flush();assert db.summary()['unique']==2

def test_prerecorded_trajectory_fixture(system):
    e,db,s=system
    fixture=json.loads((Path(__file__).parent/'fixtures'/'trajectory.json').read_text())
    e.set_geometry(fixture['geometry'])
    for frame in fixture['frames']:update(e,s,frame['timestamp'],[row(v['id'],*v['point']) for v in frame['vehicles']])
    db.flush();assert len([v for v in db.events() if v['type']=='crossing'])==1

def test_motorcycle_confirms_in_two_hits_and_threshold_fallback(system):
    e,db,s=system
    s_3hits={**s,'confirm_hits':3}
    # Car (label=2) requires 3 hits to confirm
    update(e,s_3hits,100.0,[row(1,.4,.6,label=2)])
    update(e,s_3hits,100.1,[row(1,.4,.6,label=2)])
    assert not e.tracks['offline:1'].confirmed
    update(e,s_3hits,100.2,[row(1,.4,.6,label=2)])
    assert e.tracks['offline:1'].confirmed

    # Motorcycle (label=3) confirms in min(2, 3) = 2 hits
    update(e,s_3hits,100.0,[row(2,.7,.6,label=3)])
    assert not e.tracks['offline:2'].confirmed
    update(e,s_3hits,100.1,[row(2,.7,.6,label=3)])
    assert e.tracks['offline:2'].confirmed

    # Bicycle (label=1) also confirms in 2 hits
    update(e,s_3hits,100.0,[row(3,.2,.6,label=1)])
    assert not e.tracks['offline:3'].confirmed
    update(e,s_3hits,100.1,[row(3,.2,.6,label=1)])
    assert e.tracks['offline:3'].confirmed
