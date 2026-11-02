import numpy as np
import pytest
from services.counting import LineState
from services.road_geometry import anchor,normalize_box,canvas_point,inside,validate_geometry,calibration_matrix


def test_bbox_coordinates():
    assert normalize_box([10,20,50,80],100,200)==[.1,.1,.5,.4]
    assert anchor([10,20,50,80],100,200)==(.3,.4)

def test_overlay_letterbox():
    assert canvas_point(500,500,1000,1000,1920,1080)==(.5,.5)
    assert canvas_point(0,218.75,1000,1000,1920,1080)==pytest.approx((0,0))
    assert canvas_point(1000,781.25,1000,1000,1920,1080)==pytest.approx((1,1))

def test_point_in_lane():
    assert inside((.5,.5),[[0,0],[1,0],[1,1],[0,1]])
    assert not inside((1.1,.5),[[0,0],[1,0],[1,1],[0,1]])

def test_crossing_direction_and_idempotency():
    s=LineState();line=[[.1,.5],[.9,.5]]
    assert s.update((.5,.4),0,line,1000,1000) is None
    event=s.update((.5,.6),.2,line,1000,1000)
    assert event['direction']=='negative_to_positive' and event['confirmed']
    for i in range(100):assert s.update((.5,.6),.3+i*.01,line,1000,1000) is None
    assert s.update((.5,.4),2,line,1000,1000)['direction']=='positive_to_negative'
    assert s.update((.5,.6),2.2,line,1000,1000) is None

def test_jitter_does_not_count():
    s=LineState();line=[[0,.5],[1,.5]]
    for i,y in enumerate([.49,.499,.501,.498,.502,.499,.50,.497]):
        assert s.update((.4,y),i*.1,line,1000,1000) is None

def test_beyond_line_endpoint_does_not_count():
    s=LineState();line=[[.2,.5],[.8,.5]]
    s.update((.9,.4),0,line,1000,1000)
    assert s.update((.9,.6),.2,line,1000,1000) is None

def test_gap_crossing_uncertain():
    s=LineState();line=[[0,.5],[1,.5]]
    s.update((.4,.3),0,line,1000,1000)
    assert not s.update((.4,.7),3,line,1000,1000)['confirmed']

def test_validate_rejects_bad_geometry():
    with pytest.raises(ValueError):validate_geometry({'items':[{'kind':'line','points':[[0,0],[float('nan'),1]]}]})
    with pytest.raises(ValueError):validate_geometry({'items':[{'kind':'lane','points':[[0,0],[1,1]]}]})

def test_calibration_requires_independent_check():
    config={'image':[[0,0],[1,0],[1,1],[0,1]],'world':[[0,0],[10,0],[10,20],[0,20]],'measured':True}
    with pytest.raises(ValueError):calibration_matrix(config)
    config.update(check_image=[[.1,.5],[.9,.5]],check_metres=8)
    assert np.isfinite(calibration_matrix(config)).all()
    config['check_metres']=15
    with pytest.raises(ValueError):calibration_matrix(config)
