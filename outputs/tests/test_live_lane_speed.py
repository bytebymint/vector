import cv2
import numpy as np
import pytest
from services.lane_speed import lane_projection, live_lane_speed


@pytest.mark.parametrize('reverse', [False, True])
def test_perspective_speed_before_boundary_gate(reverse):
    lane = dict(points=[[.45,.1],[.55,.1],[1,1],[0,1]], distance_metres=100)
    matrix = lane_projection(lane)
    times = np.linspace(0, 1.2, 13)
    positions = 60 - times*10 if reverse else 20 + times*10
    world = np.asarray([[[.5,float(y)] for y in positions]], np.float32)
    image = cv2.perspectiveTransform(world, np.linalg.inv(matrix))[0]
    history = [(float(t),float(p[0]),float(p[1])) for t,p in zip(times,image)]
    assert live_lane_speed(history,lane,matrix) == pytest.approx(36,abs=.001)
    # An interrupted or very short track cannot supply a speed.
    assert live_lane_speed(history[:3],lane,matrix) is None
    assert live_lane_speed(history[:6]+[(4,*history[-1][1:])],lane,matrix) is None


def test_stationary_and_outside_lane():
    lane = dict(points=[[.3,.2],[.5,.2],[.5,.8],[.3,.8]],distance_metres=60)
    matrix = lane_projection(lane)
    assert live_lane_speed([(i*.1,.4,.5) for i in range(14)],lane,matrix) == pytest.approx(0, abs=1e-8)
    assert live_lane_speed([(i*.1,.7,.5) for i in range(14)],lane,matrix) is None
