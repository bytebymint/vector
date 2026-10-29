"""Average physical speed between two reviewed lane gates, using source time."""
from dataclasses import dataclass, field
import math
import cv2
import numpy as np
from .road_geometry import signed_distance, segments_intersect, inside


def lane_projection(lane):
    """Longitudinal metric for a straight planar lane with parallel road edges.

    Transverse scale is deliberately arbitrary: only the longitudinal coordinate
    is used. Accuracy depends on reviewed corners and the entered road length.
    """
    points = np.asarray(lane['points'], np.float32)
    length = lane.get('distance_metres')
    if points.shape != (4, 2) or not length or not cv2.isContourConvex(points):
        return None
    target = np.asarray([[0, 0], [1, 0], [1, length], [0, length]], np.float32)
    matrix = cv2.getPerspectiveTransform(points, target)
    return matrix if np.isfinite(matrix).all() and abs(np.linalg.det(matrix)) > 1e-9 else None


def live_lane_speed(history, lane, matrix):
    """Robust rolling longitudinal km/h, without requiring boundary crossings."""
    if matrix is None or len(history) < 5:
        return None
    samples = []
    # Use only a contiguous run inside this lane, never bridge an occlusion.
    for sample in reversed(history):
        if history[-1][0] - sample[0] > 1.5 or not inside(sample[1:], lane['points']):
            break
        if samples and samples[-1][0] - sample[0] > .6:
            break
        samples.append(sample)
    samples.reverse()
    if len(samples) < 5 or samples[-1][0] - samples[0][0] < .6:
        return None
    times = np.asarray([s[0] - samples[-1][0] for s in samples], float)
    if np.any(np.diff(times) <= 0):
        return None
    xy = np.asarray([s[1:] for s in samples], float)
    homogeneous = np.column_stack([xy, np.ones(len(xy))]) @ matrix.T
    if np.min(np.abs(homogeneous[:, 2])) < 1e-8:
        return None
    positions = homogeneous[:, 1] / homogeneous[:, 2]
    times -= times.mean()
    centred = positions - positions.mean()
    velocity = float(np.dot(times, centred) / np.dot(times, times))
    residual = float(np.sqrt(np.mean((centred - velocity * times) ** 2)))
    kmh = abs(velocity) * 3.6
    if not math.isfinite(kmh) or kmh > 200 or residual > 1.2:
        return None
    return kmh


@dataclass
class LaneTimer:
    gates: dict = field(default_factory=dict)
    entry: tuple | None = None
    complete: bool = False

    def update(self, point, stamp, lane, width, height):
        if self.complete: return None
        points = lane['points']
        crossings = []
        for name, line in (('start', points[:2]), ('end', [points[3], points[2]])):
            distance = signed_distance(point, line, width, height)
            side = 1 if distance > 3 else -1 if distance < -3 else 0
            if not side: continue
            previous = self.gates.get(name)
            if previous and stamp <= previous[2]:
                self.entry = None
                self.gates.clear()
                return None
            if previous and stamp-previous[2]>.75: self.entry=None
            if previous and side != previous[0] and segments_intersect(previous[1], point, *line):
                gap = stamp - previous[2]
                if gap <= .75:
                    # Interpolate between observations bracketing the physical gate.
                    old_distance = signed_distance(previous[1], line, width, height)
                    fraction = abs(old_distance) / (abs(old_distance) + abs(distance))
                    crossings.append((previous[2] + fraction * gap, name))
                else:
                    self.entry = None
            self.gates[name] = (side, tuple(point), stamp)
        for crossing_time, gate in sorted(crossings):
            if self.entry is None:
                self.entry = (gate, crossing_time)
            elif gate != self.entry[0]:
                elapsed = crossing_time - self.entry[1]
                distance = lane['distance_metres']
                kmh = distance / elapsed * 3.6 if elapsed > 0 else math.inf
                if .4 <= elapsed <= 300 and math.isfinite(kmh) and 0 <= kmh <= 200:
                    self.complete = True
                    return dict(kmh=kmh, elapsed_seconds=elapsed, distance_metres=distance,
                                direction='start_to_end' if self.entry[0] == 'start' else 'end_to_start',
                                start_time=self.entry[1], end_time=crossing_time)
                self.entry = None
            else:
                # A return across the entry gate cancels the incomplete traversal.
                self.entry = None
        if self.entry and (stamp - self.entry[1] > 300 or not inside(point, points)):
            self.entry = None
        return None
