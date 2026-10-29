"""Normalized road geometry; image coordinates have y increasing downward."""
from __future__ import annotations
import math
import uuid
import cv2
import numpy as np


def anchor(box: list[float], width: int, height: int) -> tuple[float, float]:
    return ((box[0] + box[2]) / (2 * width), box[3] / height)


def normalize_box(box, width, height):
    return [float(v / (width if i % 2 == 0 else height)) for i, v in enumerate(box)]


def canvas_point(x, y, canvas_width, canvas_height, video_width, video_height):
    """Map a CSS canvas coordinate through object-fit:contain, including letterboxing."""
    scale = min(canvas_width / video_width, canvas_height / video_height)
    left, top = (canvas_width-video_width*scale)/2, (canvas_height-video_height*scale)/2
    return ((x-left)/(video_width*scale), (y-top)/(video_height*scale))


def inside(point, polygon):
    return len(polygon) >= 3 and cv2.pointPolygonTest(np.asarray(polygon, np.float32), point, False) >= 0


def signed_distance(point, line, width, height):
    p, a, b = [np.asarray(v, float) * [width, height] for v in (point, *line)]
    delta = b-a
    offset=p-a
    return float((delta[0]*offset[1]-delta[1]*offset[0]) / max(np.linalg.norm(delta), 1e-9))


def segments_intersect(p, q, a, b):
    p, q, a, b = map(lambda v: np.asarray(v, float), (p, q, a, b))
    r, s = q-p, b-a
    cross=lambda u,v:float(u[0]*v[1]-u[1]*v[0])
    denominator = cross(r, s)
    if abs(denominator) < 1e-10:
        return False
    t, u = cross(a-p, s)/denominator, cross(a-p, r)/denominator
    return 0 <= t <= 1 and 0 <= u <= 1


def validate_geometry(value):
    if not isinstance(value, dict):
        raise ValueError('Geometry must be an object')
    result = {'items': [], 'calibration': value.get('calibration')}
    if len(value.get('items', [])) > 100:
        raise ValueError('At most 100 geometry items')
    ids = set()
    for item in value.get('items', []):
        kind = item.get('kind')
        if kind not in ('lane', 'road', 'restricted', 'entry', 'exit', 'zone', 'line', 'guide'):
            raise ValueError('Unsupported geometry kind')
        points = item.get('points', [])
        if (len(points) != 2) if kind == 'line' else (not (2 if kind == 'guide' else 3) <= len(points) <= 80):
            raise ValueError('A line needs two points; guides need 2–80 points; polygons need 3–80 vertices')
        for p in points:
            if len(p) != 2 or any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in p):
                raise ValueError('Coordinates must be finite normalized [0,1] pairs')
        ident = str(item.get('id') or uuid.uuid4().hex)[:80]
        if ident in ids:
            raise ValueError('Geometry IDs must be unique')
        ids.add(ident)
        direction = item.get('direction', [0, -1])
        if len(direction) != 2 or any(not isinstance(v, (float, int)) or not math.isfinite(v) or abs(v)>1 for v in direction):
            raise ValueError('Direction must be a normalized vector')
        result['items'].append(dict(id=ident, kind=kind, points=points, name=str(item.get('name', kind))[:60],
            color=str(item.get('color', '#26d9d0'))[:20], visible=bool(item.get('visible', True)),
            direction=direction, role=item.get('role') if item.get('role') in ('in', 'out') else None))
        distance = item.get('distance_metres')
        if distance is not None:
            if kind != 'lane' or len(points) != 4 or not cv2.isContourConvex(np.asarray(points, np.float32)):
                raise ValueError('Lane speed measurement needs a convex four-corner lane: first two corners START, last two END')
            if isinstance(distance, bool) or not isinstance(distance, (float, int)) or not math.isfinite(distance) or not .5 <= distance <= 5000:
                raise ValueError('Enter a measured lane length between 0.5 and 5000 metres')
            result['items'][-1]['distance_metres'] = float(distance)
        suggestion = item.get('suggestion')
        if isinstance(suggestion, dict):
            result['items'][-1]['suggestion'] = dict(method='observed straight vehicle paths',
                supporting_tracks=max(0, min(250, int(suggestion.get('supporting_tracks', 0)))), reviewed=bool(distance))
    if result['calibration']:
        calibration_matrix(result['calibration'])  # Refuse an unverified physical scale.
    return result


def calibration_matrix(config):
    image, world = np.asarray(config.get('image', []), np.float32), np.asarray(config.get('world', []), np.float32)
    if image.shape != (4, 2) or world.shape != (4, 2) or not np.isfinite(image).all() or not np.isfinite(world).all():
        raise ValueError('Calibration needs four image points and four measured world points in metres')
    if np.any(image < 0) or np.any(image > 1) or abs(cv2.contourArea(cv2.convexHull(image))) < .001:
        raise ValueError('Calibration image points must span a nondegenerate region')
    if abs(cv2.contourArea(cv2.convexHull(world))) < .01:
        raise ValueError('World points must span a ground plane')
    matrix = cv2.getPerspectiveTransform(image, world)
    validation = np.asarray(config.get('check_image', []), np.float32)
    distance = float(config.get('check_metres', 0))
    if validation.shape != (2, 2) or not np.isfinite(validation).all() or distance <= 0:
        raise ValueError('Add two independent validation points and their measured distance')
    projected = cv2.perspectiveTransform(validation[None], matrix)[0]
    error = abs(float(np.linalg.norm(projected[1]-projected[0]))-distance)/distance
    if error > .1 or not config.get('measured'):
        raise ValueError('Calibration requires measured references and independent error ≤10%')
    return matrix


def ground_speed(history, matrix, valid_area=None):
    """Metres/second from a timestamped ground-plane least-squares fit.

    Reject short, interrupted or geometrically unstable tracks rather than
    presenting detector jitter as a measured physical speed.
    """
    if matrix is None or len(history)<4:return None
    samples=[p for p in history if history[-1][0]-p[0]<=.85]
    if len(samples)<4 or samples[-1][0]-samples[0][0]<.35:return None
    times=np.asarray([p[0]-samples[-1][0] for p in samples],float)
    if np.any(np.diff(times)<=0) or np.max(np.diff(times))>.5:return None
    if valid_area and any(not inside(p[1:],valid_area) for p in samples):return None
    world=cv2.perspectiveTransform(np.asarray([[p[1:] for p in samples]],np.float32),matrix)[0].astype(float)
    times-=times.mean();centred=world-world.mean(axis=0)
    velocity=(times[:,None]*centred).sum(axis=0)/np.dot(times,times)
    residual=float(np.sqrt(np.mean(np.sum((centred-times[:,None]*velocity)**2,axis=1))))
    speed=float(np.linalg.norm(velocity))
    if not np.isfinite(speed) or residual>.8 or speed>160/3.6:return None
    return speed


def suggestions(image):
    """Unverified image heuristics, never a trained lane segmentation claim."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 35), (180, 75, 220))
    mask[:height//3] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > width*height*.03:
            hull = cv2.approxPolyDP(cv2.convexHull(contour), .025*cv2.arcLength(contour, True), True)
            if len(hull) >= 3:
                candidates.append(dict(id=uuid.uuid4().hex, kind='road', name='Unverified road suggestion',
                    points=(hull[:, 0]/[width,height]).tolist(), color='#fbbf24', visible=True, direction=[0,-1]))
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 70, 160)
    edges[:height//3] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 45, minLineLength=width//8, maxLineGap=15)
    return {'items': candidates, 'markings': [] if lines is None else (lines[:20,0]/[width,height,width,height]).tolist(),
            'warning': 'OpenCV appearance suggestions only. Inspect and edit before saving; edges may be kerbs, shadows or vehicles.'}
