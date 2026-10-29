"""Conservative lane-corridor proposals learned from observed vehicle paths.

These are observed routes, not a trained lane-marking detector. Turning, empty,
overlapping and sparsely observed lanes require operator review.
"""
from collections import deque
import math
import time
import uuid
import numpy as np


class LaneLearner:
    def __init__(self):
        self.epoch = None
        self.paths = {}

    def observe(self, epoch, tracks):
        if epoch != self.epoch:
            self.epoch = epoch
            self.paths.clear()
        now = time.monotonic()
        for track in tracks:
            if not track.get('confirmed') or not track.get('history'): continue
            uid = track['uid']
            record = self.paths.setdefault(uid, {'points': deque(maxlen=240), 'updated': now})
            last = record['points'][-1][0] if record['points'] else -math.inf
            record['points'].extend(tuple(p) for p in track['history'] if p[0] > last)
            record['updated'] = now
        self.paths = {k: v for k, v in self.paths.items() if now-v['updated'] < 600}
        while len(self.paths) > 250: del self.paths[next(iter(self.paths))]

    def snapshot(self):
        return [list(v['points']) for v in self.paths.values()]


def suggest_lanes(paths, width, height):
    scale = np.array([width, height], float)
    diagonal = float(np.linalg.norm(scale))
    routes = []
    for history in paths:
        if len(history) < 8: continue
        p = np.array([row[1:3] for row in history], float) * scale
        if not np.isfinite(p).all() or np.linalg.norm(p[-1]-p[0]) < diagonal*.1: continue
        mean = p.mean(axis=0)
        _, singular, vectors = np.linalg.svd(p-mean, full_matrices=False)
        if len(singular) < 2 or singular[0] < 1 or singular[1]/singular[0] > .12: continue
        direction = vectors[0]
        if np.dot(direction, p[-1]-p[0]) < 0: direction = -direction
        routes.append(dict(points=p, mean=mean, direction=direction))
    groups = []
    for route in routes:
        chosen = None
        for group in groups:
            ref = group[0]
            d = ref['direction']; normal = np.array([-d[1], d[0]])
            lateral = abs(float(np.dot(route['mean']-ref['mean'], normal)))
            a, b = route['points']@d, ref['points']@d
            overlap = min(a.max(), b.max()) - max(a.min(), b.min())
            if np.dot(d, route['direction']) > .97 and lateral < diagonal*.013 and overlap > diagonal*.04:
                chosen = group; break
        if chosen is None: groups.append([route])
        else: chosen.append(route)
    items = []
    for group in sorted(groups, key=len, reverse=True):
        if len(group) < 3: continue
        d = np.mean([r['direction'] for r in group], axis=0); d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        cloud = np.concatenate([r['points'] for r in group])
        along = cloud@d; lateral = cloud@n
        low, high = np.percentile(along, [15, 85])
        center = float(np.median(lateral))
        radius = max(diagonal*.012, float(np.percentile(abs(lateral-center), 90))+diagonal*.004)
        if high-low < diagonal*.08: continue
        corners = np.array([d*low+n*(center-radius), d*low+n*(center+radius),
                            d*high+n*(center+radius), d*high+n*(center-radius)]) / scale
        corners = np.clip(corners, .002, .998)
        direction_normalized=d/scale
        angle=math.atan2(direction_normalized[1],direction_normalized[0])
        compass=[[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]
        direction=compass[round(angle/(math.pi/4))%8]
        items.append(dict(id='learned-'+uuid.uuid4().hex[:12], kind='lane',
            name=f'Lane {len(items)+1}', points=corners.tolist(),
            color=['#56ded2','#8eb6ff','#f5c36b','#d7a6f5'][len(items)%4],
            direction=direction, visible=True, role=None,
            distance_metres=None, suggestion=dict(method='observed straight vehicle paths',
                supporting_tracks=len(group), reviewed=False)))
        if len(items) == 12: break
    return dict(items=items, observed_tracks=len(paths), usable_paths=len(routes),
                warning='Review every corridor against the image. These are observed traffic paths; empty lanes and turning lanes may be missed. Move the four corners to the actual lane edges, then enter the measured start-to-end distance.')
