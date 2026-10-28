"""Session-scoped, confirmed person tracks and live count."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .road_geometry import normalize_box


@dataclass
class PersonTrack:
    ident: int
    first: float
    last: float
    bbox: np.ndarray
    confidence: float
    hits: int = 0
    confirmed: bool = False


class PeopleCounter:
    """Count unique confirmed tracker IDs once per active camera/tracker session."""
    def __init__(self):
        self.epoch = None
        self.tracks: dict[int, PersonTrack] = {}
        self.counted_ids: set[int] = set()
        self.unique_count = 0
        self.current: list[dict] = []

    def reset(self, epoch):
        self.epoch = epoch
        self.tracks.clear()
        self.counted_ids.clear()
        self.unique_count = 0
        self.current = []

    def update(self, rows, stamp, width, height, lifetime=2.5, confirm_hits=2):
        seen = set()
        for row in rows:
            if len(row) < 7:
                continue
            box = np.asarray(row[:4], dtype=float)
            ident, confidence = int(row[4]), float(row[5])
            if not np.isfinite(box).all() or not np.isfinite(confidence) or box[2] <= box[0] or box[3] <= box[1]:
                continue
            track = self.tracks.get(ident)
            if track is None or stamp - track.last > lifetime:
                track = PersonTrack(ident, stamp, stamp, box.copy(), confidence)
                self.tracks[ident] = track
            else:
                # Favor the latest observation so boxes do not trail walkers,
                # while retaining enough filtering to suppress detector jitter.
                track.bbox = .78 * box + .22 * track.bbox
                track.confidence = .78 * confidence + .22 * track.confidence
                track.last = stamp
            track.hits += 1
            # Strong one-frame detections can enter the visible overlay now;
            # weaker detections must persist across observations before counting.
            if not track.confirmed and (
                track.hits >= max(2, int(confirm_hits)) or track.confidence >= .55
            ):
                track.confirmed = True
                if ident not in self.counted_ids:
                    self.counted_ids.add(ident)
                    self.unique_count += 1
            seen.add(ident)
        for ident, track in list(self.tracks.items()):
            if stamp - track.last > lifetime:
                del self.tracks[ident]
        self.current = [
            dict(id=t.ident, uid=f'{self.epoch}:{t.ident}', bbox=normalize_box(t.bbox, width, height),
                 confidence=t.confidence, hits=t.hits, confirmed=t.confirmed,
                 first_seen=t.first, last_seen=t.last, duration=max(0, t.last-t.first))
            for ident, t in self.tracks.items() if ident in seen
        ]
        return self.current

    def state(self):
        visible = sum(person['confirmed'] for person in self.current)
        return dict(visible=visible, unique=self.unique_count, tracks=list(self.current),
                    session=self.epoch)
