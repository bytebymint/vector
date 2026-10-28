"""Finite-segment crossings with a spatial deadband and per-ID idempotency."""
from dataclasses import dataclass, field
from .road_geometry import signed_distance, segments_intersect


@dataclass
class LineState:
    side: int = 0
    point: tuple | None = None
    timestamp: float = 0
    counted: set = field(default_factory=set)

    def update(self, point, timestamp, line, width, height, deadband=4., max_gap=3.0):
        distance = signed_distance(point, line, width, height)
        side = 1 if distance > deadband else -1 if distance < -deadband else 0
        if side == 0:
            return None
        event = None
        if self.side and side != self.side and self.point is not None and segments_intersect(self.point, point, *line):
            direction = 'negative_to_positive' if side == 1 else 'positive_to_negative'
            if direction not in self.counted:
                self.counted.add(direction)
                event = {'direction': direction, 'confirmed': timestamp-self.timestamp < max_gap,
                         'crossing_gap_seconds': timestamp-self.timestamp}
        self.side, self.point, self.timestamp = side, tuple(point), timestamp
        return event
