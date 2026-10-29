"""Bound live metadata independently of the full track inspection history."""
from __future__ import annotations

MAX_TRAJECTORY_SAMPLES = 40  # Covers a .85-second speed fit even at 30 AI FPS.


def compact_frame(frame: dict) -> dict:
    """Return a detached live payload; inspection data remains in TrafficEngine.

    Track snapshots are immutable after publication. Copy the small containers
    here, rather than repeatedly serializing up to 40 detections and 50 events
    per vehicle at inference cadence.
    """
    result = {**frame, 'tracks': []}
    for track in frame.get('tracks', ()):
        item = {key: value for key, value in track.items()
                if key not in ('events', 'detection_history', 'history')}
        item['history'] = list(track.get('history', ())[-MAX_TRAJECTORY_SAMPLES:])
        result['tracks'].append(item)
    return result
