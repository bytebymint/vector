"""Keep historical totals identical while the dashboard uses covering indexes."""
import time

from services.storage import Storage


def test_indexed_summary_counts_only_confirmed_crossings_and_all_tracks(tmp_path):
    storage = Storage(tmp_path / 'analytics.sqlite3')
    now = time.time()
    try:
        for i, category in enumerate(('car', 'car', 'motorcycle')):
            storage.enqueue('track', dict(uid=f't:{i}', first_seen=now-10,
                last_seen=now, **{'class': category}, lane=None, duration=10))
        for i, (kind, confirmed, role, lane) in enumerate([
            ('crossing', True, 'in', 'lane-a'),
            ('crossing', True, 'out', None),
            ('crossing', True, 'in', 'lane-a'),
            ('crossing', False, 'out', 'lane-a'),
            ('observed', True, 'in', None),
        ]):
            storage.enqueue('event', dict(key=f'event-{i}', timestamp=now,
                type=kind, confirmed=confirmed, role=role, lane=lane,
                direction='positive_to_negative', uid='t:0', **{'class': 'car'}))
        storage.flush()
        result = storage.summary()
        assert result['unique'] == 3
        assert result['classes'] == {'car': 2, 'motorcycle': 1}
        assert result['roles'] == {'in': 2, 'out': 1}
        assert result['directions'] == {'positive_to_negative': 3}
        assert result['lane_crossings'] == {'lane-a': 2, 'Unassigned': 1}
        assert sum(row['count'] for row in result['flow']) == 3
        assert result['observed_per_minute'] == 3
        assert result['average_observed_dwell_seconds'] == 10
    finally:
        storage.close()
