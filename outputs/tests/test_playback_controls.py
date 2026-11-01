import pytest
from unittest.mock import MagicMock, patch
from services.stream_manager import StreamManager
from app import create_app


def test_stream_manager_pause_resume():
    manager = StreamManager()
    assert not manager.is_paused
    
    # Test pause
    res = manager.pause()
    assert res['is_paused'] is True
    assert manager.is_paused is True
    
    # Test resume
    res = manager.resume()
    assert res['is_paused'] is False
    assert manager.is_paused is False


def test_stream_manager_seek_and_skip():
    manager = StreamManager()
    manager.playback_duration = 300.0
    manager.playback_position = 50.0
    
    # Test seek
    res = manager.seek(120.0)
    assert res['position'] == 120.0
    assert manager.playback_position == 120.0
    
    # Test skip forward +30s
    res = manager.skip(30.0)
    assert res['position'] == 150.0
    assert manager.playback_position == 150.0
    
    # Test skip backward -30s
    res = manager.skip(-30.0)
    assert res['position'] == 120.0
    assert manager.playback_position == 120.0
    
    # Test boundary clamping at 0
    res = manager.skip(-200.0)
    assert res['position'] == 0.0
    assert manager.playback_position == 0.0
    
    # Test boundary clamping at duration
    res = manager.skip(500.0)
    assert res['position'] == 300.0
    assert manager.playback_position == 300.0


def test_playback_position_preserved_on_reconnect():
    manager = StreamManager()
    manager.playback_position = 85.5
    
    # When restarting/reconnecting the same stream, playback_position must NOT reset to 0
    manager.restart()
    assert manager.playback_position == 85.5
    
    # Only when a new stream ID is authorized does position reset to 0
    manager.stream_id = "streamA1234"
    manager.authorize("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert manager.stream_id == "dQw4w9WgXcQ"
    assert manager.playback_position == 0.0


def test_video_control_endpoints(tmp_path):
    db_path = tmp_path / "traffic_test.sqlite3"
    app = create_app(start_services=False, database=db_path)
    client = app.test_client()
    
    # Get state
    resp = client.get('/api/video/state')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'is_paused' in data
    assert 'position' in data
    
    # Test pause
    resp = client.post('/api/video/control', json={'action': 'pause'})
    assert resp.status_code == 200
    assert resp.get_json()['is_paused'] is True
    
    # Test resume
    resp = client.post('/api/video/control', json={'action': 'resume'})
    assert resp.status_code == 200
    assert resp.get_json()['is_paused'] is False
    
    # Test seek
    resp = client.post('/api/video/control', json={'action': 'seek', 'seconds': 45.0})
    assert resp.status_code == 200
    assert resp.get_json()['position'] == 45.0
    
    # Test skip +30s
    resp = client.post('/api/video/control', json={'action': 'skip', 'delta': 30.0})
    assert resp.status_code == 200
    assert resp.get_json()['position'] == 75.0
    
    # Test skip -30s
    resp = client.post('/api/video/control', json={'action': 'skip', 'delta': -30.0})
    assert resp.status_code == 200
    assert resp.get_json()['position'] == 45.0
    
    # Test invalid action
    resp = client.post('/api/video/control', json={'action': 'invalid'})
    assert resp.status_code == 400
