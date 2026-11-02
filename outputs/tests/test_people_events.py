"""Unit tests verifying PeopleEventEmitter real-time communication service.

Tests cover:
- Event formatting and compliance with interface contracts schemas
- Ensuring no heavy pixel arrays or raw image bytes are sent over WebSocket
- Rate-limiting and throttling behavior for frames (e.g. 15-20 fps) and state (1 Hz)
- Client connection tracking, disconnection handling, and safe no-op when 0 clients connected
- Multi-threaded concurrency and Flask-SocketIO integration
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pytest

from services.people_events import (
    PeopleEventEmitter,
    sanitize_metadata,
    assert_no_image_bytes,
    EVENT_PEOPLE_FRAME,
    EVENT_PEOPLE_CAPTURE,
    EVENT_PEOPLE_PROFILE_UPDATE,
    EVENT_PEOPLE_ENHANCEMENT_READY,
    EVENT_PEOPLE_STATE,
)
from services.face_contracts import (
    FramePacket,
    PersonTrack,
    FaceObservation,
    FacePose,
    FaceQualityResult,
    CaptureRecord,
    PersonProfile,
    AnalysisState,
)


class MockSocketIO:
    """Mock Flask-SocketIO instance for fast, deterministic unit tests."""

    def __init__(self) -> None:
        self.emitted: List[Dict[str, Any]] = []
        self.handlers: Dict[str, Any] = {}

    def emit(self, event: str, data: Any, namespace: str = "/") -> None:
        self.emitted.append({"event": event, "data": data, "namespace": namespace})

    def on(self, event: str, namespace: str = "/"):
        def decorator(f):
            self.handlers[f"{namespace}:{event}"] = f
            return f
        return decorator

    def clear(self) -> None:
        self.emitted.clear()


def test_dictionary_frame_keeps_face_tracks_and_nested_metadata():
    socket=MockSocketIO()
    emitter=PeopleEventEmitter(socketio=socket)
    emitter.on_connect('test')
    emitter.emit_frame({'timestamp':1.,'tracking_kind':'face','epoch':'test',
                        'tracks':[{'track_id':7,'tracking_kind':'face','confirmed':False,
                                   'bbox':[100,100,150,160],'confidence':.8}],
                        'faces':[{'track_id':7,'bbox':[100,100,150,160],'confidence':.8,
                                  'quality':{'score':32.,'is_acceptable':False},
                                  'pose':{'category':'left_angle','yaw':-20.,'pitch':0.}}]},force=True)
    payload=socket.emitted[-1]['data']
    assert len(payload['persons'])==1
    person=payload['persons'][0]
    assert person['track_id']==7 and person['tracking_kind']=='face'
    assert person['confirmed'] is False
    assert person['face']['quality']['score']==32.
    assert person['face']['pose']['yaw']==-20.


class FakeClock:
    """Controllable clock for deterministic rate-limit testing."""

    def __init__(self, start_time: float = 1000.0) -> None:
        self._current_time = float(start_time)

    def __call__(self) -> float:
        return self._current_time

    def advance(self, seconds: float) -> None:
        self._current_time += float(seconds)

    def set(self, time_val: float) -> None:
        self._current_time = float(time_val)


# =============================================================================
# 1. Schema and Formatting Verification
# =============================================================================

def test_emit_frame_schema_with_dataclasses():
    """Verify people:frame formatting when supplied PersonTrack and FaceObservation dataclasses."""
    mock_sio = MockSocketIO()
    emitter = PeopleEventEmitter(socketio=mock_sio)
    emitter.clients = 1  # 1 client connected

    track = PersonTrack(
        track_id=1,
        session_id="sess_01:1",
        first_seen=10.0,
        last_seen=12.5,
        bbox=(50.0, 60.0, 200.0, 450.0),
        confidence=0.92,
        confirmed=True,
    )
    pose = FacePose(yaw=-5.0, pitch=3.0, roll=0.0, category="frontal")
    quality = FaceQualityResult(
        score=87.5, sharpness=110.0, motion_blur=0.05,
        resolution_score=90.0, lighting_score=85.0, pose_score=88.0, size_score=85.0
    )
    face = FaceObservation(
        observation_id="obs_01",
        track_id=1,
        timestamp=12.5,
        bbox=(75.0, 70.0, 150.0, 160.0),
        confidence=0.96,
        landmarks=[(90.0, 100.0), (130.0, 100.0), (110.0, 120.0), (95.0, 140.0), (125.0, 140.0)],
        pose=pose,
        quality=quality,
        face_crop=np.zeros((90, 75, 3), dtype=np.uint8),  # Must NOT be emitted!
    )

    result = emitter.emit_frame(
        sequence=42,
        timestamp=12.5,
        source_id="camera_main",
        fps=19.8,
        tracks=[track],
        faces={1: face},
    )

    assert result is True
    assert len(mock_sio.emitted) == 1
    event = mock_sio.emitted[0]
    assert event["event"] == EVENT_PEOPLE_FRAME
    data = event["data"]

    # Frame-level schema validation
    assert data["sequence"] == 42
    assert data["timestamp"] == 12.5
    assert data["source_id"] == "camera_main"
    assert data["fps"] == 19.8
    assert len(data["persons"]) == 1

    person = data["persons"][0]
    assert person["track_id"] == 1
    assert person["session_id"] == "sess_01:1"
    assert person["bbox"] == [50.0, 60.0, 200.0, 450.0]
    assert person["confidence"] == 0.92
    assert person["state"] == "active"

    # Face-level metadata validation
    assert person["face"] is not None
    assert person["face"]["bbox"] == [75.0, 70.0, 150.0, 160.0]
    assert person["face"]["pose_category"] == "frontal"
    assert person["face"]["quality_score"] == 87.5
    assert person["face"]["yaw"] == -5.0
    assert len(person["face"]["landmarks"]) == 5

    # Convenient flat aliases
    assert person["face_bbox"] == [75.0, 70.0, 150.0, 160.0]
    assert person["pose_category"] == "frontal"
    assert person["quality_score"] == 87.5

    # Verify no raw image bytes leaked
    assert_no_image_bytes(data)


def test_emit_frame_schema_with_multiple_persons_and_missing_face():
    """Verify people:frame handles multiple persons where some lack face observations."""
    mock_sio = MockSocketIO()
    emitter = PeopleEventEmitter(socketio=mock_sio)
    emitter.clients = 1

    track1 = PersonTrack(track_id=1, session_id="s1:1", first_seen=1.0, last_seen=2.0,
                         bbox=(10.0, 20.0, 100.0, 200.0), confidence=0.89)
    track2 = PersonTrack(track_id=2, session_id="s1:2", first_seen=1.5, last_seen=2.0,
                         bbox=(300.0, 40.0, 450.0, 350.0), confidence=0.75)

    face1 = FaceObservation(
        observation_id="obs_1", track_id=1, timestamp=2.0,
        bbox=(20.0, 25.0, 60.0, 70.0), confidence=0.9,
        pose=FacePose(yaw=10.0, pitch=-2.0, roll=0.0, category="right_angle"),
        quality=FaceQualityResult(score=79.2, sharpness=95.0, motion_blur=0.1,
                                 resolution_score=80.0, lighting_score=75.0,
                                 pose_score=78.0, size_score=75.0)
    )

    emitter.emit_frame(sequence=10, tracks=[track1, track2], faces={1: face1})

    data = mock_sio.emitted[0]["data"]
    assert len(data["persons"]) == 2

    # Track 1 has face
    p1 = next(p for p in data["persons"] if p["track_id"] == 1)
    assert p1["face"] is not None
    assert p1["pose_category"] == "right_angle"
    assert p1["quality_score"] == 79.2

    # Track 2 does not have face
    p2 = next(p for p in data["persons"] if p["track_id"] == 2)
    assert p2["face"] is None
    assert p2["face_bbox"] is None
    assert p2["quality_score"] is None


def test_emit_capture_schema():
    """Verify people:capture event formatting and URL generation."""
    mock_sio = MockSocketIO()
    emitter = PeopleEventEmitter(socketio=mock_sio)
    emitter.clients = 1

    # Using CaptureRecord dataclass
    record = CaptureRecord(
        capture_id="cap_20261024_001",
        track_id=7,
        session_id="session_abc:7",
        timestamp=105.4,
        category="frontal",
        quality_score=91.4,
        pose_category="frontal",
        yaw=2.1,
        pitch=-1.4,
        original_path="data/captures/cap_20261024_001_orig.jpg",
        crop_width=160,
        crop_height=160,
    )

    res = emitter.emit_capture(record)
    assert res is True
    assert len(mock_sio.emitted) == 1
    event = mock_sio.emitted[0]
    assert event["event"] == EVENT_PEOPLE_CAPTURE

    data = event["data"]
    assert data["session_id"] == "session_abc:7"
    assert data["capture_id"] == "cap_20261024_001"
    assert data["category"] == "frontal"
    assert data["quality_score"] == 91.4
    assert "/api/people/captures/cap_20261024_001/image" in data["crop_url"]
    assert "thumbnail_url" in data
    assert_no_image_bytes(data)

    # Also test with raw kwargs
    mock_sio.clear()
    emitter.emit_capture(
        session_id="sess_kw:9",
        capture_id="cap_kw_999",
        category="left_angle",
        quality_score=82.7,
        thumbnail_url="/api/people/captures/cap_kw_999/thumb.jpg",
    )
    data2 = mock_sio.emitted[0]["data"]
    assert data2["session_id"] == "sess_kw:9"
    assert data2["capture_id"] == "cap_kw_999"
    assert data2["category"] == "left_angle"
    assert data2["quality_score"] == 82.7
    assert data2["thumbnail_url"] == "/api/people/captures/cap_kw_999/thumb.jpg"


def test_emit_profile_update_schema():
    """Verify people:profile_update event formatting."""
    mock_sio = MockSocketIO()
    emitter = PeopleEventEmitter(socketio=mock_sio)
    emitter.clients = 1

    profile = PersonProfile(
        session_id="session_01:5",
        track_id=5,
        camera_id="cam_east",
        first_seen=100.0,
        last_seen=112.5,
        duration=12.5,
        status="active",
        total_observations=18,
        best_quality_score=89.6,
        metadata={"note": "verified"},
    )

    res = emitter.emit_profile_update(profile)
    assert res is True
    assert len(mock_sio.emitted) == 1
    event = mock_sio.emitted[0]
    assert event["event"] == EVENT_PEOPLE_PROFILE_UPDATE

    data = event["data"]
    assert data["session_id"] == "session_01:5"
    assert data["track_id"] == 5
    assert data["status"] == "active"
    assert data["duration"] == 12.5
    assert data["best_quality_score"] == 89.6
    assert data["total_observations"] == 18
    assert data["metadata"] == {"note": "verified"}
    assert_no_image_bytes(data)


def test_emit_enhancement_ready_schema():
    """Verify people:enhancement_ready event formatting."""
    mock_sio = MockSocketIO()
    emitter = PeopleEventEmitter(socketio=mock_sio)
    emitter.clients = 1

    res = emitter.emit_enhancement_ready(
        capture_id="cap_enh_123",
        session_id="sess_01:3",
        enhanced_url="/api/people/captures/cap_enh_123/image?type=enhanced",
        enhancement_method="fsrcnn_x2",
        quality_score=94.2,
        track_id=3,
    )
    assert res is True
    assert len(mock_sio.emitted) == 1
    event = mock_sio.emitted[0]
    assert event["event"] == EVENT_PEOPLE_ENHANCEMENT_READY

    data = event["data"]
    assert data["capture_id"] == "cap_enh_123"
    assert data["session_id"] == "sess_01:3"
    assert data["enhanced_url"] == "/api/people/captures/cap_enh_123/image?type=enhanced"
    assert data["enhancement_method"] == "fsrcnn_x2"
    assert data["quality_score"] == 94.2
    assert data["track_id"] == 3
    assert_no_image_bytes(data)


def test_emit_state_schema():
    """Verify people:state event formatting with AnalysisState and aliases."""
    mock_sio = MockSocketIO()
    emitter = PeopleEventEmitter(socketio=mock_sio)
    emitter.clients = 1

    state = AnalysisState(
        is_running=True,
        active_tracks_count=4,
        total_profiles_count=12,
        total_captures_count=25,
        fps=18.4,
        status_message="Processing",
    )

    res = emitter.emit_state(state, force=True)
    assert res is True
    assert len(mock_sio.emitted) == 1
    event = mock_sio.emitted[0]
    assert event["event"] == EVENT_PEOPLE_STATE

    data = event["data"]
    assert data["is_running"] is True
    # Verify both standard and count-suffix aliases are present
    assert data["active_tracks"] == 4
    assert data["active_tracks_count"] == 4
    assert data["total_profiles"] == 12
    assert data["total_profiles_count"] == 12
    assert data["total_captures"] == 25
    assert data["total_captures_count"] == 25
    assert data["fps"] == 18.4
    assert data["status_message"] == "Processing"
    assert_no_image_bytes(data)


# =============================================================================
# 2. Strict Bandwidth Protection: No Raw Pixel Bytes
# =============================================================================

def test_frame_metadata_excludes_raw_heavy_pixel_bytes():
    """Verify raw image arrays, bytes, and buffers are strictly stripped."""
    mock_sio = MockSocketIO()
    emitter = PeopleEventEmitter(socketio=mock_sio)
    emitter.clients = 1

    # Heavy payload with prohibited raw image types
    full_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    crop_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    raw_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF" * 1000
    memory_buf = memoryview(b"\x00\x01\x02\x03" * 500)

    heavy_dict = {
        "sequence": 100,
        "timestamp": 50.0,
        "image": full_image,
        "raw_bytes": raw_jpeg,
        "persons": [
            {
                "track_id": 1,
                "bbox": np.array([10.0, 20.0, 80.0, 160.0]),
                "face_crop": crop_face,
                "thumbnail_bytes": memory_buf,
                "face": {
                    "observation_id": "obs_1",
                    "bbox": [15.0, 25.0, 75.0, 90.0],
                    "crop_original": crop_face,
                    "pose_category": "frontal",
                    "quality_score": 86.0,
                }
            }
        ]
    }

    emitter.emit_frame(heavy_dict)
    assert len(mock_sio.emitted) == 1
    emitted_data = mock_sio.emitted[0]["data"]

    # Prohibited keys must be stripped
    assert "image" not in emitted_data
    assert "raw_bytes" not in emitted_data

    person_out = emitted_data["persons"][0]
    assert "face_crop" not in person_out
    assert "thumbnail_bytes" not in person_out
    assert "crop_original" not in person_out["face"]

    # Verify bbox was preserved and converted from ndarray to python list
    assert person_out["bbox"] == [10.0, 20.0, 80.0, 160.0]
    assert isinstance(person_out["bbox"], list)
    assert all(isinstance(v, float) for v in person_out["bbox"])

    # Payload must pass strict assertion
    assert_no_image_bytes(emitted_data)


def test_sanitize_metadata_handles_all_types():
    """Unit test for sanitize_metadata recursive scrubbing."""
    test_data = {
        "str": "hello",
        "int": 42,
        "float": 3.14159,
        "np_float": np.float32(2.71828),
        "np_int": np.int64(100),
        "np_bool": np.bool_(True),
        "forbidden_image": np.zeros((100, 100, 3), dtype=np.uint8),
        "forbidden_bytes": b"binary_data_here",
        "coord_vector": np.array([1.5, 2.5, 3.5, 4.5]),
        "landmarks_2d": np.array([[10.0, 20.0], [30.0, 40.0]]),
        "nested": {
            "crop_original": np.zeros((50, 50, 3)),
            "score": 95.0,
        }
    }

    cleaned = sanitize_metadata(test_data)
    assert cleaned["str"] == "hello"
    assert cleaned["int"] == 42
    assert cleaned["float"] == 3.1416
    assert cleaned["np_float"] == 2.7183
    assert cleaned["np_int"] == 100
    assert cleaned["np_bool"] is True
    assert "forbidden_image" not in cleaned
    assert "forbidden_bytes" not in cleaned
    assert cleaned["coord_vector"] == [1.5, 2.5, 3.5, 4.5]
    assert cleaned["landmarks_2d"] == [[10.0, 20.0], [30.0, 40.0]]
    assert "crop_original" not in cleaned["nested"]
    assert cleaned["nested"]["score"] == 95.0


# =============================================================================
# 3. Rate-Limiting and Throttling
# =============================================================================

def test_frame_rate_limiting_at_20_fps():
    """Verify frame rate limiter strictly enforces max 20 fps (~0.05s interval)."""
    mock_sio = MockSocketIO()
    clock = FakeClock(1000.0)
    emitter = PeopleEventEmitter(socketio=mock_sio, max_fps=20.0, time_fn=clock)
    emitter.clients = 1

    # Frame 1 at t=0.0 -> Emitted
    assert emitter.emit_frame(sequence=1) is True
    assert len(mock_sio.emitted) == 1

    # Frame 2 at t=0.01 (too fast, 10ms later) -> Throttled
    clock.advance(0.01)
    assert emitter.emit_frame(sequence=2) is False
    assert len(mock_sio.emitted) == 1

    # Frame 3 at t=0.03 (still too fast, 30ms total) -> Throttled
    clock.advance(0.02)
    assert emitter.emit_frame(sequence=3) is False
    assert len(mock_sio.emitted) == 1

    # Frame 4 at t=0.051 (51ms elapsed >= 50ms) -> Emitted
    clock.advance(0.021)
    assert emitter.emit_frame(sequence=4) is True
    assert len(mock_sio.emitted) == 2

    # Frame 5 at t=0.055 (too fast, 4ms later) -> Throttled
    clock.advance(0.004)
    assert emitter.emit_frame(sequence=5) is False
    assert len(mock_sio.emitted) == 2

    # Check metrics
    metrics = emitter.get_metrics()
    assert metrics["emitted_frames"] == 2
    assert metrics["throttled_frames"] == 3


def test_frame_rate_limiting_at_15_fps():
    """Verify frame rate limiter supports configurable max 15 fps (~0.0667s interval)."""
    mock_sio = MockSocketIO()
    clock = FakeClock(1000.0)
    emitter = PeopleEventEmitter(socketio=mock_sio, max_fps=15.0, time_fn=clock)
    emitter.clients = 1

    assert emitter.emit_frame(sequence=1) is True

    # At 50ms later (which would pass 20fps but must fail 15fps):
    clock.advance(0.050)
    assert emitter.emit_frame(sequence=2) is False

    # At 70ms total (>66.7ms for 15fps):
    clock.advance(0.020)
    assert emitter.emit_frame(sequence=3) is True
    assert len(mock_sio.emitted) == 2


def test_frame_force_bypasses_throttling():
    """Verify force=True bypasses frame rate limiting immediately."""
    mock_sio = MockSocketIO()
    clock = FakeClock(1000.0)
    emitter = PeopleEventEmitter(socketio=mock_sio, max_fps=20.0, time_fn=clock)
    emitter.clients = 1

    assert emitter.emit_frame(sequence=1) is True
    # Immediate subsequent call with force=True
    assert emitter.emit_frame(sequence=2, force=True) is True
    assert len(mock_sio.emitted) == 2


def test_state_on_change_and_periodic_1hz_sync():
    """Verify people:state emits on change and 1 Hz periodic sync, throttling unchanged state."""
    mock_sio = MockSocketIO()
    clock = FakeClock(1000.0)
    emitter = PeopleEventEmitter(socketio=mock_sio, state_interval_s=1.0, time_fn=clock)
    emitter.clients = 1

    initial_state = {
        "is_running": True,
        "active_tracks": 2,
        "total_profiles": 5,
        "total_captures": 10,
        "fps": 18.0,
        "status_message": "Tracking",
    }

    # 1. First state emission -> Emitted
    assert emitter.emit_state(initial_state) is True
    assert len(mock_sio.emitted) == 1

    # 2. Same state at t=0.2s (< 1.0s) -> Throttled
    clock.advance(0.2)
    assert emitter.emit_state(initial_state) is False
    assert len(mock_sio.emitted) == 1

    # 3. Same state at t=0.8s (< 1.0s) -> Throttled
    clock.advance(0.6)
    assert emitter.emit_state(initial_state) is False
    assert len(mock_sio.emitted) == 1

    # 4. Same state at t=1.05s (>= 1.0s elapsed) -> Emitted (periodic 1 Hz heartbeat)
    clock.advance(0.25)
    assert emitter.emit_state(initial_state) is True
    assert len(mock_sio.emitted) == 2

    # 5. State CHANGE at t=1.1s (e.g. new capture count) -> Emitted immediately!
    clock.advance(0.05)
    changed_state = dict(initial_state, total_captures=11)
    assert emitter.emit_state(changed_state) is True
    assert len(mock_sio.emitted) == 3

    # 6. Unchanged from changed_state at t=1.2s -> Throttled
    clock.advance(0.1)
    assert emitter.emit_state(changed_state) is False
    assert len(mock_sio.emitted) == 3


# =============================================================================
# 4. Client Connection Tracking, Disconnection & Safe No-Op
# =============================================================================

def test_safe_no_op_when_no_clients_connected():
    """Verify emitter safely no-ops and never emits when client count is 0."""
    mock_sio = MockSocketIO()
    emitter = PeopleEventEmitter(socketio=mock_sio)
    assert emitter.client_count == 0
    assert emitter.has_clients is False

    # Every emit call must return False and emit nothing to SocketIO
    assert emitter.emit_frame(sequence=1) is False
    assert emitter.emit_capture(session_id="s1", capture_id="c1", category="frontal", quality_score=90.0) is False
    assert emitter.emit_profile_update(session_id="s1", track_id=1, status="active") is False
    assert emitter.emit_enhancement_ready(capture_id="c1", session_id="s1", enhanced_url="/url") is False
    assert emitter.emit_state(is_running=True) is False

    assert len(mock_sio.emitted) == 0


def test_client_lifecycle_and_disconnection_handling():
    """Verify connection and disconnection tracking with sids and anonymous clients."""
    mock_sio = MockSocketIO()
    emitter = PeopleEventEmitter(socketio=mock_sio)

    connect_events = []
    disconnect_events = []
    emitter.add_connect_callback(lambda sid: connect_events.append(sid))
    emitter.add_disconnect_callback(lambda sid, reason: disconnect_events.append((sid, reason)))

    # Initial state
    assert emitter.client_count == 0

    # Client A connects with sid
    emitter.on_connect(sid="client_A")
    assert emitter.client_count == 1
    assert emitter.has_clients is True
    assert connect_events == ["client_A"]

    # Client B connects with sid
    emitter.on_connect(sid="client_B")
    assert emitter.client_count == 2

    # Anonymous client connects
    emitter.on_connect()
    assert emitter.client_count == 3

    # Now emissions succeed
    assert emitter.emit_frame(sequence=1) is True
    assert len(mock_sio.emitted) == 1

    # Client A disconnects
    emitter.on_disconnect(sid="client_A", reason="user navigation")
    assert emitter.client_count == 2
    assert disconnect_events[-1] == ("client_A", "user navigation")

    # Anonymous client disconnects
    emitter.on_disconnect()
    assert emitter.client_count == 1

    # Client B disconnects
    emitter.on_disconnect(sid="client_B")
    assert emitter.client_count == 0
    assert emitter.has_clients is False

    # Disconnect when already 0 -> does not go negative
    emitter.on_disconnect(sid="unknown_client")
    assert emitter.client_count == 0

    # Post-disconnection: emissions must now safely no-op again
    assert emitter.emit_frame(sequence=2) is False
    assert len(mock_sio.emitted) == 1  # No new emissions


def test_property_setter_and_reset_metrics():
    """Verify clients property setter and metrics management."""
    emitter = PeopleEventEmitter()
    assert emitter.clients == 0
    emitter.clients = 5
    assert emitter.clients == 5
    assert emitter.client_count == 5

    emitter.clients = 0
    assert emitter.clients == 0

    emitter.reset_metrics()
    metrics = emitter.get_metrics()
    assert metrics["emitted_frames"] == 0
    assert metrics["throttled_frames"] == 0
    assert metrics["emit_errors"] == 0


# =============================================================================
# 5. Multi-Threaded Concurrency
# =============================================================================

def test_multithreaded_concurrent_emission():
    """Verify thread-safety when multiple worker threads emit concurrently."""
    mock_sio = MockSocketIO()
    emitter = PeopleEventEmitter(socketio=mock_sio, max_fps=100.0)
    emitter.clients = 2

    errors = []

    def worker_frames(thread_id: int):
        for seq in range(20):
            try:
                emitter.emit_frame(sequence=seq, timestamp=time.monotonic(), force=True)
            except Exception as e:
                errors.append(e)

    def worker_captures(thread_id: int):
        for i in range(10):
            try:
                emitter.emit_capture(
                    session_id=f"sess_{thread_id}:{i}",
                    capture_id=f"cap_{thread_id}_{i}",
                    category="frontal",
                    quality_score=85.0,
                )
            except Exception as e:
                errors.append(e)

    threads = [
        threading.Thread(target=worker_frames, args=(t,)) for t in range(4)
    ] + [
        threading.Thread(target=worker_captures, args=(t,)) for t in range(2)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert len(errors) == 0
    assert len(mock_sio.emitted) > 0
    assert emitter.get_metrics()["emit_errors"] == 0


# =============================================================================
# 6. Flask and Flask-SocketIO Real Integration
# =============================================================================

def test_flask_socketio_integration():
    """Verify end-to-end event broadcast using real Flask and Flask-SocketIO test client."""
    from flask import Flask
    from flask_socketio import SocketIO

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "secret_for_tests"
    sio = SocketIO(app, async_mode="threading")

    emitter = PeopleEventEmitter(socketio=sio)
    emitter.init_app(app, sio)

    # Connect with Flask-SocketIO test client
    client = sio.test_client(app)
    assert client.is_connected()
    # Ensure emitter recognizes connected client
    emitter.clients = 1

    # Broadcast frame event
    emitter.emit_frame(sequence=101, timestamp=123.456, fps=20.0, force=True)
    received = client.get_received()
    frame_events = [m for m in received if m["name"] == EVENT_PEOPLE_FRAME]
    assert len(frame_events) == 1
    assert frame_events[0]["args"][0]["sequence"] == 101

    # Broadcast capture event
    emitter.emit_capture(
        session_id="profile:1",
        capture_id="cap_test_real",
        category="frontal",
        quality_score=92.5,
    )
    received = client.get_received()
    capture_events = [m for m in received if m["name"] == EVENT_PEOPLE_CAPTURE]
    assert len(capture_events) == 1
    assert capture_events[0]["args"][0]["capture_id"] == "cap_test_real"

    client.disconnect()
