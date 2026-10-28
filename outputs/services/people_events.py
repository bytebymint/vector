"""Real-Time Communication Service for People Tracking & Facial Capture events.

Broadcaster service integrating with Flask-SocketIO to synchronize frontend clients
with real-time people tracking, facial capture, and enhancement events over WebSocket.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

try:
    from services.face_contracts import (
        PersonTrack,
        FaceObservation,
        CaptureRecord,
        PersonProfile,
        AnalysisState,
    )
except ImportError:  # Fallback for relative package execution
    try:
        from .face_contracts import (
            PersonTrack,
            FaceObservation,
            CaptureRecord,
            PersonProfile,
            AnalysisState,
        )
    except ImportError:
        PersonTrack = None
        FaceObservation = None
        CaptureRecord = None
        PersonProfile = None
        AnalysisState = None

logger = logging.getLogger(__name__)

# Canonical WebSocket Event Names
EVENT_PEOPLE_FRAME = "people:frame"
EVENT_PEOPLE_CAPTURE = "people:capture"
EVENT_PEOPLE_PROFILE_UPDATE = "people:profile_update"
EVENT_PEOPLE_ENHANCEMENT_READY = "people:enhancement_ready"
EVENT_PEOPLE_STATE = "people:state"

# Keys that identify raw image or heavy binary payloads to strip
IMAGE_PAYLOAD_KEYS = frozenset({
    "image",
    "frame",
    "frame_bytes",
    "face_crop",
    "crop_original",
    "raw_bytes",
    "bytes",
    "raw_image",
    "thumbnail_bytes",
    "crop",
})


def sanitize_metadata(obj: Any) -> Any:
    """Recursively sanitize metadata for WebSocket transmission.

    Guarantees that raw pixel byte arrays, heavy NumPy images, and binary
    buffers are completely excluded to prevent bandwidth saturation, while
    preserving bounding boxes, scalar scores, landmarks, and nested dict structures.
    """
    if obj is None:
        return None

    if isinstance(obj, (bool, str, int)):
        return obj

    if isinstance(obj, float):
        return round(obj, 4) if np.isfinite(obj) else 0.0

    if isinstance(obj, (np.floating, np.integer)):
        val = obj.item()
        return round(val, 4) if isinstance(val, float) and np.isfinite(val) else val

    if isinstance(obj, np.bool_):
        return bool(obj)

    # Completely strip raw binary buffers
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return None

    # Handle NumPy ndarrays
    if isinstance(obj, np.ndarray):
        # Allow small 1D coordinate vectors (e.g. bbox coordinates [x1, y1, x2, y2])
        if obj.ndim == 1 and obj.size <= 8:
            return [sanitize_metadata(x) for x in obj.tolist()]
        # Allow small 2D coordinate lists (e.g. 5-point face landmarks shape (5, 2))
        elif obj.ndim == 2 and obj.shape[1] in (2, 3) and obj.shape[0] <= 100:
            return [[sanitize_metadata(c) for c in pt] for pt in obj.tolist()]
        else:
            # Multi-dimensional or large array represents image pixels; strip it!
            return None

    # Handle objects with to_dict()
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return sanitize_metadata(obj.to_dict())

    # Handle dictionaries
    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in obj.items():
            key_str = str(k)
            if key_str.lower() in IMAGE_PAYLOAD_KEYS:
                continue
            val = sanitize_metadata(v)
            # Retain non-None values, or explicitly allowed None fields
            if val is not None or key_str in (
                "error",
                "enhanced_url",
                "best_capture",
                "face",
                "pose",
                "quality",
                "face_bbox",
                "pose_category",
                "quality_score",
            ):
                cleaned[key_str] = val
        return cleaned

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        cleaned_list = []
        for item in obj:
            if isinstance(item, (bytes, bytearray, memoryview)):
                continue
            cleaned_item = sanitize_metadata(item)
            if cleaned_item is not None:
                cleaned_list.append(cleaned_item)
        return cleaned_list

    # Fallback to string representation for other types
    return str(obj)


def assert_no_image_bytes(payload: Any) -> None:
    """Validate that a payload contains no heavy image bytes or ndarrays.

    Raises ValueError if raw pixel data or forbidden image keys are detected.
    """
    if isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError("Payload contains raw bytes")
    if isinstance(payload, np.ndarray) and (payload.ndim >= 3 or payload.size > 200):
        raise ValueError(f"Payload contains raw image ndarray shape={payload.shape}")
    if isinstance(payload, dict):
        for k, v in payload.items():
            if str(k).lower() in IMAGE_PAYLOAD_KEYS:
                raise ValueError(f"Payload contains forbidden image key: {k}")
            assert_no_image_bytes(v)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            assert_no_image_bytes(item)


class PeopleEventEmitter:
    """Event broadcaster service integrating with Flask-SocketIO.

    Synchronizes the frontend with People Tracking and Facial Capture pipeline
    events. Implements intelligent rate limiting, connection tracking, payload
    sanitization, and on-change/periodic state synchronization.
    """

    def __init__(
        self,
        socketio: Optional[Any] = None,
        namespace: str = "/",
        max_fps: float = 20.0,
        state_interval_s: float = 1.0,
        time_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        """Initialize the people event broadcaster.

        Args:
            socketio: Optional Flask-SocketIO instance or compatible mock.
            namespace: SocketIO namespace (default '/').
            max_fps: Maximum emission rate for per-frame metadata (default 20.0 fps).
            state_interval_s: Periodic state heartbeat interval in seconds (default 1.0 s).
            time_fn: Time provider function for testing (default time.monotonic).
        """
        self.socketio = socketio
        self.namespace = namespace
        self._max_fps = max(1.0, float(max_fps))
        self.state_interval_s = max(0.1, float(state_interval_s))
        self._time_fn = time_fn or time.monotonic

        self._lock = threading.RLock()
        self._connected_sids: Set[str] = set()
        self._anonymous_clients: int = 0

        # Timing and rate-limiting state
        self._last_frame_emit_time: float = 0.0
        self._last_state_emit_time: float = 0.0
        self._last_state_summary: Optional[Tuple[Any, ...]] = None

        # Telemetry metrics
        self._emitted_frames: int = 0
        self._throttled_frames: int = 0
        self._emitted_captures: int = 0
        self._emitted_profiles: int = 0
        self._emitted_enhancements: int = 0
        self._emitted_states: int = 0
        self._throttled_states: int = 0
        self._emit_errors: int = 0

        # Callbacks
        self._connect_callbacks: List[Callable[[Optional[str]], None]] = []
        self._disconnect_callbacks: List[Callable[[Optional[str], Optional[str]], None]] = []

        if self.socketio is not None:
            self.register_handlers(self.socketio)

    # -------------------------------------------------------------------------
    # Properties & Configuration
    # -------------------------------------------------------------------------

    @property
    def max_fps(self) -> float:
        """Maximum frames per second permitted for frame event broadcast."""
        return self._max_fps

    @max_fps.setter
    def max_fps(self, val: float) -> None:
        with self._lock:
            self._max_fps = max(1.0, float(val))

    @property
    def min_frame_interval(self) -> float:
        """Minimum time interval in seconds between frame emissions."""
        return 1.0 / self._max_fps

    @property
    def client_count(self) -> int:
        """Total number of currently connected frontend clients."""
        with self._lock:
            return len(self._connected_sids) + max(0, self._anonymous_clients)

    @property
    def clients(self) -> int:
        """Alias for client_count matching Flask application convention."""
        return self.client_count

    @clients.setter
    def clients(self, val: int) -> None:
        """Set client count directly (convenience for tests and standalone setups)."""
        with self._lock:
            self._anonymous_clients = max(0, int(val))
            self._connected_sids.clear()

    @property
    def has_clients(self) -> bool:
        """True if one or more clients are actively connected."""
        return self.client_count > 0

    def attach_socketio(self, socketio: Any) -> None:
        """Attach a SocketIO instance and register connection lifecycle hooks."""
        with self._lock:
            self.socketio = socketio
            self.register_handlers(socketio)

    def init_app(self, app: Any, socketio: Optional[Any] = None) -> None:
        """Flask extension init_app pattern."""
        sio = socketio or getattr(app, "extensions", {}).get("socketio")
        if sio is not None:
            self.attach_socketio(sio)
        if hasattr(app, "extensions"):
            app.extensions["people_events"] = self

    # -------------------------------------------------------------------------
    # Connection Lifecycle Handlers
    # -------------------------------------------------------------------------

    def on_connect(self, sid: Optional[str] = None) -> None:
        """Handle new client connection."""
        with self._lock:
            if sid:
                self._connected_sids.add(str(sid))
            else:
                self._anonymous_clients += 1
            current = self.client_count

        logger.debug("Socket.IO client connected (sid=%s, total=%d)", sid, current)
        for cb in list(self._connect_callbacks):
            try:
                cb(sid)
            except Exception as e:
                logger.warning("Error in on_connect callback: %s", e)

    def on_disconnect(self, sid: Optional[str] = None, reason: Optional[str] = None) -> None:
        """Handle client disconnection."""
        with self._lock:
            if sid and str(sid) in self._connected_sids:
                self._connected_sids.discard(str(sid))
            else:
                self._anonymous_clients = max(0, self._anonymous_clients - 1)
            current = self.client_count

        logger.debug("Socket.IO client disconnected (sid=%s, reason=%s, total=%d)", sid, reason, current)
        for cb in list(self._disconnect_callbacks):
            try:
                cb(sid, reason)
            except Exception as e:
                logger.warning("Error in on_disconnect callback: %s", e)

    def add_connect_callback(self, cb: Callable[[Optional[str]], None]) -> None:
        """Register a callback invoked when a client connects."""
        self._connect_callbacks.append(cb)

    def add_disconnect_callback(self, cb: Callable[[Optional[str], Optional[str]], None]) -> None:
        """Register a callback invoked when a client disconnects."""
        self._disconnect_callbacks.append(cb)

    def register_handlers(self, socketio: Optional[Any] = None) -> None:
        """Register SocketIO connect and disconnect event listeners."""
        sio = socketio or self.socketio
        if sio is None or not hasattr(sio, "on"):
            return

        try:
            @sio.on("connect", namespace=self.namespace)
            def _handle_connect(*args, **kwargs):
                sid = None
                try:
                    import flask
                    if flask.has_request_context():
                        sid = getattr(flask.request, "sid", None)
                except Exception:
                    pass
                self.on_connect(sid=sid)

            @sio.on("disconnect", namespace=self.namespace)
            def _handle_disconnect(*args, **kwargs):
                sid = None
                reason = kwargs.get("reason")
                if args and isinstance(args[0], str):
                    reason = args[0]
                try:
                    import flask
                    if flask.has_request_context():
                        sid = getattr(flask.request, "sid", None)
                except Exception:
                    pass
                self.on_disconnect(sid=sid, reason=reason)
        except Exception as e:
            logger.debug("Handler registration with namespace encountered: %s", e)

    # -------------------------------------------------------------------------
    # Internal Emission Helper
    # -------------------------------------------------------------------------

    def _emit(self, event: str, payload: Dict[str, Any]) -> bool:
        """Emit payload over Socket.IO to connected clients."""
        if self.socketio is None or not self.has_clients:
            return False

        try:
            if self.namespace and self.namespace != "/":
                try:
                    self.socketio.emit(event, payload, namespace=self.namespace)
                except TypeError:
                    self.socketio.emit(event, payload)
            else:
                self.socketio.emit(event, payload)
            return True
        except Exception as exc:
            with self._lock:
                self._emit_errors += 1
            logger.warning("Failed emitting %s event: %s", event, exc)
            return False

    # -------------------------------------------------------------------------
    # Event Broadcasters
    # -------------------------------------------------------------------------

    def emit_frame(
        self,
        frame_or_data: Optional[Any] = None,
        tracks: Optional[List[Any]] = None,
        faces: Optional[Union[Dict[int, Any], List[Any]]] = None,
        *,
        sequence: Optional[int] = None,
        timestamp: Optional[float] = None,
        source_id: Optional[str] = None,
        fps: Optional[float] = None,
        persons: Optional[List[Dict[str, Any]]] = None,
        force: bool = False,
    ) -> bool:
        """Broadcast lightweight per-frame metadata (people:frame).

        Throttled to max_fps (default 15-20 fps). Completely excludes raw image pixels.
        Safe no-op when no clients are connected.

        Args:
            frame_or_data: Dict or FramePacket object containing sequence/timestamp metadata.
            tracks: Optional list of PersonTrack objects or track dicts.
            faces: Optional dict (track_id -> FaceObservation) or list of FaceObservation objects.
            sequence: Explicit frame sequence index.
            timestamp: Explicit acquisition timestamp.
            source_id: Camera/source identifier.
            fps: Current processing frame rate.
            persons: Explicit formatted list of active person metadata dictionaries.
            force: Bypass rate limiting if True.

        Returns:
            bool: True if event was emitted, False if throttled or no clients connected.
        """
        if not self.has_clients:
            return False

        now = self._time_fn()
        with self._lock:
            elapsed = now - self._last_frame_emit_time
            if not force and elapsed < self.min_frame_interval:
                self._throttled_frames += 1
                return False
            self._last_frame_emit_time = now
            self._emitted_frames += 1

        payload: Dict[str, Any] = {}

        # 1. Base frame timing and sequence
        if isinstance(frame_or_data, dict):
            payload.update(frame_or_data)
        elif frame_or_data is not None and hasattr(frame_or_data, "to_dict"):
            payload.update(frame_or_data.to_dict())

        if sequence is not None:
            payload["sequence"] = int(sequence)
        elif "sequence" not in payload:
            payload["sequence"] = 0

        if timestamp is not None:
            payload["timestamp"] = float(timestamp)
        elif "timestamp" not in payload:
            payload["timestamp"] = round(time.time(), 3)

        if source_id is not None:
            payload["source_id"] = str(source_id)
        elif "source_id" not in payload:
            payload["source_id"] = "default"

        if fps is not None:
            payload["fps"] = round(float(fps), 1)

        # 2. Persons / Tracks metadata
        formatted_persons: List[Dict[str, Any]] = []
        if tracks is None and persons is None and 'persons' not in payload:
            tracks = payload.get('tracks')
        if faces is None:
            faces = payload.get('faces')

        if persons is not None:
            for p in persons:
                if isinstance(p, dict):
                    formatted_persons.append(self._format_person_dict(p))
                elif hasattr(p, "to_dict"):
                    formatted_persons.append(self._format_person_dict(p.to_dict()))
        elif tracks is not None:
            # Map faces by track_id for lookup
            face_map: Dict[int, Any] = {}
            if isinstance(faces, dict):
                face_map = faces
            elif isinstance(faces, list):
                for f in faces:
                    tid = getattr(f, "track_id", None) if not isinstance(f, dict) else f.get("track_id")
                    if tid is not None:
                        face_map[int(tid)] = f

            for track in tracks:
                track_dict = track.to_dict() if hasattr(track, "to_dict") else dict(track)
                tid = track_dict.get("track_id", track_dict.get("id"))
                matching_face = face_map.get(int(tid)) if tid is not None else None
                formatted_persons.append(self._format_track_and_face(track_dict, matching_face))
        elif "persons" in payload and isinstance(payload["persons"], list):
            formatted_persons = [
                self._format_person_dict(p if isinstance(p, dict) else p.to_dict())
                for p in payload["persons"]
            ]

        payload["persons"] = formatted_persons

        # 3. Strip any heavy binary image data
        clean_payload = sanitize_metadata(payload)

        # 4. Transmit
        return self._emit(EVENT_PEOPLE_FRAME, clean_payload)

    def _format_track_and_face(self, track_dict: Dict[str, Any], face: Optional[Any]) -> Dict[str, Any]:
        """Combine person track and localized face observation into frame schema."""
        tid = track_dict.get("track_id", track_dict.get("id", 0))
        bbox = track_dict.get("bbox", [0.0, 0.0, 0.0, 0.0])
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()

        item: Dict[str, Any] = {
            "track_id": int(tid),
            "session_id": str(track_dict.get("session_id", f"session:{tid}")),
            "bbox": [round(float(c), 2) for c in bbox],
            "confidence": round(float(track_dict.get("confidence", 1.0)), 4),
            "state": str(track_dict.get("state", "active")),
            "confirmed": bool(track_dict.get("confirmed", True)),
            "tracking_kind": track_dict.get("tracking_kind", "person"),
            "duration": track_dict.get("duration", 0),
            "hits": track_dict.get("hits", 0),
            "face": None,
            "face_bbox": None,
            "pose_category": None,
            "quality_score": None,
        }

        if face is not None:
            face_dict = face.to_dict() if hasattr(face, "to_dict") else dict(face)
            f_bbox = face_dict.get("bbox", [0.0, 0.0, 0.0, 0.0])
            if isinstance(f_bbox, np.ndarray):
                f_bbox = f_bbox.tolist()

            pose = face_dict.get("pose") or {}
            if hasattr(pose, "to_dict"):
                pose = pose.to_dict()
            pose_cat = pose.get("category", face_dict.get("pose_category", "uncertain"))

            quality = face_dict.get("quality") or {}
            if hasattr(quality, "to_dict"):
                quality = quality.to_dict()
            qual_score = quality.get("score", face_dict.get("quality_score", 0.0))

            face_obj = {
                "observation_id": str(face_dict.get("observation_id", f"obs_{tid}")),
                "bbox": [round(float(c), 2) for c in f_bbox],
                "confidence": round(float(face_dict.get("confidence", 1.0)), 4),
                "landmarks": face_dict.get("landmarks", []),
                "pose_category": str(pose_cat),
                "yaw": round(float(pose.get("yaw", face_dict.get("yaw", 0.0))), 1),
                "pitch": round(float(pose.get("pitch", face_dict.get("pitch", 0.0))), 1),
                "quality_score": round(float(qual_score), 1),
                "pose": pose,
                "quality": quality,
            }
            item["face"] = face_obj
            item["face_bbox"] = face_obj["bbox"]
            item["pose_category"] = face_obj["pose_category"]
            item["quality_score"] = face_obj["quality_score"]

        return item

    def _format_person_dict(self, p: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize an existing person dict into the standardized frame schema."""
        tid = p.get("track_id", p.get("id", 0))
        bbox = p.get("bbox", [0.0, 0.0, 0.0, 0.0])
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()

        face = p.get("face")
        face_bbox = p.get("face_bbox")
        pose_category = p.get("pose_category")
        quality_score = p.get("quality_score")

        if face is not None:
            if hasattr(face, "to_dict"):
                face = face.to_dict()
            if isinstance(face, dict):
                if face_bbox is None:
                    face_bbox = face.get("bbox")
                if pose_category is None:
                    pose = face.get("pose")
                    if isinstance(pose, dict):
                        pose_category = pose.get("category")
                    else:
                        pose_category = face.get("pose_category")
                if quality_score is None:
                    qual = face.get("quality")
                    if isinstance(qual, dict):
                        quality_score = qual.get("score")
                    else:
                        quality_score = face.get("quality_score")
        elif face_bbox is not None:
            face = {
                "bbox": face_bbox,
                "pose_category": pose_category or "uncertain",
                "quality_score": quality_score or 0.0,
            }

        return {
            "track_id": int(tid),
            "session_id": str(p.get("session_id", f"session:{tid}")),
            "bbox": [round(float(c), 2) for c in bbox],
            "confidence": round(float(p.get("confidence", 1.0)), 4),
            "state": str(p.get("state", "active")),
            "face": face,
            "face_bbox": face_bbox,
            "pose_category": pose_category,
            "quality_score": round(float(quality_score), 1) if quality_score is not None else None,
        }

    def emit_capture(
        self,
        capture_or_data: Optional[Any] = None,
        *,
        session_id: Optional[str] = None,
        capture_id: Optional[str] = None,
        category: Optional[str] = None,
        quality_score: Optional[float] = None,
        thumbnail_url: Optional[str] = None,
        crop_url: Optional[str] = None,
        track_id: Optional[int] = None,
        pose_category: Optional[str] = None,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        timestamp: Optional[float] = None,
        is_enhanced: Optional[bool] = None,
        enhanced_url: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        """Broadcast notification when a new best capture is saved (people:capture).

        Safe no-op when no clients are connected.

        Required fields in broadcast payload:
            - session_id: Profile session identifier.
            - capture_id: Unique capture ID.
            - category: Capture category (frontal, left_angle, etc.).
            - quality_score: Assessed quality score (0-100).
            - thumbnail_url / crop_url: Relative URL endpoint to fetch image asset.
        """
        if not self.has_clients and not force:
            return False

        payload: Dict[str, Any] = {}
        if isinstance(capture_or_data, dict):
            payload.update(capture_or_data)
        elif capture_or_data is not None and hasattr(capture_or_data, "to_dict"):
            payload.update(capture_or_data.to_dict())

        # Override with explicit keyword arguments if provided
        if session_id is not None:
            payload["session_id"] = str(session_id)
        if capture_id is not None:
            payload["capture_id"] = str(capture_id)
        if category is not None:
            payload["category"] = str(category)
        if quality_score is not None:
            payload["quality_score"] = float(quality_score)
        if thumbnail_url is not None:
            payload["thumbnail_url"] = str(thumbnail_url)
        if crop_url is not None:
            payload["crop_url"] = str(crop_url)
        if track_id is not None:
            payload["track_id"] = int(track_id)
        if pose_category is not None:
            payload["pose_category"] = str(pose_category)
        if yaw is not None:
            payload["yaw"] = float(yaw)
        if pitch is not None:
            payload["pitch"] = float(pitch)
        if timestamp is not None:
            payload["timestamp"] = float(timestamp)
        if is_enhanced is not None:
            payload["is_enhanced"] = bool(is_enhanced)
        if enhanced_url is not None:
            payload["enhanced_url"] = str(enhanced_url)

        cid = payload.get("capture_id", "cap_unknown")
        # Synthesize URLs if not present
        if "crop_url" not in payload and "original_url" in payload:
            payload["crop_url"] = payload["original_url"]
        elif "crop_url" not in payload:
            payload["crop_url"] = f"/api/people/captures/{cid}/image?type=original"

        if "thumbnail_url" not in payload:
            payload["thumbnail_url"] = payload.get("crop_url", f"/api/people/captures/{cid}/image?type=original")

        if "session_id" not in payload:
            payload["session_id"] = f"session:{payload.get('track_id', 0)}"
        if "category" not in payload:
            payload["category"] = "overall"
        if "quality_score" not in payload:
            payload["quality_score"] = 0.0

        payload["quality_score"] = round(float(payload["quality_score"]), 1)

        clean = sanitize_metadata(payload)
        with self._lock:
            self._emitted_captures += 1
        return self._emit(EVENT_PEOPLE_CAPTURE, clean)

    def emit_profile_update(
        self,
        profile_or_data: Optional[Any] = None,
        *,
        session_id: Optional[str] = None,
        track_id: Optional[int] = None,
        status: Optional[str] = None,
        duration: Optional[float] = None,
        best_quality_score: Optional[float] = None,
        total_observations: Optional[int] = None,
        best_capture: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
        force: bool = False,
    ) -> bool:
        """Broadcast profile metadata or state updates (people:profile_update).

        Safe no-op when no clients are connected.
        """
        if not self.has_clients and not force:
            return False

        payload: Dict[str, Any] = {}
        if isinstance(profile_or_data, dict):
            payload.update(profile_or_data)
        elif profile_or_data is not None and hasattr(profile_or_data, "to_dict"):
            payload.update(profile_or_data.to_dict())

        if session_id is not None:
            payload["session_id"] = str(session_id)
        if track_id is not None:
            payload["track_id"] = int(track_id)
        if status is not None:
            payload["status"] = str(status)
        if duration is not None:
            payload["duration"] = round(float(duration), 2)
        if best_quality_score is not None:
            payload["best_quality_score"] = round(float(best_quality_score), 1)
        if total_observations is not None:
            payload["total_observations"] = int(total_observations)
        if best_capture is not None:
            payload["best_capture"] = (
                best_capture.to_dict() if hasattr(best_capture, "to_dict") else dict(best_capture)
            )
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        if timestamp is not None:
            payload["timestamp"] = float(timestamp)
        elif "timestamp" not in payload:
            payload["timestamp"] = round(time.time(), 3)

        clean = sanitize_metadata(payload)
        with self._lock:
            self._emitted_profiles += 1
        return self._emit(EVENT_PEOPLE_PROFILE_UPDATE, clean)

    def emit_enhancement_ready(
        self,
        capture_or_data: Optional[Any] = None,
        *,
        capture_id: Optional[str] = None,
        session_id: Optional[str] = None,
        enhanced_url: Optional[str] = None,
        enhancement_method: Optional[str] = None,
        quality_score: Optional[float] = None,
        track_id: Optional[int] = None,
        timestamp: Optional[float] = None,
        force: bool = False,
    ) -> bool:
        """Broadcast notification when an enhanced capture is complete (people:enhancement_ready).

        Safe no-op when no clients are connected.
        """
        if not self.has_clients and not force:
            return False

        payload: Dict[str, Any] = {}
        if isinstance(capture_or_data, dict):
            payload.update(capture_or_data)
        elif capture_or_data is not None and hasattr(capture_or_data, "to_dict"):
            payload.update(capture_or_data.to_dict())

        if capture_id is not None:
            payload["capture_id"] = str(capture_id)
        if session_id is not None:
            payload["session_id"] = str(session_id)
        if enhanced_url is not None:
            payload["enhanced_url"] = str(enhanced_url)
        if enhancement_method is not None:
            payload["enhancement_method"] = str(enhancement_method)
        if quality_score is not None:
            payload["quality_score"] = round(float(quality_score), 1)
        if track_id is not None:
            payload["track_id"] = int(track_id)
        if timestamp is not None:
            payload["timestamp"] = float(timestamp)
        elif "timestamp" not in payload:
            payload["timestamp"] = round(time.time(), 3)

        cid = payload.get("capture_id", "cap_unknown")
        if "enhanced_url" not in payload:
            payload["enhanced_url"] = f"/api/people/captures/{cid}/image?type=enhanced"

        clean = sanitize_metadata(payload)
        with self._lock:
            self._emitted_enhancements += 1
        return self._emit(EVENT_PEOPLE_ENHANCEMENT_READY, clean)

    def emit_state(
        self,
        state_or_data: Optional[Any] = None,
        *,
        is_running: Optional[bool] = None,
        active_tracks: Optional[int] = None,
        active_tracks_count: Optional[int] = None,
        total_profiles: Optional[int] = None,
        total_profiles_count: Optional[int] = None,
        total_captures: Optional[int] = None,
        total_captures_count: Optional[int] = None,
        fps: Optional[float] = None,
        status_message: Optional[str] = None,
        error: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        """Broadcast system state synchronization (people:state).

        Supports periodic (~1 Hz heartbeat) or on-change state sync.
        Safe no-op when no clients are connected.

        Payload schema:
            is_running, active_tracks, total_profiles, total_captures, fps, status_message
        """
        if not self.has_clients and not force:
            return False

        payload: Dict[str, Any] = {}
        if isinstance(state_or_data, dict):
            payload.update(state_or_data)
        elif state_or_data is not None and hasattr(state_or_data, "to_dict"):
            payload.update(state_or_data.to_dict())

        # Resolve field aliases
        if is_running is not None:
            payload["is_running"] = bool(is_running)
        elif "is_running" not in payload:
            payload["is_running"] = False

        # Active tracks aliases
        act = (
            active_tracks
            if active_tracks is not None
            else active_tracks_count
            if active_tracks_count is not None
            else payload.get("active_tracks", payload.get("active_tracks_count", 0))
        )
        payload["active_tracks"] = int(act)
        payload["active_tracks_count"] = int(act)

        # Total profiles aliases
        tp = (
            total_profiles
            if total_profiles is not None
            else total_profiles_count
            if total_profiles_count is not None
            else payload.get("total_profiles", payload.get("total_profiles_count", 0))
        )
        payload["total_profiles"] = int(tp)
        payload["total_profiles_count"] = int(tp)

        # Total captures aliases
        tc = (
            total_captures
            if total_captures is not None
            else total_captures_count
            if total_captures_count is not None
            else payload.get("total_captures", payload.get("total_captures_count", 0))
        )
        payload["total_captures"] = int(tc)
        payload["total_captures_count"] = int(tc)

        # FPS
        f_val = fps if fps is not None else payload.get("fps", 0.0)
        payload["fps"] = round(float(f_val), 1)

        # Status & Error
        if status_message is not None:
            payload["status_message"] = str(status_message)
        elif "status_message" not in payload:
            payload["status_message"] = "Idle"

        if error is not None:
            payload["error"] = str(error)
        elif "error" not in payload:
            payload["error"] = None

        if "timestamp" not in payload:
            payload["timestamp"] = round(time.time(), 3)

        # State summary tuple for change detection
        state_summary = (
            payload["is_running"],
            payload["active_tracks"],
            payload["total_profiles"],
            payload["total_captures"],
            payload["status_message"],
            payload["error"],
        )

        now = self._time_fn()
        with self._lock:
            state_changed = state_summary != self._last_state_summary
            time_elapsed = now - self._last_state_emit_time
            periodic_due = time_elapsed >= self.state_interval_s

            if not force and not state_changed and not periodic_due:
                self._throttled_states += 1
                return False

            self._last_state_emit_time = now
            self._last_state_summary = state_summary
            self._emitted_states += 1

        clean = sanitize_metadata(payload)
        return self._emit(EVENT_PEOPLE_STATE, clean)

    # -------------------------------------------------------------------------
    # Diagnostics & Telemetry
    # -------------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational telemetry metrics."""
        with self._lock:
            return {
                "clients": self.client_count,
                "emitted_frames": self._emitted_frames,
                "throttled_frames": self._throttled_frames,
                "emitted_captures": self._emitted_captures,
                "emitted_profiles": self._emitted_profiles,
                "emitted_enhancements": self._emitted_enhancements,
                "emitted_states": self._emitted_states,
                "throttled_states": self._throttled_states,
                "emit_errors": self._emit_errors,
                "max_fps": self._max_fps,
                "state_interval_s": self.state_interval_s,
            }

    def reset_metrics(self) -> None:
        """Reset internal telemetry counters."""
        with self._lock:
            self._emitted_frames = 0
            self._throttled_frames = 0
            self._emitted_captures = 0
            self._emitted_profiles = 0
            self._emitted_enhancements = 0
            self._emitted_states = 0
            self._throttled_states = 0
            self._emit_errors = 0
