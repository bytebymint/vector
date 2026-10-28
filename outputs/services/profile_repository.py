"""Anonymous session profile and facial capture persistence system.

SQLite-backed thread-safe repository with atomic disk image persistence,
multi-angle quality indexing, robust traversal security, and retention cleanup.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from .face_contracts import CaptureCandidate, CaptureRecord, PersonProfile
except (ImportError, ValueError):
    from services.face_contracts import CaptureCandidate, CaptureRecord, PersonProfile

LOG = logging.getLogger(__name__)

# Default paths relative to workspace outputs
DEFAULT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = DEFAULT_ROOT / "data" / "traffic.sqlite3"
DEFAULT_STORAGE_DIR = DEFAULT_ROOT / "data" / "captures"

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=10000;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS people_profiles (
    session_id TEXT PRIMARY KEY,
    track_id INTEGER NOT NULL,
    camera_id TEXT NOT NULL,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    duration REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'active',
    best_capture_id TEXT,
    total_observations INTEGER NOT NULL DEFAULT 0,
    best_quality_score REAL NOT NULL DEFAULT 0.0,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS people_captures (
    capture_id TEXT PRIMARY KEY,
    track_id INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    category TEXT NOT NULL,
    quality_score REAL NOT NULL,
    pose_category TEXT NOT NULL,
    yaw REAL NOT NULL DEFAULT 0.0,
    pitch REAL NOT NULL DEFAULT 0.0,
    original_path TEXT NOT NULL,
    crop_width INTEGER NOT NULL DEFAULT 0,
    crop_height INTEGER NOT NULL DEFAULT 0,
    enhanced_path TEXT,
    is_enhanced INTEGER NOT NULL DEFAULT 0,
    enhancement_method TEXT,
    file_size_bytes INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    FOREIGN KEY(session_id) REFERENCES people_profiles(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_people_profiles_last_seen ON people_profiles(last_seen);
CREATE INDEX IF NOT EXISTS idx_people_profiles_first_seen ON people_profiles(first_seen);
CREATE INDEX IF NOT EXISTS idx_people_profiles_status ON people_profiles(status);
CREATE INDEX IF NOT EXISTS idx_people_profiles_camera_id ON people_profiles(camera_id);
CREATE INDEX IF NOT EXISTS idx_people_profiles_duration ON people_profiles(duration);
CREATE INDEX IF NOT EXISTS idx_people_profiles_quality ON people_profiles(best_quality_score);

CREATE INDEX IF NOT EXISTS idx_people_captures_session_id ON people_captures(session_id);
CREATE INDEX IF NOT EXISTS idx_people_captures_category ON people_captures(category);
CREATE INDEX IF NOT EXISTS idx_people_captures_quality ON people_captures(quality_score);
CREATE INDEX IF NOT EXISTS idx_people_captures_timestamp ON people_captures(timestamp);
"""


class ProfileRepository:
    """Thread-safe SQLite and filesystem repository for PersonProfiles and facial captures."""

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        storage_dir: Optional[Union[str, Path]] = None,
    ):
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.storage_dir = (
            Path(storage_dir) if storage_dir is not None else DEFAULT_STORAGE_DIR
        )

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._all_conns: List[sqlite3.Connection] = []
        self._all_conns_lock = threading.Lock()
        self._closed = False

        self._init_db()

    # -------------------------------------------------------------------------
    # Connection Management & Lifecycle
    # -------------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("ProfileRepository has been closed")
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=10000;")
            conn.execute("PRAGMA foreign_keys=ON;")
            self._local.conn = conn
            with self._all_conns_lock:
                self._all_conns.append(conn)
        return conn

    @contextmanager
    def connection(self):
        """Context manager providing thread-local SQLite connection for queries."""
        conn = self._get_connection()
        yield conn

    @contextmanager
    def transaction(self):
        """Context manager providing thread-safe, lock-protected write transaction."""
        with self._write_lock:
            conn = self._get_connection()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _init_db(self):
        with self.transaction() as conn:
            conn.executescript(SCHEMA_SQL)

    def close(self):
        """Close all connections held by the repository."""
        with self._write_lock:
            if not self._closed:
                self._closed = True
                with self._all_conns_lock:
                    for conn in self._all_conns:
                        try:
                            conn.close()
                        except Exception:
                            pass
                    self._all_conns.clear()

    # -------------------------------------------------------------------------
    # Session ID & Security Validation Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def format_session_id(
        camera_id: str, session_epoch: Union[str, int, float], track_id: int
    ) -> str:
        """Create a standard anonymous session ID: `{camera_id}:{session_epoch}:{track_id}`."""
        return f"{camera_id}:{session_epoch}:{track_id}"

    @staticmethod
    def parse_session_id(session_id: str) -> Tuple[str, str, int]:
        """Parse anonymous session ID into (camera_id, session_epoch, track_id)."""
        parts = str(session_id).split(":")
        if len(parts) >= 3:
            cam_id = parts[0]
            epoch = parts[1]
            try:
                track_id = int(parts[2])
            except ValueError:
                track_id = 0
            return cam_id, epoch, track_id
        return "unknown", "unknown", 0

    def sanitize_session_id(self, session_id: str) -> str:
        """Sanitize session ID for filesystem use, removing colons, slashes, and traversal characters."""
        clean = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", str(session_id))
        clean = re.sub(r"^\.+", "", clean)
        clean = clean.replace("..", "_")
        clean = clean.strip(". _-")
        return clean or "session_default"

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent directory traversal and filesystem-incompatible characters."""
        clean = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", str(filename))
        clean = re.sub(r"^\.+", "", clean)
        clean = clean.replace("..", "_")
        clean = clean.strip(". _-")
        return clean or "capture_default.jpg"

    def get_session_dir(self, session_id: str) -> Path:
        """Get validated filesystem directory for a session under storage_dir."""
        safe_session = self.sanitize_session_id(session_id)
        session_dir = (self.storage_dir / safe_session).resolve()
        base_dir = self.storage_dir.resolve()
        try:
            session_dir.relative_to(base_dir)
        except ValueError:
            raise ValueError(
                f"Directory traversal detected for session ID: {session_id}"
            )
        return session_dir

    def validate_safe_path(self, target_path: Union[str, Path]) -> Path:
        """Validate that a target path resides strictly inside storage_dir."""
        target = Path(target_path).resolve()
        base = self.storage_dir.resolve()
        try:
            target.relative_to(base)
        except ValueError:
            raise ValueError(f"Path outside allowed storage directory: {target_path}")
        return target

    # -------------------------------------------------------------------------
    # Disk Image Persistence (Atomic Writes)
    # -------------------------------------------------------------------------

    def write_image_atomic(
        self,
        session_id: str,
        filename: str,
        image_data: Union[np.ndarray, bytes, bytearray],
    ) -> Path:
        """Atomically write image data to the session capture directory."""
        session_dir = self.get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        safe_filename = self.sanitize_filename(filename)
        dest_path = (session_dir / safe_filename).resolve()
        try:
            dest_path.relative_to(session_dir)
        except ValueError:
            raise ValueError(f"Directory traversal detected for filename: {filename}")

        if isinstance(image_data, np.ndarray):
            if image_data.size == 0:
                raise ValueError("Cannot write empty image array")
            extension = ".png" if safe_filename.lower().endswith(".png") else ".jpg"
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 1] if extension == ".png" else [int(cv2.IMWRITE_JPEG_QUALITY), 97]
            success, buf = cv2.imencode(extension, image_data, params)
            if not success:
                raise ValueError(f"Failed to encode image to JPEG for {filename}")
            data_bytes = buf.tobytes()
        elif isinstance(image_data, (bytes, bytearray)):
            data_bytes = bytes(image_data)
            if len(data_bytes) == 0:
                raise ValueError("Cannot write empty bytes")
        else:
            raise TypeError(f"Unsupported image data type: {type(image_data)}")

        temp_file = session_dir / f".{safe_filename}.{uuid.uuid4().hex}.tmp"
        try:
            with open(temp_file, "wb") as f:
                f.write(data_bytes)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_file, dest_path)
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

        return dest_path

    # -------------------------------------------------------------------------
    # Entity Conversion Helpers
    # -------------------------------------------------------------------------

    def _row_to_capture(self, row: sqlite3.Row) -> CaptureRecord:
        return CaptureRecord(
            capture_id=str(row["capture_id"]),
            track_id=int(row["track_id"]),
            session_id=str(row["session_id"]),
            timestamp=float(row["timestamp"]),
            category=str(row["category"]),
            quality_score=float(row["quality_score"]),
            pose_category=str(row["pose_category"]),
            yaw=float(row["yaw"]),
            pitch=float(row["pitch"]),
            original_path=str(row["original_path"]),
            crop_width=int(row["crop_width"]),
            crop_height=int(row["crop_height"]),
            enhanced_path=(
                str(row["enhanced_path"])
                if row["enhanced_path"] is not None
                else None
            ),
            is_enhanced=bool(row["is_enhanced"]),
            enhancement_method=(
                str(row["enhancement_method"])
                if row["enhancement_method"] is not None
                else None
            ),
            file_size_bytes=int(row["file_size_bytes"]),
            created_at=float(row["created_at"]) if row["created_at"] else 0.0,
        )

    def _save_capture_in_conn(
        self, conn: sqlite3.Connection, capture: CaptureRecord
    ):
        conn.execute(
            """INSERT INTO people_captures (
                capture_id, track_id, session_id, timestamp, category, quality_score,
                pose_category, yaw, pitch, original_path, crop_width, crop_height,
                enhanced_path, is_enhanced, enhancement_method, file_size_bytes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(capture_id) DO UPDATE SET
                quality_score=excluded.quality_score,
                original_path=excluded.original_path,
                crop_width=excluded.crop_width,
                crop_height=excluded.crop_height,
                enhanced_path=excluded.enhanced_path,
                is_enhanced=excluded.is_enhanced,
                enhancement_method=excluded.enhancement_method,
                file_size_bytes=excluded.file_size_bytes""",
            (
                capture.capture_id,
                capture.track_id,
                capture.session_id,
                capture.timestamp,
                capture.category,
                capture.quality_score,
                capture.pose_category,
                capture.yaw,
                capture.pitch,
                capture.original_path,
                capture.crop_width,
                capture.crop_height,
                capture.enhanced_path,
                int(capture.is_enhanced),
                capture.enhancement_method,
                capture.file_size_bytes,
                capture.created_at or time.time(),
            ),
        )

    # -------------------------------------------------------------------------
    # Profile CRUD & Updates
    # -------------------------------------------------------------------------

    def save_profile(self, profile: PersonProfile) -> PersonProfile:
        """Create or update a PersonProfile and ensure associated captures are saved."""
        now = time.time()
        best_cap_id = (
            profile.best_capture.capture_id if profile.best_capture else None
        )

        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO people_profiles (
                    session_id, track_id, camera_id, first_seen, last_seen, duration,
                    status, best_capture_id, total_observations, best_quality_score,
                    metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    track_id=excluded.track_id,
                    camera_id=excluded.camera_id,
                    first_seen=excluded.first_seen,
                    last_seen=excluded.last_seen,
                    duration=excluded.duration,
                    status=excluded.status,
                    best_capture_id=COALESCE(excluded.best_capture_id, people_profiles.best_capture_id),
                    total_observations=excluded.total_observations,
                    best_quality_score=MAX(excluded.best_quality_score, people_profiles.best_quality_score),
                    metadata=excluded.metadata,
                    updated_at=excluded.updated_at""",
                (
                    profile.session_id,
                    profile.track_id,
                    profile.camera_id,
                    profile.first_seen,
                    profile.last_seen,
                    profile.duration,
                    profile.status,
                    best_cap_id,
                    profile.total_observations,
                    profile.best_quality_score,
                    json.dumps(profile.metadata or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )

            # Persist best capture and multi-angle captures if present
            if profile.best_capture:
                self._save_capture_in_conn(conn, profile.best_capture)
            for cap in profile.captures_by_category.values():
                self._save_capture_in_conn(conn, cap)
            for cap in profile.capture_history:
                self._save_capture_in_conn(conn, cap)

        return profile

    def create_or_update_profile(self, profile: PersonProfile) -> PersonProfile:
        """Alias for save_profile."""
        return self.save_profile(profile)

    def get_or_create_profile(
        self,
        session_id: str,
        track_id: int,
        camera_id: str,
        timestamp: Optional[float] = None,
        status: str = "active",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PersonProfile:
        """Fetch existing profile or create a new initialized profile."""
        existing = self.get_profile(session_id)
        if existing is not None:
            return existing

        ts = timestamp if timestamp is not None else time.time()
        new_profile = PersonProfile(
            session_id=session_id,
            track_id=track_id,
            camera_id=camera_id,
            first_seen=ts,
            last_seen=ts,
            duration=0.0,
            status=status,
            metadata=metadata or {},
        )
        return self.save_profile(new_profile)

    def update_profile_status(
        self,
        session_id: str,
        status: str,
        last_seen: Optional[float] = None,
    ) -> bool:
        """Update profile status (active/lost/completed) and duration."""
        now = time.time()
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT first_seen, last_seen FROM people_profiles WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return False

            first_seen = float(row["first_seen"])
            current_last_seen = float(row["last_seen"])
            new_last_seen = (
                max(current_last_seen, float(last_seen))
                if last_seen is not None
                else current_last_seen
            )
            duration = max(0.0, new_last_seen - first_seen)

            conn.execute(
                """UPDATE people_profiles SET
                    status = ?,
                    last_seen = ?,
                    duration = ?,
                    updated_at = ?
                WHERE session_id = ?""",
                (status, new_last_seen, duration, now, session_id),
            )
        return True

    def get_profile(self, session_id: str) -> Optional[PersonProfile]:
        """Fetch a single PersonProfile with all associated captures and angle breakdown."""
        with self.connection() as conn:
            p_row = conn.execute(
                "SELECT * FROM people_profiles WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if p_row is None:
                return None

            c_rows = conn.execute(
                "SELECT * FROM people_captures WHERE session_id = ? ORDER BY timestamp ASC, created_at ASC",
                (session_id,),
            ).fetchall()

        captures = [self._row_to_capture(r) for r in c_rows]
        captures_by_category: Dict[str, CaptureRecord] = {}
        best_capture: Optional[CaptureRecord] = None
        best_cap_id = p_row["best_capture_id"]

        for cap in captures:
            # Multi-angle grouping: maintain highest quality capture for each angle/category
            cat = cap.category
            if (
                cat not in captures_by_category
                or cap.quality_score > captures_by_category[cat].quality_score
            ):
                captures_by_category[cat] = cap

            if best_cap_id and cap.capture_id == best_cap_id:
                best_capture = cap

        if best_capture is None and captures:
            best_capture = max(captures, key=lambda c: c.quality_score)

        return PersonProfile(
            session_id=str(p_row["session_id"]),
            track_id=int(p_row["track_id"]),
            camera_id=str(p_row["camera_id"]),
            first_seen=float(p_row["first_seen"]),
            last_seen=float(p_row["last_seen"]),
            duration=float(p_row["duration"]),
            status=str(p_row["status"]),
            best_capture=best_capture,
            captures_by_category=captures_by_category,
            capture_history=captures,
            total_observations=int(p_row["total_observations"]),
            best_quality_score=float(p_row["best_quality_score"]),
            metadata=json.loads(p_row["metadata"]) if p_row["metadata"] else {},
        )

    # -------------------------------------------------------------------------
    # Capture Persistence & Attachment
    # -------------------------------------------------------------------------

    def save_capture(
        self,
        capture: CaptureRecord,
        original_image: Optional[Union[np.ndarray, bytes]] = None,
        enhanced_image: Optional[Union[np.ndarray, bytes]] = None,
    ) -> CaptureRecord:
        """Persist facial capture to disk and SQLite, auto-updating associated profile."""
        now = time.time()
        if capture.created_at <= 0.0:
            capture.created_at = now

        # Write original crop
        if original_image is not None:
            orig_filename = f"{capture.capture_id}_orig.png"
            orig_path = self.write_image_atomic(
                capture.session_id, orig_filename, original_image
            )
            capture.original_path = str(orig_path.as_posix())
            capture.file_size_bytes = os.path.getsize(orig_path)
            if capture.crop_width == 0 or capture.crop_height == 0:
                if isinstance(original_image, np.ndarray):
                    capture.crop_height, capture.crop_width = (
                        original_image.shape[:2]
                    )
                else:
                    arr = cv2.imdecode(
                        np.frombuffer(original_image, np.uint8),
                        cv2.IMREAD_UNCHANGED,
                    )
                    if arr is not None:
                        capture.crop_height, capture.crop_width = arr.shape[:2]

        # Write enhanced crop if provided
        if enhanced_image is not None:
            enh_filename = f"{capture.capture_id}_enhanced.jpg"
            enh_path = self.write_image_atomic(
                capture.session_id, enh_filename, enhanced_image
            )
            capture.enhanced_path = str(enh_path.as_posix())
            capture.is_enhanced = True

        with self.transaction() as conn:
            p_row = conn.execute(
                "SELECT session_id, first_seen, last_seen, best_quality_score, best_capture_id, total_observations FROM people_profiles WHERE session_id = ?",
                (capture.session_id,),
            ).fetchone()

            if p_row is None:
                cam_id, _, _ = self.parse_session_id(capture.session_id)
                conn.execute(
                    """INSERT INTO people_profiles (
                        session_id, track_id, camera_id, first_seen, last_seen, duration,
                        status, best_capture_id, total_observations, best_quality_score,
                        metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        capture.session_id,
                        capture.track_id,
                        cam_id,
                        capture.timestamp,
                        capture.timestamp,
                        0.0,
                        "active",
                        capture.capture_id,
                        1,
                        capture.quality_score,
                        "{}",
                        now,
                        now,
                    ),
                )
            else:
                first_seen = float(p_row["first_seen"])
                last_seen = max(
                    float(p_row["last_seen"]), float(capture.timestamp)
                )
                duration = max(0.0, last_seen - first_seen)
                best_quality = float(p_row["best_quality_score"])
                best_capture_id = p_row["best_capture_id"]
                total_obs = int(p_row["total_observations"]) + 1

                if (
                    capture.quality_score > best_quality
                    or best_capture_id is None
                ):
                    best_quality = capture.quality_score
                    best_capture_id = capture.capture_id

                conn.execute(
                    """UPDATE people_profiles SET
                        last_seen = ?,
                        duration = ?,
                        total_observations = ?,
                        best_quality_score = ?,
                        best_capture_id = ?,
                        updated_at = ?
                    WHERE session_id = ?""",
                    (
                        last_seen,
                        duration,
                        total_obs,
                        best_quality,
                        best_capture_id,
                        now,
                        capture.session_id,
                    ),
                )

            self._save_capture_in_conn(conn, capture)

        return capture

    def save_capture_record(
        self,
        capture: CaptureRecord,
        original_crop: Optional[Union[np.ndarray, bytes]] = None,
        enhanced_crop: Optional[Union[np.ndarray, bytes]] = None,
    ) -> CaptureRecord:
        """Alias for save_capture."""
        return self.save_capture(
            capture,
            original_image=original_crop,
            enhanced_image=enhanced_crop,
        )

    def save_candidate(
        self,
        session_id: str,
        candidate: CaptureCandidate,
        enhanced_crop: Optional[Union[np.ndarray, bytes]] = None,
        enhancement_method: Optional[str] = None,
    ) -> CaptureRecord:
        """Persist a candidate face frame as a permanent CaptureRecord."""
        h, w = (
            candidate.crop_original.shape[:2]
            if isinstance(candidate.crop_original, np.ndarray)
            else (0, 0)
        )
        record = CaptureRecord(
            capture_id=candidate.candidate_id,
            track_id=candidate.track_id,
            session_id=session_id,
            timestamp=candidate.timestamp,
            category=candidate.category,
            quality_score=candidate.quality_score,
            pose_category=candidate.pose.category,
            yaw=candidate.pose.yaw,
            pitch=candidate.pose.pitch,
            original_path="",
            crop_width=w,
            crop_height=h,
            enhancement_method=enhancement_method,
        )
        saved = self.save_capture(
            record,
            original_image=candidate.crop_original,
            enhanced_image=enhanced_crop,
        )
        if candidate.verification_score >= .90 and candidate.verification_method == 'yunet+res10ssd-v1':
            with self.transaction() as conn:
                row=conn.execute('SELECT metadata FROM people_profiles WHERE session_id=?',(session_id,)).fetchone()
                metadata=json.loads(row['metadata'] or '{}')
                metadata.update(face_validation_version=2,verification_method=candidate.verification_method,
                                verification_confidence=candidate.verification_score)
                conn.execute('UPDATE people_profiles SET metadata=? WHERE session_id=?',
                             (json.dumps(metadata),session_id))
        return saved

    def attach_enhanced_image(
        self,
        capture_id: str,
        enhanced_crop: Union[np.ndarray, bytes],
        enhancement_method: Optional[str] = "real-esrgan",
    ) -> Optional[CaptureRecord]:
        """Attach an enhanced image crop to an existing CaptureRecord."""
        capture = self.get_capture(capture_id)
        if capture is None:
            return None

        filename = f"{capture_id}_enhanced.jpg"
        saved_path = self.write_image_atomic(
            capture.session_id, filename, enhanced_crop
        )
        capture.enhanced_path = str(saved_path.as_posix())
        capture.is_enhanced = True
        capture.enhancement_method = enhancement_method

        with self.transaction() as conn:
            conn.execute(
                """UPDATE people_captures SET
                    enhanced_path = ?,
                    is_enhanced = 1,
                    enhancement_method = ?
                WHERE capture_id = ?""",
                (capture.enhanced_path, capture.enhancement_method, capture_id),
            )
        return capture

    def get_capture(self, capture_id: str) -> Optional[CaptureRecord]:
        """Retrieve a CaptureRecord by its unique ID."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM people_captures WHERE capture_id = ?",
                (capture_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_capture(row)

    def get_capture_image_path(
        self, capture_id: str, image_type: str = "original"
    ) -> Optional[Path]:
        """Resolve and validate the filesystem path to a capture's image."""
        capture = self.get_capture(capture_id)
        if capture is None:
            return None

        path_str = (
            capture.enhanced_path
            if image_type == "enhanced"
            else capture.original_path
        )
        if not path_str:
            return None

        raw_path = Path(path_str)
        if not raw_path.is_absolute():
            candidate = (self.storage_dir / raw_path).resolve()
            if not candidate.exists():
                candidate = (DEFAULT_ROOT / raw_path).resolve()
        else:
            candidate = raw_path.resolve()

        try:
            validated = self.validate_safe_path(candidate)
            if validated.is_file():
                return validated
        except ValueError:
            return None
        return None

    def get_capture_image_bytes(
        self, capture_id: str, image_type: str = "original"
    ) -> Optional[bytes]:
        """Read and return image bytes for a capture."""
        path = self.get_capture_image_path(capture_id, image_type=image_type)
        if path is not None and path.is_file():
            return path.read_bytes()
        return None

    def get_capture_image_array(
        self, capture_id: str, image_type: str = "original"
    ) -> Optional[np.ndarray]:
        """Read and decode an image crop as a BGR numpy array."""
        data = self.get_capture_image_bytes(capture_id, image_type=image_type)
        if data is None:
            return None
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    # -------------------------------------------------------------------------
    # Indexing, Filtering, Sorting & Pagination
    # -------------------------------------------------------------------------

    def list_profiles(
        self,
        page: int = 1,
        page_size: int = 50,
        search: Optional[str] = None,
        sort_by: str = "recent",
        order: str = "desc",
        status: Optional[str] = None,
        camera_id: Optional[str] = None,
        min_quality: Optional[float] = None,
        include_captures: bool = False,
    ) -> List[PersonProfile]:
        """List profiles with flexible search, filtering, sorting, and pagination."""
        sort_map = {
            "recent": "last_seen",
            "last_seen": "last_seen",
            "first_seen": "first_seen",
            "duration": "duration",
            "quality": "best_quality_score",
            "quality_score": "best_quality_score",
            "best_quality_score": "best_quality_score",
            "observations": "total_observations",
            "total_observations": "total_observations",
            "created_at": "created_at",
        }
        col = sort_map.get(sort_by.lower(), "last_seen")
        direction = "ASC" if str(order).upper() == "ASC" else "DESC"

        clauses: List[str] = []
        params: List[Any] = []

        if search:
            search_str = f"%{search.strip()}%"
            try:
                search_int = int(search.strip())
                clauses.append(
                    "(session_id LIKE ? OR camera_id LIKE ? OR track_id = ?)"
                )
                params.extend([search_str, search_str, search_int])
            except ValueError:
                clauses.append("(session_id LIKE ? OR camera_id LIKE ?)")
                params.extend([search_str, search_str])

        if status:
            clauses.append("status = ?")
            params.append(status)

        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)

        if min_quality is not None:
            clauses.append("best_quality_score >= ?")
            params.append(float(min_quality))

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        page = max(1, int(page))
        page_size = max(1, int(page_size))
        offset = (page - 1) * page_size

        query = f"SELECT * FROM people_profiles {where_sql} ORDER BY {col} {direction} LIMIT ? OFFSET ?"
        exec_params = list(params)
        exec_params.extend([page_size, offset])

        with self.connection() as conn:
            p_rows = conn.execute(query, exec_params).fetchall()

            if not p_rows:
                return []

            # Batch load best captures
            best_cap_ids = [
                r["best_capture_id"]
                for r in p_rows
                if r["best_capture_id"] is not None
            ]
            cap_map: Dict[str, CaptureRecord] = {}
            if best_cap_ids:
                placeholders = ",".join("?" for _ in best_cap_ids)
                c_rows = conn.execute(
                    f"SELECT * FROM people_captures WHERE capture_id IN ({placeholders})",
                    best_cap_ids,
                ).fetchall()
                for cr in c_rows:
                    cap_rec = self._row_to_capture(cr)
                    cap_map[cap_rec.capture_id] = cap_rec

            # Optionally batch load complete capture histories
            full_captures_by_session: Dict[str, List[CaptureRecord]] = {}
            if include_captures:
                session_ids = [r["session_id"] for r in p_rows]
                s_placeholders = ",".join("?" for _ in session_ids)
                all_c_rows = conn.execute(
                    f"SELECT * FROM people_captures WHERE session_id IN ({s_placeholders}) ORDER BY timestamp ASC",
                    session_ids,
                ).fetchall()
                for acr in all_c_rows:
                    c_obj = self._row_to_capture(acr)
                    full_captures_by_session.setdefault(
                        c_obj.session_id, []
                    ).append(c_obj)

        profiles: List[PersonProfile] = []
        for r in p_rows:
            sid = str(r["session_id"])
            best_cap = cap_map.get(r["best_capture_id"])

            if include_captures:
                sess_caps = full_captures_by_session.get(sid, [])
                cat_map: Dict[str, CaptureRecord] = {}
                for c in sess_caps:
                    if (
                        c.category not in cat_map
                        or c.quality_score > cat_map[c.category].quality_score
                    ):
                        cat_map[c.category] = c
                if best_cap is None and sess_caps:
                    best_cap = max(sess_caps, key=lambda c: c.quality_score)
                profile = PersonProfile(
                    session_id=sid,
                    track_id=int(r["track_id"]),
                    camera_id=str(r["camera_id"]),
                    first_seen=float(r["first_seen"]),
                    last_seen=float(r["last_seen"]),
                    duration=float(r["duration"]),
                    status=str(r["status"]),
                    best_capture=best_cap,
                    captures_by_category=cat_map,
                    capture_history=sess_caps,
                    total_observations=int(r["total_observations"]),
                    best_quality_score=float(r["best_quality_score"]),
                    metadata=(
                        json.loads(r["metadata"]) if r["metadata"] else {}
                    ),
                )
            else:
                profile = PersonProfile(
                    session_id=sid,
                    track_id=int(r["track_id"]),
                    camera_id=str(r["camera_id"]),
                    first_seen=float(r["first_seen"]),
                    last_seen=float(r["last_seen"]),
                    duration=float(r["duration"]),
                    status=str(r["status"]),
                    best_capture=best_cap,
                    captures_by_category=(
                        {best_cap.category: best_cap} if best_cap else {}
                    ),
                    capture_history=[best_cap] if best_cap else [],
                    total_observations=int(r["total_observations"]),
                    best_quality_score=float(r["best_quality_score"]),
                    metadata=(
                        json.loads(r["metadata"]) if r["metadata"] else {}
                    ),
                )
            profiles.append(profile)

        return profiles

    def count_profiles(
        self,
        search: Optional[str] = None,
        status: Optional[str] = None,
        camera_id: Optional[str] = None,
        min_quality: Optional[float] = None,
    ) -> int:
        """Count profiles matching specified filters."""
        clauses: List[str] = []
        params: List[Any] = []

        if search:
            search_str = f"%{search.strip()}%"
            try:
                search_int = int(search.strip())
                clauses.append(
                    "(session_id LIKE ? OR camera_id LIKE ? OR track_id = ?)"
                )
                params.extend([search_str, search_str, search_int])
            except ValueError:
                clauses.append("(session_id LIKE ? OR camera_id LIKE ?)")
                params.extend([search_str, search_str])

        if status:
            clauses.append("status = ?")
            params.append(status)

        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)

        if min_quality is not None:
            clauses.append("best_quality_score >= ?")
            params.append(float(min_quality))

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT COUNT(*) FROM people_profiles {where_sql}"

        with self.connection() as conn:
            row = conn.execute(query, params).fetchone()
            return int(row[0]) if row else 0

    def get_profiles_paginated(
        self,
        page: int = 1,
        page_size: int = 20,
        search: Optional[str] = None,
        sort_by: str = "recent",
        order: str = "desc",
        status: Optional[str] = None,
        camera_id: Optional[str] = None,
        min_quality: Optional[float] = None,
        include_captures: bool = False,
    ) -> Dict[str, Any]:
        """Return standardized pagination response containing profiles and metadata."""
        total = self.count_profiles(
            search=search,
            status=status,
            camera_id=camera_id,
            min_quality=min_quality,
        )
        items = self.list_profiles(
            page=page,
            page_size=page_size,
            search=search,
            sort_by=sort_by,
            order=order,
            status=status,
            camera_id=camera_id,
            min_quality=min_quality,
            include_captures=include_captures,
        )
        total_pages = math.ceil(total / page_size) if page_size > 0 else 1
        return {
            "items": [item.to_dict() for item in items],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, total_pages),
        }

    def list_captures(
        self,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CaptureRecord]:
        """List capture records with optional session and category filters."""
        clauses: List[str] = []
        params: List[Any] = []

        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if category:
            clauses.append("category = ?")
            params.append(category)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM people_captures {where_sql} ORDER BY timestamp ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_capture(r) for r in rows]

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def count_captures(self) -> int:
        """Count rows without loading every capture and materializing its metadata."""
        with self.connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM people_captures").fetchone()[0])

    def get_statistics(self) -> Dict[str, Any]:
        """Compute aggregate system statistics across profiles and captures."""
        with self.connection() as conn:
            total_profiles = conn.execute(
                "SELECT COUNT(*) FROM people_profiles"
            ).fetchone()[0]
            active_tracks = conn.execute(
                "SELECT COUNT(*) FROM people_profiles WHERE status = 'active'"
            ).fetchone()[0]
            lost_tracks = conn.execute(
                "SELECT COUNT(*) FROM people_profiles WHERE status = 'lost'"
            ).fetchone()[0]
            completed_tracks = conn.execute(
                "SELECT COUNT(*) FROM people_profiles WHERE status = 'completed'"
            ).fetchone()[0]

            saved_captures = conn.execute(
                "SELECT COUNT(*) FROM people_captures"
            ).fetchone()[0]
            enhanced_captures = conn.execute(
                "SELECT COUNT(*) FROM people_captures WHERE is_enhanced = 1"
            ).fetchone()[0]

            avg_q_row = conn.execute(
                "SELECT AVG(quality_score) FROM people_captures"
            ).fetchone()[0]
            avg_quality = (
                round(float(avg_q_row), 1) if avg_q_row is not None else 0.0
            )

            avg_dur_row = conn.execute(
                "SELECT AVG(duration) FROM people_profiles"
            ).fetchone()[0]
            avg_duration = (
                round(float(avg_dur_row), 2) if avg_dur_row is not None else 0.0
            )

            obs_row = conn.execute(
                "SELECT SUM(total_observations) FROM people_profiles"
            ).fetchone()[0]
            total_obs = int(obs_row) if obs_row is not None else 0

            cat_rows = conn.execute(
                "SELECT category, COUNT(*) FROM people_captures GROUP BY category"
            ).fetchall()
            captures_by_cat = {r[0]: r[1] for r in cat_rows}

        return {
            "total_profiles": total_profiles,
            "active_tracks": active_tracks,
            "lost_tracks": lost_tracks,
            "completed_tracks": completed_tracks,
            "saved_captures": saved_captures,
            "enhanced_captures": enhanced_captures,
            "avg_quality_score": avg_quality,
            "avg_duration": avg_duration,
            "total_observations": total_obs,
            "captures_by_category": captures_by_cat,
        }

    # -------------------------------------------------------------------------
    # Cascade Deletion & Retention Cleanup
    # -------------------------------------------------------------------------

    def delete_profile(self, session_id: str) -> bool:
        """Cascade delete profile, removing database rows and disk image folders."""
        with self.transaction() as conn:
            p_row = conn.execute(
                "SELECT session_id FROM people_profiles WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if p_row is None:
                return False

            c_rows = conn.execute(
                "SELECT original_path, enhanced_path FROM people_captures WHERE session_id = ?",
                (session_id,),
            ).fetchall()

            conn.execute(
                "DELETE FROM people_captures WHERE session_id = ?",
                (session_id,),
            )
            conn.execute(
                "DELETE FROM people_profiles WHERE session_id = ?",
                (session_id,),
            )

        # Remove individual image files
        for r in c_rows:
            for field in ("original_path", "enhanced_path"):
                p_str = r[field]
                if p_str:
                    try:
                        p = Path(p_str)
                        if p.is_file():
                            p.unlink()
                    except Exception:
                        pass

        # Safely remove session folder
        try:
            session_dir = self.get_session_dir(session_id)
            if session_dir.exists() and session_dir.is_dir():
                shutil.rmtree(session_dir, ignore_errors=True)
        except Exception:
            pass

        return True

    def delete_capture(self, capture_id: str) -> bool:
        """Delete single capture, removing disk assets and adjusting profile best captures."""
        with self.transaction() as conn:
            c_row = conn.execute(
                "SELECT session_id, original_path, enhanced_path FROM people_captures WHERE capture_id = ?",
                (capture_id,),
            ).fetchone()
            if c_row is None:
                return False

            session_id = c_row["session_id"]
            orig_path = c_row["original_path"]
            enh_path = c_row["enhanced_path"]

            conn.execute(
                "DELETE FROM people_captures WHERE capture_id = ?",
                (capture_id,),
            )

            p_row = conn.execute(
                "SELECT best_capture_id FROM people_profiles WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if p_row and p_row["best_capture_id"] == capture_id:
                next_best = conn.execute(
                    "SELECT capture_id, quality_score FROM people_captures WHERE session_id = ? ORDER BY quality_score DESC LIMIT 1",
                    (session_id,),
                ).fetchone()
                if next_best:
                    conn.execute(
                        """UPDATE people_profiles SET
                            best_capture_id = ?,
                            best_quality_score = ?,
                            updated_at = ?
                        WHERE session_id = ?""",
                        (
                            next_best["capture_id"],
                            float(next_best["quality_score"]),
                            time.time(),
                            session_id,
                        ),
                    )
                else:
                    conn.execute(
                        """UPDATE people_profiles SET
                            best_capture_id = NULL,
                            best_quality_score = 0.0,
                            updated_at = ?
                        WHERE session_id = ?""",
                        (time.time(), session_id),
                    )

        if orig_path:
            try:
                p = Path(orig_path)
                if p.is_file():
                    p.unlink()
            except Exception:
                pass

        if enh_path:
            try:
                p = Path(enh_path)
                if p.is_file():
                    p.unlink()
            except Exception:
                pass

        return True

    def prune_profiles(
        self,
        older_than_seconds: Optional[float] = None,
        keep_top_n: Optional[int] = None,
        keep_active: bool = True,
    ) -> int:
        """Prune stale profiles and associated captures based on age and rank retention."""
        now = time.time()
        sessions_to_prune = set()

        with self.connection() as conn:
            if older_than_seconds is not None:
                cutoff = now - float(older_than_seconds)
                query = "SELECT session_id FROM people_profiles WHERE last_seen < ?"
                params: List[Any] = [cutoff]
                if keep_active:
                    query += " AND status != 'active'"
                rows = conn.execute(query, params).fetchall()
                for r in rows:
                    sessions_to_prune.add(r["session_id"])

            if keep_top_n is not None and keep_top_n >= 0:
                query = "SELECT session_id FROM people_profiles"
                params = []
                if keep_active:
                    query += " WHERE status != 'active'"
                query += " ORDER BY best_quality_score DESC, last_seen DESC LIMIT -1 OFFSET ?"
                params.append(int(keep_top_n))
                rows = conn.execute(query, params).fetchall()
                for r in rows:
                    sessions_to_prune.add(r["session_id"])

        deleted = 0
        for sid in sessions_to_prune:
            if self.delete_profile(sid):
                deleted += 1

        return deleted
